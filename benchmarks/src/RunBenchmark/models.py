#!/usr/bin/env python3
"""Model interface used by the benchmark runners, plus a Hugging Face implementation.

Benchmarks only talk to `Model`. To evaluate something other than a stock
transformers checkpoint, subclass `Model` and pass `--model-class pkg.module:Class`.
"""

import argparse
import gc
import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float = 0.0
    stop_strings: list[str] | None = None
    # Also stop at the first newline token (LongBench samsum).
    stop_at_newline: bool = False
    min_new_tokens: int | None = None


class Model(ABC):
    """What a benchmark needs from a model. Prompts are plain strings."""

    name: str = "model"

    @classmethod
    def add_args(cls, parser):
        """Register model-specific CLI flags."""

    @classmethod
    @abstractmethod
    def from_args(cls, args):
        """Build the model from parsed CLI args (`args.model` is the model name/path)."""

    def format_prompt(self, text, chat=True):
        """Wrap a user message in the model's chat format; identity by default."""
        return text

    @abstractmethod
    def num_tokens(self, text):
        """Token count of `text` exactly as `generate` would encode it."""

    def truncate_middle(self, text, max_tokens):
        """Keep the first and last `max_tokens // 2` tokens of `text`."""
        raise NotImplementedError(f"{type(self).__name__} does not support truncate_middle")

    @abstractmethod
    def generate(self, prompts, params):
        """Return one decoded continuation (without the prompt) per prompt."""

    def probe_batch_size(self, prompt_len, max_new_tokens, max_batch_size):
        """Largest batch that fits `prompt_len + max_new_tokens` tokens per row, or None if unsupported."""
        return None

    @staticmethod
    def is_oom(exc):
        """Whether `exc` is an out-of-memory error the runner may recover from by shrinking the batch."""
        try:
            import torch
        except ImportError:
            return False
        return isinstance(exc, torch.OutOfMemoryError)

    def free_memory(self):
        """Release cached allocator memory after an OOM."""


def add_model_args(parser):
    parser.add_argument("--model", required=True, help="Model name or path, interpreted by the model class.")
    parser.add_argument(
        "--model-class",
        default=None,
        help="Model implementation as pkg.module:Class (a Model subclass). Default: Hugging Face transformers.",
    )
    return parser


def resolve_model_class(spec):
    if spec is None:
        return HFModel
    module_name, sep, class_name = spec.partition(":")
    if not sep:
        raise SystemExit(f"--model-class must look like pkg.module:Class, got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (inspect.isclass(cls) and issubclass(cls, Model)):
        raise SystemExit(f"{spec} is not a subclass of RunBenchmark.models.Model")
    return cls


def model_class_from_argv(argv=None):
    """Pre-parse --model-class so its own flags can be added before the full parse."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--model-class", default=None)
    known, _ = pre.parse_known_args(argv)
    return resolve_model_class(known.model_class)


def load_model(args):
    return resolve_model_class(args.model_class).from_args(args)


class HFModel(Model):
    """Any causal LM loadable with transformers `from_pretrained`, sharded with accelerate."""

    # Chunk length used to grow the KV cache when probing the decode phase.
    PROBE_DECODE_CHUNK = 256

    def __init__(self, model, tokenizer, name, prefill_chunk_size=None):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.prefill_chunk_size = prefill_chunk_size or None
        self._probe_cache = {}

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
        parser.add_argument(
            "--device-map",
            choices=["auto", "balanced", "balanced_low_0", "sequential"],
            default="auto",
            help="accelerate device map; weights use each GPU's free memory (no per-GPU cap).",
        )
        parser.add_argument(
            "--cpu-memory",
            default=None,
            help='Allow offloading weights to CPU up to this size, e.g. "32GiB". Off by default.',
        )
        parser.add_argument("--attn-implementation", default="sdpa")
        parser.add_argument(
            "--prefill-chunk-size",
            type=int,
            default=4096,
            help="Prefill prompts longer than this in chunks of this many tokens (0 disables).",
        )
        parser.add_argument("--trust-remote-code", action="store_true")
        return parser

    @classmethod
    def from_args(cls, args):
        import torch
        from accelerate.utils import get_max_memory
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

        load_args = {
            "dtype": args.dtype if args.dtype == "auto" else getattr(torch, args.dtype),
            "device_map": args.device_map,
            "low_cpu_mem_usage": True,
            "attn_implementation": args.attn_implementation,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.cpu_memory is not None:
            max_memory = get_max_memory()
            max_memory["cpu"] = args.cpu_memory
            load_args["max_memory"] = max_memory

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        # Checkpoints with a vision tower load through the image-text-to-text auto class.
        multimodal = getattr(config, "vision_config", None) is not None
        model_cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        print(f"model: {args.model} ({model_cls.__name__}, {args.dtype}, device_map={args.device_map})", flush=True)
        model = model_cls.from_pretrained(args.model, **load_args).eval()

        placement = getattr(model, "hf_device_map", {})
        offloaded = sorted({str(d) for d in placement.values() if d in ("cpu", "disk")})
        if offloaded and args.cpu_memory is None:
            raise RuntimeError(
                f"Model partly offloaded to {offloaded}; generation would be very slow. "
                "Free GPU memory or pass --cpu-memory to allow offload explicitly."
            )

        # Text-only prompts: a multimodal processor's tokenizer is the same one.
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model, tokenizer, args.model, args.prefill_chunk_size)

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def format_prompt(self, text, chat=True):
        """Chat-formatted text; BOS is left out because the tokenizer adds it at encode time."""
        if not chat or not self.tokenizer.chat_template:
            return text
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        bos = self.tokenizer.bos_token
        if bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]
        return prompt

    def num_tokens(self, text):
        return len(self.tokenizer(text)["input_ids"])

    def truncate_middle(self, text, max_tokens):
        # Counted with special tokens, so BOS occupies a slot of the first half.
        ids = self.tokenizer(text)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        half = max_tokens // 2
        return (
            self.tokenizer.decode(ids[:half], skip_special_tokens=True)
            + self.tokenizer.decode(ids[-half:], skip_special_tokens=True)
        )

    def _eos_token_ids(self):
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            return [self.tokenizer.eos_token_id]
        return list(eos) if isinstance(eos, (list, tuple)) else [eos]

    def _generation_kwargs(self, params, input_len):
        kwargs = {"max_new_tokens": params.max_new_tokens}
        if params.temperature > 0:
            kwargs.update(do_sample=True, temperature=params.temperature)
        else:
            # Override sampling defaults from the checkpoint's generation_config.
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if params.stop_strings:
            kwargs.update(stop_strings=params.stop_strings, tokenizer=self.tokenizer)
        if params.stop_at_newline:
            newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
            kwargs["eos_token_id"] = self._eos_token_ids() + [newline_id]
        if params.min_new_tokens:
            kwargs["min_new_tokens"] = params.min_new_tokens
        if self.prefill_chunk_size and input_len > self.prefill_chunk_size:
            kwargs["prefill_chunk_size"] = self.prefill_chunk_size
        return kwargs

    def generate(self, prompts, params):
        import torch

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.input_device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self._generation_kwargs(params, input_len))
        return self.tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)

    def free_memory(self):
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def _simulate(self, batch_size, prompt_len, max_new_tokens):
        """Run the forward passes of a worst-case generation without decoding token by token.

        The prompt is prefilled exactly as `generate` would (chunked if long), then
        the KV cache is grown to the full output length in short chunks, whose
        activations are close to those of single-token decode steps.
        """
        import torch

        total = prompt_len + max_new_tokens
        ids = torch.full((batch_size, total), self.tokenizer.pad_token_id, device=self.input_device)
        # One padded position makes the mask non-trivial, as in a left-padded batch.
        mask = torch.ones_like(ids)
        mask[0, 0] = 0

        prefill_chunk = prompt_len
        if self.prefill_chunk_size and prompt_len > self.prefill_chunk_size:
            prefill_chunk = self.prefill_chunk_size
        bounds = list(range(prefill_chunk, prompt_len, prefill_chunk)) + [prompt_len]
        bounds += list(range(prompt_len + self.PROBE_DECODE_CHUNK, total, self.PROBE_DECODE_CHUNK)) + [total]

        extra = {"logits_to_keep": 1} if self.model._supports_logits_to_keep() else {}
        cache, start = None, 0
        with torch.inference_mode():
            for end in bounds:
                if end <= start:
                    continue
                out = self.model(
                    input_ids=ids[:, start:end],
                    attention_mask=mask[:, :end],
                    past_key_values=cache,
                    use_cache=True,
                    **extra,
                )
                cache = out.past_key_values
                del out
                start = end
        del cache, ids, mask

    def _fits(self, batch_size, prompt_len, max_new_tokens):
        import torch

        try:
            self._simulate(batch_size, prompt_len, max_new_tokens)
            ok = True
        except torch.OutOfMemoryError:
            ok = False
        self.free_memory()
        return ok

    def probe_batch_size(self, prompt_len, max_new_tokens, max_batch_size):
        """Doubling then binary search; returns 0 if even batch size 1 does not fit."""
        key = (prompt_len, max_new_tokens, max_batch_size)
        if key in self._probe_cache:
            return self._probe_cache[key]

        def fits(b):
            ok = self._fits(b, prompt_len, max_new_tokens)
            print(f"  probe batch={b:<4} len={prompt_len}+{max_new_tokens}: {'ok' if ok else 'OOM'}", flush=True)
            return ok

        best = 0
        if fits(1):
            best, failed = 1, None
            while best < max_batch_size:
                trial = min(best * 2, max_batch_size)
                if fits(trial):
                    best = trial
                else:
                    failed = trial
                    break
            if failed is not None:
                lo, hi = best, failed
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid
                best = lo
        self._probe_cache[key] = best
        return best


def main():
    parser = argparse.ArgumentParser(description="Generate a reply to one prompt (quick model check).")
    add_model_args(parser)
    model_class_from_argv().add_args(parser)
    parser.add_argument("--prompt", default="Introduce yourself in 100 words.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-chat", action="store_true", help="Send the prompt without the chat template.")
    args = parser.parse_args()

    model = load_model(args)
    [answer] = model.generate(
        [model.format_prompt(args.prompt, chat=not args.no_chat)],
        GenerationParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature),
    )
    if hasattr(model, "model") and hasattr(model.model, "hf_device_map"):
        print("\n=== device map ===")
        print(model.model.hf_device_map)
    print("\n=== answer ===")
    print(answer)


if __name__ == "__main__":
    main()
