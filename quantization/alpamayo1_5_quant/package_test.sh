echo "== venv ==" ; which python ; python --version
echo "== driver / GPU ==" ; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "== nvcc ==" ; which nvcc ; nvcc --version | tail -2 ; echo "CUDA_HOME=$CUDA_HOME"
echo "== gcc ==" ; gcc --version | head -1
echo "== env flags ==" ; echo "FLASH_ATTENTION_FORCE_BUILD=$FLASH_ATTENTION_FORCE_BUILD FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS VIRTUAL_ENV=$VIRTUAL_ENV UV_NO_SYNC=$UV_NO_SYNC"
echo "== RAM ==" ; free -g | head -2
echo "== python packages ==" ; python - <<'EOF'
import importlib, warnings
warnings.filterwarnings("ignore")
import torch
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}  gpu_cap={torch.cuda.get_device_capability()}")
for mod, attr in [("torchvision","__version__"),("numpy","__version__"),("transformers","__version__"),
                  ("modelopt","__version__"),("tensorrt","__version__"),("tensorrt_llm","__version__"),
                  ("flash_attn","__version__"),("ninja","__version__")]:
    try:
        m = importlib.import_module(mod); print(f"{mod:<14} {getattr(m, attr, '?')}")
    except Exception as e:
        print(f"{mod:<14} FAILED: {type(e).__name__}: {str(e)[:150]}")
for mod in ["flash_attn_2_cuda", "tensorrt_llm._torch"]:
    try:
        importlib.import_module(mod); print(f"{mod:<20} import OK")
    except Exception as e:
        print(f"{mod:<20} FAILED: {str(e)[:150]}")
EOF
echo "== dependency conflicts ==" ; uv pip check --python "$(which python)"
