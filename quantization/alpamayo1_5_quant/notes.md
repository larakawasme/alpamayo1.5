## quantize.py

Need custom device_map to ensure that (vlm layer i, expert layer i) always land on the same GPU when quantizing using multiple GPUs. Added to utils.py

```
PYTORCH_ALLOC_CONF=expandable_segments:True python quantize.py --quant_format nvfp4 --save_model_dir ../alpamayo1_5_quant_model/ --num_traj_samples 1 --num_of_calib_clips 2
```

## eval.py
```
python eval.py --ckpt ../alpamayo1_5_quant_model/alpamayo1.5_nvfp4_calib100/ --limit 10 --num_traj_samples 1
```

## NSight Systems

```
nsys profile \
  -o eval_nvfp4 \
  -t cuda,nvtx,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --force-overwrite=true \
  python eval.py --profile --ckpt ../alpamayo1_5_quant_model/alpamayo1.5_nvfp4_calib100/ --limit 3 --num_traj_samples 1
```

## inference_all_clips.py
