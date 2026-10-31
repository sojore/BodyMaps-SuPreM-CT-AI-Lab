# Audited SuPreM inference path

This document records the configuration used for the BodyMaps browser-triggered inference demonstration on 31 October 2026.

## Environment

- Google Colab, NVIDIA Tesla T4.
- Isolated Python 3.9.25 environment created with `uv`.
- PyTorch 1.11.0+cu113.
- MONAI 0.9.0.
- NumPy 1.24.4.
- Streamlit 1.49.1 for the Python 3.9 web layer.
- SuPreM commit: `3e729951c61f729cd2fa6ceefa8e021bcfcd4985`.

## Checkpoint

Checkpoint: `supervised_suprem_unet_2100.pth`

Source: `https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth`

Expected SHA-256:

`c8dc39c86cd8d715e5ef0b5bb4ba4335ba45237127829ea016f75ee8beca0a93`

The file hash is checked **before** `torch.load` because PyTorch 1.11 does not provide the newer `weights_only` loading path.

## Strict checkpoint audit

Upstream SuPreM direct inference assigns checkpoint tensors by iteration order. For this demonstration, that block is replaced in a separate copy of the inference script with a strict name-based loader.

The audit:
1. verifies checkpoint SHA-256;
2. loads `checkpoint['net']` only after the hash passes;
3. strips only recognized `module.` / `model.` prefixes;
4. compares model/checkpoint key sets;
5. checks tensor shapes and dtypes;
6. rejects normalized-key collisions;
7. calls `model.load_state_dict(..., strict=True)`;
8. fails closed if any check differs.

Observed audit result:

```text
audit_pass: true
model_tensor_count: 117
checkpoint_tensor_count: 117
missing: []
unexpected: []
duplicate_normalized_keys: []
wrong_shape: []
wrong_dtype: []
```

## Inference command

The browser backend launches the same audited inference path in an isolated subprocess:

```bash
/content/suprem_env/bin/python -W ignore inference_bodymaps_verified.py \
  --save_dir <temporary-output-dir> \
  --checkpoint <verified-checkpoint> \
  --data_root_path <temporary-input-dir> \
  --backbone unet --store_result --suprem \
  --batch_size 1 --num_workers 0 \
  --roi_x 96 --roi_y 96 --roi_z 96 \
  --overlap 0.25
```

The subprocess must exit 0 and produce all nine expected NIfTI masks before the browser receives a result ZIP.

## Browser integration

`inference_backend.py` accepts either a `.nii.gz` CT or a ZIP containing exactly one file named `ct.nii.gz`. ZIP uploads are checked for uncompressed size and path traversal. Each request uses a private temporary directory.

The environment variable `SUPREM_PYTHON` must remain the virtual-environment path, e.g. `/content/suprem_env/bin/python`. Do **not** call `Path.resolve()` on that interpreter path, because resolving the venv symlink can select the underlying base interpreter and bypass the environment where PyTorch/MONAI are installed.

## Demonstrated cases

| Case | Shape | Mean Dice | Mean IoU |
|---|---:|---:|---:|
| BDMAP_00000338 | (502, 348, 71) | 0.9252 | 0.8639 |
| BDMAP_00000015 | (502, 342, 33) | 0.8930 | 0.8301 |
| BDMAP_00000018 | (238, 178, 170) | 0.8882 | 0.8172 |

These values are case-level technical checks against provided reference masks. They are not claimed as independent benchmark performance because training/pretraining overlap has not been established.
