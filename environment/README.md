# `pet-core` unified Python environment

This directory describes the single Python environment used by the desktop-pet project.

## Fixed runtime

- Conda environment: `pet-core`
- Python: `3.10.20`
- PyTorch: `2.10.0+cu130`
- TorchVision: `0.25.0+cu130`
- TorchAudio: `2.10.0+cu130`
- GPU target: RTX 5070 Ti Laptop / compute capability `sm_120`

PyTorch is intentionally absent from `requirements.txt`.  Its exact versions are present only in
`constraints.txt`, so pip may validate them but may not replace the manually installed CUDA wheels.

## Install and verify

Run from the workspace root in PowerShell:

```powershell
conda activate pet-core
python -m pip install -r environment/requirements.txt -c environment/constraints.txt
python environment/smoke_test.py
```

The smoke test is offline: it does not download model weights.  It checks the Python package graph,
CUDA execution, TorchVision custom CUDA operators, ONNX Runtime, and lightweight forward/import paths
through the upstream repositories.

`jsonschema==4.23.0` is included so PET's cross-language NDJSON tests always
run full Draft 2020-12 validation instead of their codec-only fallback.

`requirements.lock.txt` is the exact installed snapshot for auditing. It includes the manually
installed CUDA Torch wheels, so use it as a reference rather than feeding it directly to a default
PyPI index.

## Deliberate exclusions

The following are not part of the unified environment:

- TensorRT, `cuda-python`, ONNX Runtime GPU, and StreamDiffusion `stable-fast`;
- PyTorch3D, Mujoco/D4RL, old `gym==0.21`, Chumpy, Open3D, and old spaCy;
- bitsandbytes, Triton-specific repository extensions, and the legacy pinned Torch wheels;
- xFormers for the first baseline. PyTorch 2.10 SDPA is used until a matching Windows/CUDA 13/
  `sm_120` wheel is validated separately.

Do not install any upstream `requirements.txt` wholesale into `pet-core`; most of them contain an old
Torch pin.  For now the repositories are loaded directly from their source directories by the smoke
test.  Once a base is selected, we can package only that adapter cleanly instead of registering
conflicting upstream metadata.
