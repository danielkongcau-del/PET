"""Offline acceptance test for the unified pet-core environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "pet-core-huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
EXPECTED = {
    "torch": "2.10.0+cu130",
    "torchvision": "0.25.0+cu130",
    "torchaudio": "2.10.0+cu130",
    "numpy": "1.26.4",
    "diffusers": "0.25.0",
    "huggingface-hub": "0.25.2",
    "transformers": "4.38.2",
    "setuptools": "80.9.0",
    "lightning": "2.1.4",
    "pytorch-lightning": "2.1.4",
    "torchmetrics": "1.3.2",
    "jsonschema": "4.23.0",
}


class Results:
    def __init__(self) -> None:
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, fn: Callable[[], object]) -> None:
        try:
            detail = fn()
            suffix = "" if detail is None else f" - {detail}"
            print(f"[PASS] {name}{suffix}")
        except Exception as exc:  # noqa: BLE001 - this is an acceptance harness
            self.failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name} - {type(exc).__name__}: {exc}")
            traceback.print_exc()


def check_versions() -> str:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    if actual != EXPECTED:
        raise AssertionError(f"version drift: {actual!r}")
    return json.dumps(actual, ensure_ascii=False)


def check_imports() -> str:
    modules = [
        "accelerate",
        "ale_py",
        "av",
        "cv2",
        "diffusers",
        "einops",
        "gymnasium",
        "h5py",
        "hydra",
        "imageio",
        "jsonschema",
        "lightning",
        "matplotlib",
        "numpy",
        "omegaconf",
        "onnx",
        "onnxruntime",
        "pandas",
        "peft",
        "pygame",
        "sentence_transformers",
        "torchmetrics",
        "torcheval",
        "transformers",
        "wandb",
    ]
    for module in modules:
        importlib.import_module(module)
    return f"{len(modules)} modules"


def check_pip() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr).strip())
    return proc.stdout.strip()


def check_denylist() -> str:
    from packaging.utils import canonicalize_name

    installed = {
        canonicalize_name(distribution.metadata["Name"])
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    forbidden = {
        canonicalize_name(name)
        for name in (
            "bitsandbytes",
            "chumpy",
            "cuda-python",
            "d4rl",
            "free-mujoco-py",
            "gym",
            "mujoco-py",
            "onnx-graphsurgeon",
            "polygraphy",
            "pytorch3d",
            "tensorrt",
        )
    }
    bad = sorted(installed & forbidden)
    if bad:
        raise AssertionError(f"forbidden baseline packages: {bad}")
    ort = installed & {
        canonicalize_name("onnxruntime"),
        canonicalize_name("onnxruntime-gpu"),
        canonicalize_name("onnxruntime-directml"),
    }
    if len(ort) != 1:
        raise AssertionError(f"expected exactly one ONNX Runtime distribution, found {sorted(ort)}")
    opencv = installed & {
        canonicalize_name("opencv-python"),
        canonicalize_name("opencv-contrib-python"),
        canonicalize_name("opencv-python-headless"),
        canonicalize_name("opencv-contrib-python-headless"),
    }
    if len(opencv) != 1:
        raise AssertionError(f"expected exactly one OpenCV distribution, found {sorted(opencv)}")
    return "legacy CUDA/robotics stacks absent"


def check_cuda() -> str:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(f"unexpected capability {torch.cuda.get_device_capability(0)}")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"sm_120 missing from {torch.cuda.get_arch_list()}")

    left = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    right = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    product = left @ right
    query = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float16)
    attended = functional.scaled_dot_product_attention(query, query, query)
    torch.cuda.synchronize()
    if not torch.isfinite(product).all() or not torch.isfinite(attended).all():
        raise FloatingPointError("non-finite CUDA result")
    return f"{torch.cuda.get_device_name(0)}, sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"


def check_torchvision_cuda_ops() -> str:
    import torch
    from torchvision.ops import nms

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    kept = nms(boxes, scores, 0.5).cpu().tolist()
    if kept != [0, 2]:
        raise AssertionError(f"unexpected NMS result {kept}")
    return "torchvision::nms CUDA kernel"


def check_torchaudio() -> str:
    import torch
    import torchaudio

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=400)(torch.randn(1, 16000))
    if tuple(spectrogram.shape) != (1, 201, 81):
        raise AssertionError(spectrogram.shape)
    return "Spectrogram"


def check_diffusers() -> str:
    import torch
    from diffusers import LCMScheduler

    scheduler = LCMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(4, device=torch.device("cuda"))
    if scheduler.timesteps.numel() != 4 or not scheduler.timesteps.is_cuda:
        raise AssertionError("LCM scheduler did not create CUDA timesteps")
    return "LCMScheduler"


def check_onnxruntime() -> str:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("Add", ["x", "y"], ["out"])], "add", [x_info, y_info], [out_info])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    value = np.array([[1.0, 2.0]], dtype=np.float32)
    result = session.run(None, {"x": value, "y": value})[0]
    if not np.array_equal(result, value * 2):
        raise AssertionError(result)
    return f"onnxruntime {ort.__version__} CPUExecutionProvider"


def source_check(name: str, source: Path, code: str, timeout: int = 120) -> str:
    env = os.environ.copy()
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source) if not old_path else os.pathsep.join((str(source), old_path))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(source),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"{name}: {(proc.stdout + proc.stderr).strip()}")
    detail = " | ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())
    return detail or "source import/forward"


def main() -> int:
    results = Results()
    results.check("pinned versions", check_versions)
    results.check("pip dependency graph", check_pip)
    results.check("dependency denylist", check_denylist)
    results.check("shared Python imports", check_imports)
    results.check("CUDA matmul and SDPA", check_cuda)
    results.check("TorchVision CUDA ABI", check_torchvision_cuda_ops)
    results.check("TorchAudio ABI", check_torchaudio)
    results.check("Diffusers scheduler", check_diffusers)
    results.check("ONNX Runtime", check_onnxruntime)

    results.check(
        "TAESD forward",
        lambda: source_check(
            "TAESD",
            ROOT / "third_party/pixel/taesd",
            "import torch; from taesd import TAESD; "
            "m=TAESD().cuda().eval(); image=torch.rand(1,3,64,64,device='cuda'); "
            "z=m.encoder(image); y=m.decoder(z); "
            "assert tuple(z.shape)==(1,4,8,8) and tuple(y.shape)==(1,3,64,64); "
            "print(tuple(z.shape),tuple(y.shape))",
        ),
    )
    results.check(
        "MaskGIT forward",
        lambda: source_check(
            "MaskGIT",
            ROOT / "third_party/pixel/maskgit-pytorch",
            "import torch; from Network.transformer import Transformer; "
            "m=Transformer(input_size=4,hidden_dim=32,codebook_size=16,depth=1,heads=4,mlp_dim=64,nclass=3,register=0).cuda().eval(); "
            "x=torch.randint(0,16,(1,4,4),device='cuda'); y=m(x,torch.tensor([1],device='cuda'),torch.tensor([False],device='cuda')); "
            "assert tuple(y.shape)==(1,16,17); print(tuple(y.shape))",
        ),
    )
    results.check(
        "DIAMOND world-model forward",
        lambda: source_check(
            "DIAMOND",
            ROOT / "third_party/pixel/diamond/src",
            "import torch; from models.diffusion.inner_model import InnerModel,InnerModelConfig; "
            "c=InnerModelConfig(img_channels=3,num_steps_conditioning=1,cond_channels=8,depths=[1],channels=[8],attn_depths=[False],num_actions=4); "
            "m=InnerModel(c).cuda().eval(); a=torch.randn(1,3,8,8,device='cuda'); "
            "y=m(a,torch.ones(1,device='cuda'),a,torch.zeros(1,1,dtype=torch.long,device='cuda')); "
            "assert tuple(y.shape)==(1,3,8,8); print(tuple(y.shape))",
        ),
    )
    results.check(
        "CAMDM motion forward",
        lambda: source_check(
            "CAMDM",
            ROOT / "third_party/motion/camdm/PyTorch",
            "import torch; from network.models import MotionDiffusion; "
            "m=MotionDiffusion(6,3,2,3,'6d',4,latent_dim=32,ff_size=64,num_layers=1,num_heads=4,dropout=0).cuda().eval(); "
            "x=torch.randn(1,2,3,4,device='cuda'); p=torch.randn(1,2,3,2,device='cuda'); "
            "tp=torch.randn(1,6,2,device='cuda'); tt=torch.randn(1,2,2,device='cuda'); "
            "y=m(x,torch.tensor([1],device='cuda'),p,tp,tt,torch.tensor([0],device='cuda')); "
            "assert tuple(y.shape)==(1,2,3,4); print(tuple(y.shape))",
        ),
    )
    results.check(
        "MotionLCM compatibility imports",
        lambda: source_check(
            "MotionLCM",
            ROOT / "third_party/motion/motionlcm",
            "import torch; from mld.models.schedulers.scheduling_lcm import LCMScheduler; "
            "from mld.models.architectures.mld_clip import MldTextEncoder; "
            "from mld.models.architectures.mld_denoiser import MldDenoiser; "
            "s=LCMScheduler(num_train_timesteps=1000); s.set_timesteps(4,device='cuda'); "
            "m=MldDenoiser(latent_dim=[1,16],text_dim=8,time_dim=16,ff_size=32,num_layers=1,num_heads=4,dropout=0,arch='trans_enc').cuda().eval(); "
            "sample=torch.randn(2,4,16,device='cuda'); cond=torch.randn(2,3,8,device='cuda'); "
            "out,loss=m(sample,torch.tensor([10],device='cuda'),cond); "
            "assert s.timesteps.is_cuda and tuple(out.shape)==tuple(sample.shape) and loss is None; "
            "print(tuple(out.shape),MldTextEncoder.__name__)",
        ),
    )
    results.check(
        "Diffusion Forcing model imports",
        lambda: source_check(
            "Diffusion Forcing",
            ROOT / "third_party/motion/diffusion-forcing",
            "import importlib.util,pathlib,torch; p=pathlib.Path('algorithms/diffusion_forcing/models/transformer.py'); "
            "s=importlib.util.spec_from_file_location('pet_df_transformer',p); mod=importlib.util.module_from_spec(s); s.loader.exec_module(mod); "
            "m=mod.Transformer(x_dim=4,size=32,num_layers=1,nhead=4,dim_feedforward=64).cuda().eval(); "
            "x=torch.randn(5,2,4,device='cuda'); k=torch.randint(0,10,(5,2),device='cuda'); y=m(x,k); "
            "assert tuple(y.shape)==tuple(x.shape); print(tuple(y.shape))",
        ),
    )
    results.check(
        "StreamDiffusion import",
        lambda: source_check(
            "StreamDiffusion",
            ROOT / "third_party/pixel/streamdiffusion/src",
            "from streamdiffusion import StreamDiffusion; print(StreamDiffusion.__name__)",
        ),
    )
    results.check(
        "Live2Diff import",
        lambda: source_check(
            "Live2Diff",
            ROOT / "third_party/pixel/live2diff",
            "from live2diff import StreamAnimateDiffusionDepth; print(StreamAnimateDiffusionDepth.__name__)",
        ),
    )

    try:
        version = importlib.metadata.version("xformers")
    except importlib.metadata.PackageNotFoundError:
        print("[SKIP] xFormers - using PyTorch 2.10 SDPA baseline")
    else:
        print(f"[INFO] xFormers {version} is installed; validate its CUDA kernels separately")

    print()
    if results.failed:
        print(f"FAILED: {len(results.failed)} check(s)")
        for name, detail in results.failed:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL REQUIRED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
