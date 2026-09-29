"""Build Mamba's selective-scan extension in place.

PyTorch normally rejects CUDA toolkit/PyTorch CUDA major-version mismatches.
The opt-in flag below bypasses only that guard; it does not change the compiler,
headers, or runtime linked into the extension.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import warnings

import torch
import torch.utils.cpp_extension as cpp_extension
from packaging.version import Version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-major-mismatch",
        action="store_true",
        help="permit a local CUDA compiler/PyTorch CUDA major-version mismatch",
    )
    args = parser.parse_args()

    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME is not set and nvcc was not found")
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    version_text = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release\s+(\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError(f"Could not parse CUDA compiler version from {nvcc}")
    compiler_cuda = Version(match.group(1))
    torch_cuda = Version(torch.version.cuda)
    if compiler_cuda.major != torch_cuda.major:
        if not args.allow_major_mismatch:
            raise RuntimeError(
                f"nvcc CUDA {compiler_cuda} does not match PyTorch CUDA {torch_cuda}; "
                "install a matching CUDA toolkit or pass --allow-major-mismatch "
                "after reviewing the compatibility risk"
            )
        warnings.warn(
            f"Building with nvcc CUDA {compiler_cuda} against PyTorch CUDA "
            f"{torch_cuda}; runtime compatibility must be checked on this GPU."
        )
        cpp_extension._check_cuda_version = lambda compiler, version: None

    os.environ["MAMBA_FORCE_BUILD"] = "TRUE"
    os.environ["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
    os.environ["PATH"] = f"{sys.prefix}/bin{os.pathsep}{os.environ.get('PATH', '')}"
    sys.argv = ["setup.py", "build_ext", "--inplace"]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()
