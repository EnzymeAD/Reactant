#!/usr/bin/env python3
"""Build Bazel artifacts and run Enzyme-GPU-Tests/LBM with resolved paths.

Usage:
  ./run_lbm_bazel.py
  ./run_lbm_bazel.py --jobs 96
  ./run_lbm_bazel.py --no-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> str:
    print("+", " ".join(cmd))
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            raise SystemExit(proc.returncode)
        return proc.stdout
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return ""


def first_output_file_from_cquery(enzyme_dir: Path, target: str, preferred_suffix: str | None = None) -> Path:
    out = run(
        [
            "bazel",
            "cquery",
            "--ui_event_filters=-INFO",
            "--noshow_progress",
            "--experimental_repo_remote_exec",
            "--output=files",
            target,
        ],
        cwd=enzyme_dir,
        capture=True,
    )
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"No output files reported for target {target}")
    selected = None
    if preferred_suffix is not None:
        for line in lines:
            if line.endswith(preferred_suffix):
                selected = line
                break
    if selected is None:
        selected = lines[0]

    candidate = Path(selected)
    if not candidate.is_absolute():
        candidate = enzyme_dir / candidate
    return candidate.resolve()


def bazel_bin_dir(enzyme_dir: Path) -> Path:
    out = run(
        [
            "bazel",
            "info",
            "--ui_event_filters=-INFO",
            "--noshow_progress",
            "bazel-bin",
        ],
        cwd=enzyme_dir,
        capture=True,
    )
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("Failed to resolve bazel-bin path")
    return Path(lines[-1]).resolve()


def validate_lbm_makefile(lbm_dir: Path) -> None:
    mk = lbm_dir / "Makefile"
    if not mk.exists():
        raise SystemExit(f"Missing Makefile: {mk}")
    content = mk.read_text(encoding="utf-8")

    missing_inputs = []
    if "main.cc" in content and not (lbm_dir / "main.cc").exists():
        missing_inputs.append("main.cc")
    if "parboil_cuda.c" in content and not (lbm_dir / "parboil_cuda.c").exists():
        missing_inputs.append("parboil_cuda.c")
    if "args.c" in content and not (lbm_dir / "args.c").exists():
        missing_inputs.append("args.c")
    if missing_inputs:
        raise SystemExit(
            "LBM Makefile references missing sources: "
            + ", ".join(missing_inputs)
            + f"\nFix {mk} source=... first."
        )

    if "-fno-experimental-new-pass-manager" in content:
        raise SystemExit(
            f"Found deprecated flag '-fno-experimental-new-pass-manager' in {mk}.\n"
            "Remove it from CFLAGS, then re-run this script."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bazel + LBM local runner")
    parser.add_argument(
        "--reactant",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to Reactant repo root (default: script directory)",
    )
    parser.add_argument(
        "--gpu-tests",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Enzyme-GPU-Tests",
        help="Path to Enzyme-GPU-Tests checkout",
    )
    parser.add_argument(
        "--cuda-path",
        type=str,
        default="/usr/local/cuda",
        help="CUDA root path",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 8)),
        help="Parallel jobs for make -j",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Build only (skip 'make run')",
    )
    args = parser.parse_args()

    reactant = args.reactant.resolve()
    enzyme_dir = reactant / "enzyme"
    lbm_dir = args.gpu_tests.resolve() / "LBM"

    if not enzyme_dir.exists():
        raise SystemExit(f"Missing enzyme workspace: {enzyme_dir}")
    if not lbm_dir.exists():
        raise SystemExit(f"Missing LBM dir: {lbm_dir}")

    validate_lbm_makefile(lbm_dir)

    # 1) Build all required artifacts.
    run(
        [
            "bazel",
            "build",
            "//:ReactantEnzymePlugin",
            "//:reactant-clang",
            "//:reactant-clang-resource",
            "@llvm-project//clang:clang",
            "@enzyme_ad//:libRaise.so",
        ],
        cwd=enzyme_dir,
    )

    # 2) Resolve paths.
    enzyme_path = first_output_file_from_cquery(
        enzyme_dir, "//:ReactantEnzymePlugin", preferred_suffix=".so"
    )
    clang_path = first_output_file_from_cquery(enzyme_dir, "//:reactant-clang")
    resource_dir = bazel_bin_dir(enzyme_dir) / "reactant-clang-resource"
    lib_raise_path = first_output_file_from_cquery(enzyme_dir, "@enzyme_ad//:libRaise.so")

    print("\nResolved paths:")
    print(f"  ENZYME_PATH={enzyme_path}")
    print(f"  CLANG_PATH={clang_path}")
    print(f"  CLANG_RESOURCE_DIR={resource_dir}")
    print(f"  LIB_RAISE_PATH={lib_raise_path}")

    # 3) Symbol sanity check.
    nm_out = run(["nm", "-D", str(lib_raise_path)], cwd=enzyme_dir, capture=True)
    if "runLLVMToMLIRRoundTrip" in nm_out:
        print("  libRaise symbol check: runLLVMToMLIRRoundTrip found")
    else:
        print("  WARNING: runLLVMToMLIRRoundTrip not found in libRaise.so")

    # 4) Build and run LBM.
    env = os.environ.copy()
    env["CUDA_PATH"] = args.cuda_path
    env["ENZYME_PATH"] = str(enzyme_path)
    env["CLANG_RESOURCE_DIR"] = str(resource_dir)
    env["CLANG_PATH"] = f"{clang_path} -resource-dir={resource_dir}"
    env["LIB_RAISE_PATH"] = str(lib_raise_path)

    run(
        [
            "make",
            "-j",
            str(args.jobs),
            f"CUDA_PATH={env['CUDA_PATH']}",
            f"CLANG_PATH={env['CLANG_PATH']}",
            f"ENZYME_PATH={env['ENZYME_PATH']}",
            f"LIB_RAISE_PATH={env['LIB_RAISE_PATH']}",
        ],
        cwd=lbm_dir,
        env=env,
    )
    if not args.no_run:
        run(["make", "run"], cwd=lbm_dir, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
