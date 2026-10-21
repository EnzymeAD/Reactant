import argparse
import array
import json
import os
import re
import subprocess
import sys

# One minute less than the step's timeout-minutes, so the run is stopped with
# a message instead of the step being killed.
RUN_TIMEOUT = "14m"

# Largest allowed absolute difference from reference.dat. Reactant and plain
# clang builds both land within 1.2e-6 of it; velocities reach 2.4e-2.
LBM_TOLERANCE = 1e-5

# LULESH -s 45 runs 10 iterations (to t=0.01). Reactant and a plain clang 19
# CUDA build without Reactant both print this value.
LULESH_ENERGY = 2.366301e07
LULESH_TOLERANCE = 1e-6

# Where each benchmark prints its own timing: (pattern, scale, unit, extra).
TIMINGS = {
    "lbm": (r"nt: \d+ took (\d+) us", 1e-6, "s", "100 steps"),
    "xsbench": (r"Runtime: +([\d.]+) seconds", 1, "s", "-m event"),
    "rsbench": (r"Runtime: +([\d.]+) seconds", 1, "s", "-m event"),
    "lulesh": (r"Grind time \(us/z/c\) *= *([\d.eE+-]+)", 1, "us/zone/cycle", "-s 45"),
}


def run(args, cwd):
    """Runs a benchmark, streaming its output. Returns (output, exit status)."""
    proc = subprocess.Popen(
        ["timeout", "--signal=TERM", "--verbose", RUN_TIMEOUT, *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return "".join(lines), proc.wait()


def status_error(program, status):
    if status == 124:
        return f"{program} timed out after {RUN_TIMEOUT}"
    if status < 0:
        return f"{program} was killed by signal {-status}"
    return f"{program} exited with status {status}"


def load_floats(path):
    values = array.array("f")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    return values


def check_lbm(root):
    cwd = os.path.join(root, "LBM")
    data = "datasets/lbm/short"
    output, status = run(
        ["./lbm", "-i", f"{data}/input/120_120_150_ldc.of", "-o", "out.dat", "--", "100"],
        cwd,
    )
    if status:
        return output, status_error("lbm", status)
    out = load_floats(os.path.join(cwd, "out.dat"))
    ref = load_floats(os.path.join(cwd, data, "output/reference.dat"))
    if len(out) != len(ref):
        return output, f"wrote {len(out)} values, reference.dat has {len(ref)}"
    worst, bad = 0.0, 0
    for x, y in zip(out, ref):
        d = abs(x - y)
        if not d <= LBM_TOLERANCE:  # also counts NaN
            bad += 1
        if d > worst:
            worst = d
    print(f"Max difference from reference.dat: {worst:.3e} (tolerance {LBM_TOLERANCE:g})")
    if bad:
        return output, f"{bad} of {len(ref)} values differ from reference.dat by more than {LBM_TOLERANCE:g}"
    return output, None


def check_checksum(root, subdir, program):
    output, status = run([f"./{program}", "-m", "event"], os.path.join(root, subdir))
    match = re.search(r"Verification checksum: (\d+) \((.*)\)", output)
    if match and match.group(2) != "Valid":
        return output, f"verification checksum {match.group(1)} is invalid"
    if status:
        return output, status_error(program, status)
    if not match:
        return output, "no verification checksum in the output"
    return output, None


def check_lulesh(root):
    output, status = run(["./lulesh", "-s", "45"], os.path.join(root, "LULESH"))
    if status:
        return output, status_error("lulesh", status)
    match = re.search(r"Final Origin Energy = *(\S+)", output)
    if not match:
        return output, "no final origin energy in the output"
    energy = float(match.group(1))  # "-nan" parses as nan
    if not abs(energy - LULESH_ENERGY) <= LULESH_TOLERANCE * LULESH_ENERGY:
        return output, f"final origin energy {energy:e}, expected {LULESH_ENERGY:e}"
    return output, None


CHECKS = {
    "lbm": ("LBM", check_lbm),
    "xsbench": ("XSBench", lambda root: check_checksum(root, "XSBench", "XSBench")),
    "rsbench": ("RSBench", lambda root: check_checksum(root, "RSBench", "rsbench")),
    "lulesh": ("LULESH", check_lulesh),
}


def record_timing(results_dir, benchmark, name, device, output):
    pattern, scale, unit, extra = TIMINGS[benchmark]
    match = re.search(pattern, output)
    if not match:
        print(f"::warning title={name}::no timing in the output, nothing recorded")
        return
    record = {
        "name": f"{name} / {device}",
        "unit": unit,
        "value": float(match.group(1)) * scale,
        "extra": extra,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{benchmark}.json"), "w") as f:
        json.dump([record], f, indent=2)
    print(f"Recorded {record['name']}: {record['value']:g} {unit}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", choices=CHECKS)
    parser.add_argument("dir")
    parser.add_argument("--results", help="directory to write the timing record to")
    parser.add_argument("--device", default="local", help="device name for the record")
    args = parser.parse_args()

    name, check = CHECKS[args.benchmark]
    output, error = check(args.dir)
    if error:
        print(f"::error title={name}::{error}")
        sys.exit(1)
    print(f"{name}: OK")
    if args.results:
        record_timing(args.results, args.benchmark, name, args.device, output)


if __name__ == "__main__":
    main()
