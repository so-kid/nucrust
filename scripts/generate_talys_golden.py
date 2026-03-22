#!/usr/bin/env python3
"""Generate TALYS golden reference data for Fe-56(n,gamma).

This script runs TALYS and parses its output to create golden reference
data files for nucrust validation (ACC-02: T_l, ACC-03: HF cross sections).

Requirements:
  - TALYS installed (or --talys-path specified)
  - Output directory: tests/reference_data/

Usage:
  python scripts/generate_talys_golden.py --talys-path /path/to/talys
  python scripts/generate_talys_golden.py --dry-run  # placeholder data
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tests" / "reference_data"


def generate_energy_grid():
    """Generate fine energy grid: 0.001 - 20 MeV (log-spaced, 100 points)."""
    energies = []
    log_min = math.log10(0.001)
    log_max = math.log10(20.0)
    n_points = 100
    for i in range(n_points):
        e = 10 ** (log_min + (log_max - log_min) * i / (n_points - 1))
        energies.append(e)
    return energies


def run_talys(talys_path, work_dir, input_text, energy_file_text):
    """Run TALYS in work_dir with given input."""
    input_path = os.path.join(work_dir, "input")
    energy_path = os.path.join(work_dir, "energies_fine.txt")
    with open(input_path, "w") as f:
        f.write(input_text)
    with open(energy_path, "w") as f:
        f.write(energy_file_text)
    result = subprocess.run(
        [talys_path],
        cwd=work_dir,
        stdin=open(input_path),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"TALYS failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return work_dir


def parse_talys_transmission(work_dir):
    """Parse TALYS transmission_n.out (YANDF-0.4 format).

    Returns list of dicts: [{energy_mev, transmission: {l=0: T, l=1: T, ...}}, ...]
    """
    trans_file = os.path.join(work_dir, "transmission_n.out")
    if not os.path.exists(trans_file):
        raise FileNotFoundError(f"TALYS did not produce {trans_file}")

    data = []
    current_energy = None
    current_entry = None

    with open(trans_file) as f:
        for line in f:
            line = line.strip()
            # Match energy header line
            m = re.match(r"#\s+energy \[MeV\]:\s+([\d.Ee+-]+)", line)
            if m:
                if current_entry is not None:
                    data.append(current_entry)
                current_energy = float(m.group(1))
                current_entry = {
                    "energy_mev": current_energy,
                    "transmission": {},
                }
                continue

            # Skip comment/header lines
            if line.startswith("#") or not line:
                continue

            # Data line: L  T(L-1/2,L)  T(L+1/2,L)  T(L)
            parts = line.split()
            if len(parts) >= 4 and current_entry is not None:
                try:
                    l_val = int(parts[0])
                    t_avg = float(parts[3])  # T(L) = spin-averaged
                    t_minus = float(parts[1])  # T(L-1/2, L)
                    t_plus = float(parts[2])  # T(L+1/2, L)
                    current_entry["transmission"][f"l={l_val}"] = t_avg
                    current_entry["transmission"][f"l={l_val}_j-"] = t_minus
                    current_entry["transmission"][f"l={l_val}_j+"] = t_plus
                except (ValueError, IndexError):
                    continue

    if current_entry is not None:
        data.append(current_entry)

    return data


def parse_talys_cross_section(work_dir):
    """Parse TALYS residual production file rp026057.tot for Fe-56(n,x)Fe-57 = (n,gamma).

    Returns list of dicts: [{energy_mev, cross_section_mb}, ...]
    """
    # Fe-56(n,gamma)Fe-57: Z=26, A=57
    xs_file = os.path.join(work_dir, "rp026057.tot")
    if not os.path.exists(xs_file):
        raise FileNotFoundError(f"TALYS did not produce {xs_file}")

    data = []
    in_data = False

    with open(xs_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("##") and "E" in line and "xs" in line:
                in_data = True
                continue
            if line.startswith("#"):
                continue
            if in_data and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        e = float(parts[0])
                        xs = float(parts[1])
                        data.append({
                            "energy_mev": e,
                            "cross_section_mb": xs,
                        })
                    except ValueError:
                        continue

    return data


def generate_placeholder_transmission(energies):
    """Generate approximate transmission coefficients using simple estimates."""
    data = []
    A = 56
    R = 1.25 * A ** (1.0 / 3.0)
    hbar2_over_2m = 20.736

    for e in energies:
        k = math.sqrt(e / hbar2_over_2m)
        l_max = k * R
        entry = {"energy_mev": e, "transmission": {}}
        for l in range(31):
            if l == 0:
                t = min(1.0, 4.0 * k * R / (1.0 + k * R) ** 2) if e > 0 else 0.0
            else:
                x = (l - l_max) / max(0.5, 0.3 * math.sqrt(l_max + 1))
                t = 1.0 / (1.0 + math.exp(x * 3.0))
            t *= min(1.0, (e / 1.0) ** (0.5 if l == 0 else l * 0.3))
            entry["transmission"][f"l={l}"] = max(0.0, min(1.0, t))
        data.append(entry)
    return data


def generate_placeholder_cross_section(energies):
    """Generate approximate Fe-56(n,gamma) cross sections."""
    data = []
    sigma_25kev = 11.7e-3
    e_ref = 0.025

    for e in energies:
        sigma_1v = sigma_25kev * math.sqrt(e_ref / e) if e > 0 else 0
        f_e = 1.0 / (1.0 + (e / 5.0) ** 2)
        sigma = sigma_1v * f_e
        sigma = max(0.0, min(10.0, sigma))
        data.append({"energy_mev": e, "cross_section_mb": sigma * 1000.0})
    return data


def write_transmission_dat(data, filepath, source="TALYS"):
    """Write transmission coefficient data in TSV format."""
    # Find max l across all entries
    all_l = set()
    for entry in data:
        for key in entry["transmission"]:
            if key.startswith("l=") and "_j" not in key:
                all_l.add(int(key.split("=")[1]))
    l_max = max(all_l) if all_l else 30

    with open(filepath, "w") as f:
        f.write("# Fe-56(n,x) neutron transmission coefficients T_l(E)\n")
        f.write(f"# Source: {source}\n")
        if source != "TALYS":
            f.write("# WARNING: Placeholder data - replace with actual TALYS output\n")
        f.write(f"# Generated: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"# Energy range: {data[0]['energy_mev']:.6E} - {data[-1]['energy_mev']:.6E} MeV\n")
        f.write(f"# Number of energies: {len(data)}\n")
        f.write(f"# l_max: {l_max}\n")
        f.write("#\n")
        header = "# E(MeV)"
        for l in range(l_max + 1):
            header += f"\tT_l={l}"
        f.write(header + "\n")
        for entry in data:
            line = f"{entry['energy_mev']:.6E}"
            for l in range(l_max + 1):
                t = entry["transmission"].get(f"l={l}", 0.0)
                line += f"\t{t:.6E}"
            f.write(line + "\n")


def write_cross_section_dat(data, filepath, source="TALYS"):
    """Write cross section data in TSV format."""
    with open(filepath, "w") as f:
        f.write("# Fe-56(n,gamma) cross section\n")
        f.write(f"# Source: {source}\n")
        if source != "TALYS":
            f.write("# WARNING: Placeholder data - replace with actual TALYS output\n")
        f.write(f"# Generated: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"# Energy range: {data[0]['energy_mev']:.6E} - {data[-1]['energy_mev']:.6E} MeV\n")
        f.write(f"# Number of energies: {len(data)}\n")
        f.write("#\n")
        f.write("# E(MeV)\tsigma(mb)\n")
        for entry in data:
            f.write(f"{entry['energy_mev']:.6E}\t{entry['cross_section_mb']:.6E}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate TALYS golden data")
    parser.add_argument("--talys-path", default="talys", help="Path to TALYS binary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder data without running TALYS",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    energies = generate_energy_grid()

    if args.dry_run:
        print("Generating placeholder reference data (TALYS not available)...")
        trans_data = generate_placeholder_transmission(energies)
        xs_data = generate_placeholder_cross_section(energies)
        source = "placeholder (systematics)"
    else:
        print("Running TALYS for Fe-56(n,gamma)...")
        energy_text = "\n".join(f"{e:.6f}" for e in energies) + "\n"

        talys_input = """\
projectile n
element Fe
mass 56
energy energies_fine.txt
outbasic y
outtransenergy y
transpower 12
outdiscrete y
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            run_talys(args.talys_path, tmpdir, talys_input, energy_text)
            print("Parsing TALYS transmission output...")
            trans_data = parse_talys_transmission(tmpdir)
            print(f"  Parsed {len(trans_data)} energy points")
            print("Parsing TALYS cross section output...")
            xs_data = parse_talys_cross_section(tmpdir)
            print(f"  Parsed {len(xs_data)} energy points")

        source = "TALYS-2.2 (Koning-Delaroche OMP, default NLD/GSF)"

    # Write output files
    trans_path = OUTPUT_DIR / "fe56_ng_transmission.dat"
    xs_path = OUTPUT_DIR / "fe56_ng_cross_section.dat"
    write_transmission_dat(trans_data, trans_path, source)
    write_cross_section_dat(xs_data, xs_path, source)

    print(f"Written: {trans_path}")
    print(f"Written: {xs_path}")

    # Find l_max from transmission data
    all_l = set()
    for entry in trans_data:
        for key in entry["transmission"]:
            if key.startswith("l=") and "_j" not in key:
                all_l.add(int(key.split("=")[1]))
    l_max = max(all_l) if all_l else 30

    # Write metadata
    meta = {
        "source": source,
        "talys_version": "2.2",
        "target": "Fe-56",
        "projectile": "n",
        "reaction": "(n,gamma)",
        "omp": "Koning-Delaroche (2003) global nucleon",
        "nld": "TALYS default (Gilbert-Cameron composite)",
        "gsf": "TALYS default (Kopecky-Uhl EGLO)",
        "energy_range_mev": [
            trans_data[0]["energy_mev"] if trans_data else energies[0],
            trans_data[-1]["energy_mev"] if trans_data else energies[-1],
        ],
        "n_energies_transmission": len(trans_data),
        "n_energies_cross_section": len(xs_data),
        "l_max": l_max,
    }
    meta_path = OUTPUT_DIR / "fe56_ng_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Written: {meta_path}")


if __name__ == "__main__":
    main()
