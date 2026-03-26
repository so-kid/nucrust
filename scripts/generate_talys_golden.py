#!/usr/bin/env python3
"""Generate TALYS golden reference data for Fe-56(n,gamma) and U-238(n,gamma).

This script generates TALYS input files and processes output to create
golden reference data files for nucrust validation (B1, B2, B3 benchmarks).

Requirements:
  - TALYS installed and available in PATH
  - TALYS structure database installed (8 GB)
  - Output directory: tests/reference_data/talys/

Usage:
  python scripts/generate_talys_golden.py --talys-path /path/to/talys [--dry-run]

Benchmarks generated:
  B1: Fe-56(n,x) neutron transmission coefficients T_l(E)
  B2: Fe-56(n,gamma) Hauser-Feshbach cross sections
  B3: U-238(n,gamma) HF cross sections (deformed nucleus stress test)
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tests" / "reference_data" / "talys"


def generate_energy_grid(n_points=100, e_min=0.001, e_max=20.0):
    """Generate log-spaced energy grid."""
    energies = []
    log_min = math.log10(e_min)
    log_max = math.log10(e_max)
    for i in range(n_points):
        e = 10 ** (log_min + (log_max - log_min) * i / (n_points - 1))
        energies.append(e)
    return energies


def run_talys(talys_path, work_dir, input_text, energy_file_text):
    """Run TALYS in work_dir with given input. Returns stdout."""
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
        timeout=1800,  # 30 min max
    )
    if result.returncode != 0:
        print(f"TALYS failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


# ============================================================
# TALYS Output Parsers
# ============================================================

def parse_transmission_file(filepath):
    """Parse TALYS transmission_n.out (YANDF format).

    Returns list of dicts: [{"energy_mev": float, "T": {l: {"T_lminus": f, "T_lplus": f, "T_avg": f}}}]
    """
    with open(filepath) as f:
        content = f.read()

    blocks = content.split("energy [MeV]:")
    data = []
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        energy = float(lines[0].strip())
        entry = {"energy_mev": energy, "T": {}}
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    l_val = int(parts[0])
                    t_minus = float(parts[1])
                    t_plus = float(parts[2])
                    t_avg = float(parts[3])
                    entry["T"][l_val] = {
                        "T_lminus": t_minus,
                        "T_lplus": t_plus,
                        "T_avg": t_avg,
                    }
                except (ValueError, IndexError):
                    continue
        if entry["T"]:
            data.append(entry)
    return data


def parse_binary_cross_sections(filepath):
    """Parse TALYS binary.tot for channel cross sections.

    Returns list of dicts with energy and per-channel cross sections (mb).
    """
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    e = float(parts[0])
                    data.append({
                        "energy_mev": e,
                        "sigma_gamma_mb": float(parts[1]),
                        "sigma_neutron_mb": float(parts[2]),
                        "sigma_proton_mb": float(parts[3]),
                        "sigma_deuteron_mb": float(parts[4]),
                        "sigma_triton_mb": float(parts[5]),
                        "sigma_helion_mb": float(parts[6]),
                        "sigma_alpha_mb": float(parts[7]),
                    })
                except ValueError:
                    continue
    return data


def parse_all_cross_sections(filepath):
    """Parse TALYS all.tot for total, elastic, and reaction cross sections.

    Returns list of dicts with multiple cross section types (mb).
    """
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    e = float(parts[0])
                    data.append({
                        "energy_mev": e,
                        "sigma_nonelastic_mb": float(parts[1]),
                        "sigma_elastic_mb": float(parts[2]),
                        "sigma_total_mb": float(parts[3]),
                        "sigma_compound_elastic_mb": float(parts[4]),
                        "sigma_shape_elastic_mb": float(parts[5]),
                        "sigma_reaction_mb": float(parts[6]),
                    })
                except ValueError:
                    continue
    return data


# ============================================================
# Writers
# ============================================================

def write_transmission_dat(data, filepath, source="TALYS-2.2", l_max=30):
    """Write transmission coefficient data in TSV format."""
    with open(filepath, "w") as f:
        f.write(f"# Fe-56(n,x) neutron transmission coefficients T_l(E)\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Optical model: Koning-Delaroche (2003) global neutron OMP\n")
        f.write(f"# Format: T(L) = spin-averaged transmission coefficient\n")
        f.write(f"# T(L-1/2,L) and T(L+1/2,L) are spin-orbit split components\n")
        f.write(f"#\n")

        # Header for averaged T_l
        header = "# E(MeV)"
        for l in range(l_max + 1):
            header += f"\tT_l={l}"
        f.write(header + "\n")

        for entry in data:
            line = f"{entry['energy_mev']:.6E}"
            for l in range(l_max + 1):
                if l in entry["T"]:
                    t = entry["T"][l]["T_avg"]
                else:
                    t = 0.0
                line += f"\t{t:.6E}"
            f.write(line + "\n")


def write_transmission_spinorbit_dat(data, filepath, source="TALYS-2.2", l_max=30):
    """Write spin-orbit resolved transmission coefficients."""
    with open(filepath, "w") as f:
        f.write(f"# Fe-56(n,x) neutron transmission coefficients T(j,l) with spin-orbit splitting\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Columns: E(MeV), then for each l: T(l-1/2,l), T(l+1/2,l), T_avg(l)\n")
        f.write(f"#\n")

        header = "# E(MeV)"
        for l in range(l_max + 1):
            header += f"\tT(l-1/2,{l})\tT(l+1/2,{l})\tT_avg({l})"
        f.write(header + "\n")

        for entry in data:
            line = f"{entry['energy_mev']:.6E}"
            for l in range(l_max + 1):
                if l in entry["T"]:
                    t = entry["T"][l]
                    line += f"\t{t['T_lminus']:.6E}\t{t['T_lplus']:.6E}\t{t['T_avg']:.6E}"
                else:
                    line += "\t0.000000E+00\t0.000000E+00\t0.000000E+00"
            f.write(line + "\n")


def write_cross_section_dat(data, filepath, reaction, source="TALYS-2.2"):
    """Write cross section data in TSV format."""
    with open(filepath, "w") as f:
        f.write(f"# {reaction} cross section\n")
        f.write(f"# Source: {source}\n")
        f.write(f"#\n")

        if "sigma_gamma_mb" in data[0]:
            # binary.tot format
            f.write("# E(MeV)\tsigma_gamma(mb)\tsigma_neutron(mb)\tsigma_proton(mb)\t"
                    "sigma_deuteron(mb)\tsigma_triton(mb)\tsigma_helion(mb)\tsigma_alpha(mb)\n")
            for d in data:
                f.write(f"{d['energy_mev']:.6E}")
                for key in ["sigma_gamma_mb", "sigma_neutron_mb", "sigma_proton_mb",
                            "sigma_deuteron_mb", "sigma_triton_mb", "sigma_helion_mb",
                            "sigma_alpha_mb"]:
                    f.write(f"\t{d[key]:.6E}")
                f.write("\n")
        elif "sigma_total_mb" in data[0]:
            # all.tot format
            f.write("# E(MeV)\tsigma_nonelastic(mb)\tsigma_elastic(mb)\tsigma_total(mb)\t"
                    "sigma_compound_elastic(mb)\tsigma_shape_elastic(mb)\tsigma_reaction(mb)\n")
            for d in data:
                f.write(f"{d['energy_mev']:.6E}")
                for key in ["sigma_nonelastic_mb", "sigma_elastic_mb", "sigma_total_mb",
                            "sigma_compound_elastic_mb", "sigma_shape_elastic_mb",
                            "sigma_reaction_mb"]:
                    f.write(f"\t{d[key]:.6E}")
                f.write("\n")


# ============================================================
# TALYS Input Generation
# ============================================================

def talys_input_fe56():
    """TALYS input for Fe-56(n,x): transmission + cross sections."""
    return """\
# TALYS input: Fe-56(n,x)
# B1 (transmission) + B2 (cross section) benchmark
projectile n
element Fe
mass 56
energy energies_fine.txt
outbasic y
outtransenergy y
transpower 12
outdiscrete y
"""


def talys_input_u238():
    """TALYS input for U-238(n,gamma): deformed nucleus stress test (B3)."""
    return """\
# TALYS input: U-238(n,gamma)
# B3 benchmark: deformed heavy nucleus stress test
projectile n
element U
mass 238
energy energies_fine.txt
outbasic y
outtransenergy y
transpower 12
outdiscrete y
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate TALYS golden data")
    parser.add_argument("--talys-path", default="talys", help="Path to TALYS binary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder data without running TALYS",
    )
    parser.add_argument(
        "--skip-u238",
        action="store_true",
        help="Skip U-238 calculation (very slow for deformed nucleus)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    energies = generate_energy_grid(n_points=100, e_min=0.001, e_max=20.0)
    energy_text = "\n".join(f"{e:.6f}" for e in energies) + "\n"

    if args.dry_run:
        print("Dry-run mode: skipping TALYS execution")
        print("Run without --dry-run with TALYS installed to generate real data")
        return

    # Check TALYS availability
    try:
        result = subprocess.run(
            [args.talys_path],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        print(
            f"TALYS not found at '{args.talys_path}'. "
            f"Install from https://github.com/arjankoning1/talys",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- B1 + B2: Fe-56(n,x) ----
    print("=" * 60)
    print("Running TALYS for Fe-56(n,x) [B1 + B2 benchmarks]...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_talys(args.talys_path, tmpdir, talys_input_fe56(), energy_text)
        print("TALYS Fe-56 calculation complete.")

        # Parse transmission coefficients (B1)
        trans_file = os.path.join(tmpdir, "transmission_n.out")
        if os.path.exists(trans_file):
            trans_data = parse_transmission_file(trans_file)
            l_max = max(max(e["T"].keys()) for e in trans_data)
            print(f"  B1: Parsed {len(trans_data)} energy points, l_max={l_max}")

            write_transmission_dat(
                trans_data,
                OUTPUT_DIR / "fe56_ng_transmission.dat",
                l_max=l_max,
            )
            write_transmission_spinorbit_dat(
                trans_data,
                OUTPUT_DIR / "fe56_ng_transmission_spinorbit.dat",
                l_max=l_max,
            )
        else:
            print("  WARNING: transmission_n.out not found!")
            trans_data = []

        # Parse cross sections (B2)
        binary_file = os.path.join(tmpdir, "binary.tot")
        if os.path.exists(binary_file):
            binary_data = parse_binary_cross_sections(binary_file)
            print(f"  B2: Parsed {len(binary_data)} cross section points")
            write_cross_section_dat(
                binary_data,
                OUTPUT_DIR / "fe56_ng_cross_section.dat",
                reaction="Fe-56(n,x) binary channel cross sections",
            )
        else:
            print("  WARNING: binary.tot not found!")
            binary_data = []

        # Parse total cross sections
        all_file = os.path.join(tmpdir, "all.tot")
        if os.path.exists(all_file):
            all_data = parse_all_cross_sections(all_file)
            write_cross_section_dat(
                all_data,
                OUTPUT_DIR / "fe56_total_cross_section.dat",
                reaction="Fe-56(n,x) total/elastic/reaction cross sections",
            )

        # Copy TALYS input for reproducibility
        shutil.copy(os.path.join(tmpdir, "input"), OUTPUT_DIR / "fe56_talys_input.txt")
        shutil.copy(os.path.join(tmpdir, "energies_fine.txt"), OUTPUT_DIR / "energies_fine.txt")

    # Write Fe-56 metadata
    meta = {
        "source": "TALYS-2.2",
        "target": {"element": "Fe", "A": 56, "Z": 26},
        "projectile": "n",
        "reaction": "(n,gamma)",
        "optical_model": "Koning-Delaroche (2003) global neutron OMP",
        "energy_range_mev": [energies[0], energies[-1]],
        "n_energies": len(energies),
        "n_transmission_energies": len(trans_data) if trans_data else 0,
        "l_max_transmission": l_max if trans_data else 0,
        "n_cross_section_energies": len(binary_data) if binary_data else 0,
        "notes": "Generated from TALYS-2.2 (MIT License). "
                 "Reference: Koning, Hilaire, Goriely, EPJ A59(6) 131 (2023).",
        "files": {
            "transmission": "fe56_ng_transmission.dat",
            "transmission_spinorbit": "fe56_ng_transmission_spinorbit.dat",
            "binary_cross_sections": "fe56_ng_cross_section.dat",
            "total_cross_sections": "fe56_total_cross_section.dat",
            "talys_input": "fe56_talys_input.txt",
            "energy_grid": "energies_fine.txt",
        },
    }
    with open(OUTPUT_DIR / "fe56_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Written Fe-56 golden data to {OUTPUT_DIR}/")

    # ---- B3: U-238(n,x) ----
    if not args.skip_u238:
        print()
        print("=" * 60)
        print("Running TALYS for U-238(n,x) [B3 benchmark - deformed]...")
        print("  (This may take several minutes for a deformed heavy nucleus)")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_talys(args.talys_path, tmpdir, talys_input_u238(), energy_text)
            print("TALYS U-238 calculation complete.")

            # Parse transmission coefficients
            trans_file = os.path.join(tmpdir, "transmission_n.out")
            if os.path.exists(trans_file):
                trans_data_u = parse_transmission_file(trans_file)
                l_max_u = max(max(e["T"].keys()) for e in trans_data_u)
                print(f"  B3: Parsed {len(trans_data_u)} transmission energy points, l_max={l_max_u}")
                write_transmission_dat(
                    trans_data_u,
                    OUTPUT_DIR / "u238_ng_transmission.dat",
                    source="TALYS-2.2",
                    l_max=l_max_u,
                )

            # Parse cross sections
            binary_file = os.path.join(tmpdir, "binary.tot")
            if os.path.exists(binary_file):
                binary_data_u = parse_binary_cross_sections(binary_file)
                print(f"  B3: Parsed {len(binary_data_u)} cross section points")
                write_cross_section_dat(
                    binary_data_u,
                    OUTPUT_DIR / "u238_ng_cross_section.dat",
                    reaction="U-238(n,x) binary channel cross sections",
                )

            # Copy input
            shutil.copy(os.path.join(tmpdir, "input"), OUTPUT_DIR / "u238_talys_input.txt")

        # Write U-238 metadata
        meta_u = {
            "source": "TALYS-2.2",
            "target": {"element": "U", "A": 238, "Z": 92},
            "projectile": "n",
            "reaction": "(n,gamma)",
            "optical_model": "Koning-Delaroche (2003) global OMP + coupled-channel for deformed",
            "deformation": {"beta2": 0.22, "coupled_states": ["0+", "2+", "4+"]},
            "energy_range_mev": [energies[0], energies[-1]],
            "n_energies": len(energies),
            "notes": "B3 stress test: deformed heavy nucleus with fission competition.",
        }
        with open(OUTPUT_DIR / "u238_metadata.json", "w") as f:
            json.dump(meta_u, f, indent=2)
        print(f"  Written U-238 golden data to {OUTPUT_DIR}/")
    else:
        print("Skipping U-238 calculation (--skip-u238)")

    print()
    print("Done! Golden data files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
