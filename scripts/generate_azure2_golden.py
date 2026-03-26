#!/usr/bin/env python3
"""Generate AZURE2 golden reference data for 7Be(p,gamma)8B.

This script creates AZURE2 input files and processes output to create
golden reference data files for nucrust R-matrix validation (B4 benchmark).

The 7Be(p,gamma)8B reaction is critical for solar neutrino predictions.
R-matrix parameters are taken from published analyses (Barker 2000, Descouvemont 2004).

Requirements:
  - AZURE2 installed (built from https://github.com/rdeboer1/AZURE2)
  - Output directory: tests/reference_data/azure2/

Usage:
  python scripts/generate_azure2_golden.py [--azure2-path /path/to/AZURE2] [--dry-run]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "tests" / "reference_data" / "azure2"

# Physical constants
HBAR_C = 197.3269804  # MeV*fm
AMU = 931.494  # MeV/c^2
M_PROTON = 1.00727647  # amu
M_7BE = 7.01692983  # amu (7Be mass)
M_8B = 8.02461   # amu (8B mass)
Q_VALUE = 0.1375  # MeV (7Be + p -> 8B + gamma)
SEPARATION_E = 0.1375  # MeV

# 8B compound nucleus levels from published R-matrix analyses
# Reference: Barker (2000), Descouvemont & Baye (2004), Junghans et al. (2003)
# Format: (J_pi, E_x_MeV, is_bound, reduced_width_p, reduced_width_gamma)
B8_LEVELS = [
    # J^pi  E_cm(MeV) fixed  pair_in pair_out s*2  l*2  levelID active ch_fix  gamma
    # 1+ ground state (subthreshold, bound)
    {"Jpi": "2+", "E_cm": -0.1375, "bound": True, "note": "ground state"},
    # 1+ first excited (M1)
    {"Jpi": "1+", "E_cm": 0.632, "bound": False, "note": "narrow resonance at 632 keV"},
    # 3+ resonance
    {"Jpi": "3+", "E_cm": 2.183, "bound": False, "note": "broad resonance"},
]


def generate_energy_grid():
    """Generate energy grid for S-factor: 0.01 - 3.0 MeV (lab), 100 points."""
    energies = []
    log_min = math.log10(0.01)
    log_max = math.log10(3.0)
    n_points = 100
    for i in range(n_points):
        e = 10 ** (log_min + (log_max - log_min) * i / (n_points - 1))
        energies.append(e)
    return energies


def sommerfeld_parameter(e_cm, z1, z2, mu):
    """Calculate Sommerfeld parameter eta."""
    alpha = 1.0 / 137.036
    return z1 * z2 * alpha * math.sqrt(mu * AMU / (2.0 * e_cm))


def s_factor_from_cross_section(sigma_barn, e_cm_mev, z1=1, z2=4, mu_amu=None):
    """Convert cross section to astrophysical S-factor."""
    if mu_amu is None:
        mu_amu = M_PROTON * M_7BE / (M_PROTON + M_7BE)
    eta = sommerfeld_parameter(e_cm_mev, z1, z2, mu_amu)
    return sigma_barn * e_cm_mev * math.exp(2.0 * math.pi * eta) * 1e3  # eV*b


def generate_placeholder_sfactor(energies):
    """Generate approximate 7Be(p,gamma)8B S-factor using known systematics.

    S_17(0) = 20.8 ± 0.7 ± 1.0 eV*b (Junghans et al. 2003)
    The S-factor has a 1+ resonance at E_cm = 0.632 MeV.
    """
    data = []
    mu = M_PROTON * M_7BE / (M_PROTON + M_7BE)

    # S-factor parameterization from Descouvemont (2004)
    S0 = 20.8  # eV*b at E=0
    # Direct capture contribution (smooth, slowly rising)
    # Plus 1+ resonance at 0.632 MeV (Gamma ~ 35 keV)
    E_res = 0.632  # MeV
    Gamma_res = 0.035  # MeV (total width)
    Gamma_p = 0.033  # MeV (proton width)
    Gamma_g = 0.002  # MeV (gamma width)

    for e_cm in energies:
        # Direct capture (E1): smooth polynomial
        s_dc = S0 * (1.0 + 3.5 * e_cm - 0.8 * e_cm**2)

        # Breit-Wigner resonance contribution to S-factor
        eta = sommerfeld_parameter(e_cm, 1, 4, mu)
        penetrability = 2.0 * math.pi * eta / (math.exp(2.0 * math.pi * eta) - 1.0)
        bw = (Gamma_p * Gamma_g / ((e_cm - E_res)**2 + (Gamma_res/2)**2))
        # Convert BW cross section to S-factor
        sigma_bw = math.pi * (HBAR_C**2) / (2.0 * mu * AMU * e_cm) * bw
        s_res = sigma_bw * e_cm * math.exp(2.0 * math.pi * eta) * 1e3  # eV*b

        s_total = s_dc + s_res

        # Also compute cross section
        sigma = s_total / (e_cm * math.exp(2.0 * math.pi * eta) * 1e3)  # barn

        data.append({
            "E_cm_MeV": round(e_cm, 6),
            "S_factor_eVb": round(s_total, 4),
            "sigma_barn": sigma,
            "S_dc_eVb": round(s_dc, 4),
            "S_res_eVb": round(s_res, 4),
        })

    return data


def generate_azure2_config(work_dir, energies):
    """Generate AZURE2 configuration file for 7Be(p,gamma)8B.

    AZURE2 config file format has sections:
    <config>, <levels>, <segmentsData>, <segmentsExtrap>
    """
    # NucLine format (31 fields per line):
    # J  pi  E_level  level_fix  pair_in  pair_out  s*2  l*2  levelID
    # active  ch_fix  gamma  j1  pi1  j2  pi2  e2  m1  m2  z1  z2
    # entranceSepE  sepE  j3  pi3  e3  pType  chRad  g1  g2  ecMultMask

    # Particle pairs:
    # Pair 1: 7Be + p (entrance)
    #   j1=0.5 (proton), pi1=1, j2=1.5 (7Be ground state 3/2-), pi2=-1
    #   m1=1.00728, m2=7.01693, z1=1, z2=4
    #   sepE = 0.1375 MeV (Q-value)
    # Pair 2: 8B + gamma (exit, E1 capture)
    #   j1=1.0 (photon), pi1=-1, j2=2.0 (8B ground state 2+), pi2=1
    #   pType=10 (gamma)

    # Channel radius
    r0 = 1.25
    A = 8
    a_c = r0 * A**(1.0/3.0)  # ~2.5 fm

    levels_lines = []

    # --- 2+ ground state (bound, subthreshold) ---
    # 7Be+p channel: J=2, pi=+1, s=2 (j1+j2=0.5+1.5=2), l=0 (s-wave)
    # Note: s is stored as 2*s in input, l as 2*l
    levels_lines.append(
        f"2.0  1  -0.1375  0  1  1  4  0  1  1  0  0.3000  "
        f"0.5  1  1.5  -1  0.0  {M_PROTON:.5f}  {M_7BE:.5f}  1  4  "
        f"0.1375  0.1375  0  1  0.0  0  {a_c:.4f}  0.0  0.0  0"
    )
    # 8B+gamma channel for 2+ (E1 to 2+ ground state)
    levels_lines.append(
        f"2.0  1  -0.1375  0  1  2  2  2  1  1  0  0.0050  "
        f"1.0  -1  2.0  1  0.0  0.0  {M_8B:.5f}  0  4  "
        f"0.1375  0.1375  0  1  0.0  10  {a_c:.4f}  0.0  0.0  2"
    )

    # --- 1+ resonance at 0.632 MeV ---
    # 7Be+p: J=1, pi=+1, s=2, l=2 (d-wave) OR s=1, l=0 (s-wave with different coupling)
    # For simplicity: s=2 (j1+j2=2), l=2 (d-wave, so parity = (-1)^2 * pi(p) * pi(7Be) = +1*+1*(-1) = -1 ...
    # Actually for 1+ with p+7Be(3/2-): need pi = (-1)^l * pi_p * pi_7Be = (-1)^l * (+1)*(-1)
    # For pi=+1: (-1)^l * (-1) = +1 => (-1)^l = -1 => l=odd => l=1
    # s can be 1 or 2 (|j1-j2| to j1+j2 = |0.5-1.5| to 0.5+1.5 = 1 to 2)
    # Triangle: |s-l| <= J <= s+l, so |s-1| <= 1 <= s+1
    # s=1,l=1: |0|<=1<=2 OK
    # s=2,l=1: |1|<=1<=3 OK (1<=1 OK)
    levels_lines.append(
        f"1.0  1  0.632  0  1  1  2  2  2  1  0  0.5500  "
        f"0.5  1  1.5  -1  0.0  {M_PROTON:.5f}  {M_7BE:.5f}  1  4  "
        f"0.1375  0.1375  0  1  0.0  0  {a_c:.4f}  0.0  0.0  0"
    )
    # gamma channel for 1+
    levels_lines.append(
        f"1.0  1  0.632  0  1  2  2  2  2  1  0  0.0020  "
        f"1.0  -1  2.0  1  0.0  0.0  {M_8B:.5f}  0  4  "
        f"0.1375  0.1375  0  1  0.0  10  {a_c:.4f}  0.0  0.0  2"
    )

    # --- 3+ resonance at 2.183 MeV ---
    # For 3+ with p+7Be(3/2-): pi = (-1)^l * (-1) = +1 => l=odd
    # s=2,l=1: |1|<=3<=3 OK
    # s=2,l=3: |-1|<=3<=5 OK
    levels_lines.append(
        f"3.0  1  2.183  1  1  1  4  2  3  1  1  0.8000  "
        f"0.5  1  1.5  -1  0.0  {M_PROTON:.5f}  {M_7BE:.5f}  1  4  "
        f"0.1375  0.1375  0  1  0.0  0  {a_c:.4f}  0.0  0.0  0"
    )
    # gamma channel for 3+
    levels_lines.append(
        f"3.0  1  2.183  1  1  2  2  2  3  1  1  0.0010  "
        f"1.0  -1  2.0  1  0.0  0.0  {M_8B:.5f}  0  4  "
        f"0.1375  0.1375  0  1  0.0  10  {a_c:.4f}  0.0  0.0  2"
    )

    # Write energy file for extrapolation
    energy_file = os.path.join(work_dir, "extrap_energies.dat")
    with open(energy_file, "w") as f:
        for e in energies:
            # AZURE2 data format: E_lab  sigma  dsigma  (angle for differential)
            # For extrapolation, just list energies
            f.write(f"{e:.6f}  0.0  0.0\n")

    # Write config file
    output_dir = os.path.join(work_dir, "output")
    check_dir = os.path.join(work_dir, "checks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    config = f"""<config>
true
{output_dir}/
{check_dir}/
no
no
no
no
no
no
no
no
</config>
<levels>
{chr(10).join(levels_lines)}
</levels>
<segmentsData>
</segmentsData>
<segmentsTest>
1  1  -1  0.010  3.000  0.030  0.0  0.0  1.0  0
</segmentsTest>
<targetInt>
</targetInt>
"""

    config_path = os.path.join(work_dir, "be7pg.azr")
    with open(config_path, "w") as f:
        f.write(config)

    return config_path


def run_azure2(azure2_path, config_path, work_dir):
    """Run AZURE2 in calculation-only (extrapolation) mode."""
    result = subprocess.run(
        [azure2_path, "--no-gui", "--use-brune", config_path],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"AZURE2 output:\n{result.stdout}", file=sys.stderr)
        print(f"AZURE2 errors:\n{result.stderr}", file=sys.stderr)
    return result


def parse_azure2_output(work_dir):
    """Parse AZURE2 extrapolation output."""
    output_dir = os.path.join(work_dir, "output")
    data = []

    # AZURE2 outputs AZUREOut_aa=*.extrap files
    import glob
    extrap_files = glob.glob(os.path.join(output_dir, "AZUREOut_aa=*.extrap"))
    if not extrap_files:
        extrap_files = glob.glob(os.path.join(output_dir, "*.extrap"))
    if not extrap_files:
        # Try other output patterns
        extrap_files = glob.glob(os.path.join(output_dir, "*"))

    for fpath in extrap_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        e = float(parts[0])
                        sigma = float(parts[1])
                        data.append({"E_cm_MeV": e, "sigma_barn": sigma})
                    except ValueError:
                        continue

    return data


def write_sfactor_dat(data, filepath, source="placeholder"):
    """Write S-factor data in TSV format."""
    with open(filepath, "w") as f:
        f.write(f"# 7Be(p,gamma)8B astrophysical S-factor\n")
        f.write(f"# Source: {source}\n")
        if source == "placeholder":
            f.write("# WARNING: Placeholder data based on systematics. Replace with actual AZURE2 output.\n")
        f.write("#\n")
        f.write("# Reference: Junghans et al., PRC 68, 065803 (2003): S_17(0) = 20.8 +/- 0.7 +/- 1.0 eV*b\n")
        f.write("#\n")
        f.write("# E_cm(MeV)\tS_factor(eV*b)\tsigma(barn)\tS_dc(eV*b)\tS_res(eV*b)\n")
        for entry in data:
            line = f"{entry['E_cm_MeV']:.6E}\t{entry['S_factor_eVb']:.4E}"
            line += f"\t{entry['sigma_barn']:.6E}"
            if 'S_dc_eVb' in entry:
                line += f"\t{entry['S_dc_eVb']:.4E}\t{entry['S_res_eVb']:.4E}"
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate AZURE2 golden data")
    parser.add_argument(
        "--azure2-path",
        default="AZURE2",
        help="Path to AZURE2 binary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder data without running AZURE2",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    energies = generate_energy_grid()

    if args.dry_run:
        print("Generating placeholder S-factor data (AZURE2 not available)...")
        sfactor_data = generate_placeholder_sfactor(energies)
        source = "placeholder (systematics)"
    else:
        try:
            subprocess.run(
                [args.azure2_path, "--help"],
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            print(
                f"AZURE2 not found at '{args.azure2_path}'. Use --dry-run for placeholder data.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Setting up AZURE2 for 7Be(p,gamma)8B...")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generate_azure2_config(tmpdir, energies)
            print(f"Config file: {config_path}")

            result = run_azure2(args.azure2_path, config_path, tmpdir)
            if result.returncode == 0:
                azure2_data = parse_azure2_output(tmpdir)
                if azure2_data:
                    print(f"Parsed {len(azure2_data)} AZURE2 data points")
                    # Convert to S-factor
                    mu = M_PROTON * M_7BE / (M_PROTON + M_7BE)
                    sfactor_data = []
                    for d in azure2_data:
                        s = s_factor_from_cross_section(d["sigma_barn"], d["E_cm_MeV"])
                        sfactor_data.append({
                            "E_cm_MeV": d["E_cm_MeV"],
                            "S_factor_eVb": round(s, 4),
                            "sigma_barn": d["sigma_barn"],
                            "S_dc_eVb": 0.0,
                            "S_res_eVb": 0.0,
                        })
                    source = "AZURE2 (Brune parameterization)"
                else:
                    print("WARNING: No AZURE2 output found. Using placeholder data.")
                    sfactor_data = generate_placeholder_sfactor(energies)
                    source = "placeholder (AZURE2 ran but no output parsed)"
            else:
                print(f"AZURE2 failed (exit code {result.returncode}). Using placeholder data.")
                sfactor_data = generate_placeholder_sfactor(energies)
                source = "placeholder (AZURE2 failed)"

    # Write S-factor output
    sfactor_path = OUTPUT_DIR / "be7_pg_sfactor.dat"
    write_sfactor_dat(sfactor_data, sfactor_path, source)
    print(f"Written: {sfactor_path}")

    # Write metadata
    meta = {
        "source": source,
        "reaction": "7Be(p,gamma)8B",
        "compound_nucleus": "8B",
        "reference": "Junghans et al., PRC 68, 065803 (2003)",
        "S17_0_eVb": 20.8,
        "S17_0_error_eVb": [0.7, 1.0],
        "energy_range_cm_mev": [energies[0], energies[-1]],
        "n_energies": len(energies),
        "r_matrix_parameters": {
            "channel_radius_fm": 1.25 * 8**(1.0/3.0),
            "levels": [
                {"Jpi": "2+", "E_cm_MeV": -0.1375, "type": "subthreshold"},
                {"Jpi": "1+", "E_cm_MeV": 0.632, "type": "resonance"},
                {"Jpi": "3+", "E_cm_MeV": 2.183, "type": "resonance"},
            ],
            "particle_pairs": [
                {"pair": "7Be+p", "j1": 0.5, "j2": 1.5, "z1": 1, "z2": 4},
                {"pair": "8B+gamma", "type": "capture"},
            ],
        },
        "notes": "B4 benchmark for nucrust R-matrix validation.",
    }
    meta_path = OUTPUT_DIR / "be7_pg_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Written: {meta_path}")


if __name__ == "__main__":
    main()
