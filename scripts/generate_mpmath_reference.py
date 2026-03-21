#!/usr/bin/env python3
"""Generate high-precision Coulomb wave function reference data using mpmath.

Produces JSON reference data for ACC-01 validation (relative error < 1e-12).
Uses mpmath with dps=60 (50 digits + 10 safety margin).

Usage:
    cd scripts && uv run generate_mpmath_reference.py
"""

import json
import mpmath

# Set precision: 50 digits + 10 safety margin
mpmath.mp.dps = 60


def coulomb_f(l, eta, rho):
    """Regular Coulomb wave function F_l(eta, rho)."""
    return float(mpmath.coulombf(l, eta, rho))


def coulomb_g(l, eta, rho):
    """Irregular Coulomb wave function G_l(eta, rho)."""
    return float(mpmath.coulombg(l, eta, rho))


def coulomb_fp(l, eta, rho):
    """Derivative F'_l(eta, rho) via numerical differentiation."""
    h = mpmath.mpf("1e-10")
    return float((mpmath.coulombf(l, eta, rho + h) - mpmath.coulombf(l, eta, rho - h)) / (2 * h))


def coulomb_gp(l, eta, rho):
    """Derivative G'_l(eta, rho) via numerical differentiation."""
    h = mpmath.mpf("1e-10")
    return float((mpmath.coulombg(l, eta, rho + h) - mpmath.coulombg(l, eta, rho - h)) / (2 * h))


def sigma_l(l, eta):
    """Coulomb phase shift sigma_l = Im[ln Gamma(l+1+i*eta)]."""
    return float(mpmath.im(mpmath.loggamma(l + 1 + 1j * eta)))


def compute_point(l, eta, rho, label=""):
    """Compute all Coulomb quantities at a single (l, eta, rho) point."""
    eta_mp = mpmath.mpf(str(eta))
    rho_mp = mpmath.mpf(str(rho))

    print(f"  Computing l={l}, eta={eta}, rho={rho} ({label})...")

    f_val = coulomb_f(l, eta_mp, rho_mp)
    g_val = coulomb_g(l, eta_mp, rho_mp)
    fp_val = coulomb_fp(l, eta_mp, rho_mp)
    gp_val = coulomb_gp(l, eta_mp, rho_mp)
    sig = sigma_l(l, eta_mp)

    # Wronskian check: F'G - FG' should be 1
    wronskian = fp_val * g_val - f_val * gp_val

    return {
        "l": l,
        "eta": eta,
        "rho": rho,
        "label": label,
        "F": f_val,
        "G": g_val,
        "Fp": fp_val,
        "Gp": gp_val,
        "sigma": sig,
        "wronskian": wronskian,
    }


def main():
    print("Generating mpmath Coulomb reference data (dps=60)...")

    test_points = []

    # === Tier 1: Standard cases ===
    tier1 = [
        (0, 0.0, 1.0, "Bessel limit: F=sin(rho), G=cos(rho)"),
        (0, 0.0, 3.0, "Bessel limit rho=3"),
        (0, 0.0, 5.0, "Bessel limit rho=5"),
        (0, 0.0, 10.0, "Bessel limit rho=10"),
        (1, 0.0, 3.0, "Bessel l=1"),
        (2, 0.0, 5.0, "Bessel l=2"),
        (0, 1.0, 5.0, "Moderate eta"),
        (2, 1.5, 3.5, "Moderate eta, l=2"),
        (0, 1.0, 1.0, "eta=1, rho=1"),
        (1, 1.0, 3.0, "eta=1, l=1"),
    ]

    # === Tier 2: Difficult cases (barrier penetration) ===
    tier2 = [
        (0, 5.0, 3.0, "Barrier: eta=5, rho=3"),
        (0, 10.0, 5.0, "Deep barrier: eta=10, rho=5"),
        (0, 5.0, 10.0, "Near turning point: eta=5, rho=10"),
        (0, 2.0, 0.5, "Small rho, moderate eta"),
    ]

    # === Tier 3: High angular momentum ===
    tier3 = [
        (5, 2.0, 10.0, "High l=5, moderate eta"),
        (10, 5.0, 20.0, "High l=10"),
        (3, 0.0, 5.0, "Bessel l=3"),
        (5, 0.0, 8.0, "Bessel l=5"),
    ]

    # === Extra: multiple l at same (eta, rho) for recurrence validation ===
    extra = [
        (0, 2.0, 5.0, "Recurrence base l=0"),
        (1, 2.0, 5.0, "Recurrence l=1"),
        (2, 2.0, 5.0, "Recurrence l=2"),
        (3, 2.0, 5.0, "Recurrence l=3"),
    ]

    all_points = [
        ("tier1", tier1),
        ("tier2", tier2),
        ("tier3", tier3),
        ("recurrence", extra),
    ]

    results = {}
    for tier_name, points in all_points:
        print(f"\n=== {tier_name} ===")
        tier_results = []
        for l, eta, rho, label in points:
            try:
                point = compute_point(l, eta, rho, label)
                tier_results.append(point)
            except Exception as e:
                print(f"  FAILED: l={l}, eta={eta}, rho={rho}: {e}")
                tier_results.append({
                    "l": l, "eta": eta, "rho": rho, "label": label,
                    "error": str(e),
                })
        results[tier_name] = tier_results

    # Metadata
    output = {
        "generator": "mpmath",
        "mpmath_version": mpmath.__version__,
        "dps": mpmath.mp.dps,
        "description": "High-precision Coulomb wave function reference data for ACC-01 validation",
        "data": results,
    }

    output_path = "../tests/reference_data/coulomb_mpmath.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Output: {output_path}")
    total = sum(len(v) for v in results.values())
    print(f"Total points: {total}")


if __name__ == "__main__":
    main()
