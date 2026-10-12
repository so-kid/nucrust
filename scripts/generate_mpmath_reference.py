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


def _derivative(func, l, eta, rho):
    """Exact derivative X'_l = S_{l+1} X_l - R_{l+1} X_{l+1} (DLMF 33.4.4).

    Evaluated at full working precision, unlike a finite difference.
    """
    l1 = mpmath.mpf(l + 1)
    s = l1 / rho + eta / l1
    r = mpmath.sqrt(1 + (eta / l1) ** 2)
    return s * func(l, eta, rho) - r * func(l + 1, eta, rho)


def coulomb_fp(l, eta, rho):
    """Derivative F'_l(eta, rho)."""
    return float(_derivative(mpmath.coulombf, l, eta, rho))


def coulomb_gp(l, eta, rho):
    """Derivative G'_l(eta, rho)."""
    return float(_derivative(mpmath.coulombg, l, eta, rho))


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

    # === Tier 1: Standard cases (Bessel limits and moderate parameters) ===
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
        (0, 0.5, 2.0, "Small eta, small rho"),
        (1, 2.0, 8.0, "Oscillatory region l=1"),
        (3, 1.0, 10.0, "Moderate l=3, oscillatory"),
    ]

    # === Tier 2: Difficult cases (barrier penetration, forbidden region) ===
    tier2 = [
        (0, 5.0, 3.0, "Barrier: eta=5, rho=3"),
        (0, 10.0, 5.0, "Deep barrier: eta=10, rho=5"),
        (0, 5.0, 10.0, "Near turning point: eta=5, rho=10"),
        (0, 2.0, 0.5, "Small rho, moderate eta"),
        (0, 10.0, 1.0, "Deep forbidden: eta=10, rho=1"),
        (0, 10.0, 20.0, "Near turning: eta=10, rho=20"),
        (0, 20.0, 5.0, "Very deep barrier: eta=20, rho=5"),
        (0, 20.0, 40.0, "Near turning: eta=20, rho=40"),
        (1, 10.0, 3.0, "Barrier l=1: eta=10, rho=3"),
        (2, 5.0, 2.0, "Barrier l=2: eta=5, rho=2"),
    ]

    # === Tier 3: High angular momentum ===
    tier3 = [
        (5, 2.0, 10.0, "High l=5, moderate eta"),
        (10, 5.0, 20.0, "High l=10"),
        (3, 0.0, 5.0, "Bessel l=3"),
        (5, 0.0, 8.0, "Bessel l=5"),
        (10, 0.0, 15.0, "Bessel l=10"),
        (15, 3.0, 25.0, "High l=15, eta=3"),
        (20, 5.0, 40.0, "High l=20, eta=5"),
        (10, 2.0, 15.0, "High l=10, moderate eta"),
        (5, 5.0, 15.0, "l=5, eta=5, oscillatory"),
        (8, 3.0, 12.0, "l=8, moderate parameters"),
    ]

    # === Tier 4: Extreme parameters (nuclear physics regime) ===
    tier4 = [
        (0, 30.0, 10.0, "Extreme eta=30, forbidden"),
        (0, 30.0, 60.0, "Extreme eta=30, turning point"),
        (0, 50.0, 20.0, "Very extreme eta=50"),
        (0, 50.0, 100.0, "Extreme eta=50, turning point"),
        (5, 20.0, 30.0, "High l + high eta"),
        (10, 10.0, 30.0, "l=10, eta=10, intermediate"),
        (0, 0.1, 0.1, "Small rho, small eta"),
        (0, 0.01, 50.0, "Near-zero eta, large rho"),
        (3, 10.0, 5.0, "l=3, deep forbidden"),
        (15, 10.0, 40.0, "l=15, eta=10, near classical"),
    ]

    # === Extra: multiple l at same (eta, rho) for recurrence validation ===
    extra = [
        (0, 2.0, 5.0, "Recurrence base l=0"),
        (1, 2.0, 5.0, "Recurrence l=1"),
        (2, 2.0, 5.0, "Recurrence l=2"),
        (3, 2.0, 5.0, "Recurrence l=3"),
        (4, 2.0, 5.0, "Recurrence l=4"),
        (5, 2.0, 5.0, "Recurrence l=5"),
    ]

    # === Nuclear physics cases (realistic Sommerfeld parameters) ===
    nuclear = [
        # Fe-56 + n at various energies (eta values from Sommerfeld parameter)
        (0, 0.0, 4.5, "Fe56+n 1MeV: eta=0 (neutral)"),
        (5, 0.0, 4.5, "Fe56+n 1MeV: l=5"),
        (10, 0.0, 4.5, "Fe56+n 1MeV: l=10"),
        (15, 0.0, 4.5, "Fe56+n 1MeV: l=15"),
        # Be-7 + p (charged particle, solar reaction)
        (0, 3.5, 2.0, "Be7+p 100keV: eta~3.5, rho~2"),
        (1, 3.5, 2.0, "Be7+p 100keV: l=1"),
        (2, 3.5, 2.0, "Be7+p 100keV: l=2"),
        (0, 7.0, 1.0, "Be7+p 25keV: eta~7, rho~1"),
        (0, 1.5, 5.0, "Be7+p 500keV: eta~1.5, rho~5"),
        # Alpha capture
        (0, 10.0, 3.0, "Alpha+C12 300keV: eta~10"),
        (2, 10.0, 3.0, "Alpha+C12 300keV: l=2"),
        (0, 5.0, 6.0, "Alpha+C12 1MeV: eta~5"),
    ]

    # === Classically forbidden region, l > 0 and near the turning point ===
    forbidden = [
        (1, 10.0, 3.0, "Forbidden l=1: eta=10, rho=3"),
        (3, 10.0, 3.0, "Forbidden l=3: eta=10, rho=3"),
        (5, 10.0, 3.0, "Forbidden l=5: eta=10, rho=3"),
        (10, 10.0, 3.0, "Forbidden l=10: eta=10, rho=3"),
        (1, 5.0, 2.0, "Forbidden l=1: eta=5, rho=2"),
        (3, 5.0, 2.0, "Forbidden l=3: eta=5, rho=2"),
        (6, 5.0, 2.0, "Forbidden l=6: eta=5, rho=2"),
        (1, 30.0, 10.0, "Forbidden l=1: eta=30, rho=10"),
        (5, 30.0, 10.0, "Forbidden l=5: eta=30, rho=10"),
        (10, 30.0, 10.0, "Forbidden l=10: eta=30, rho=10"),
        (2, 20.0, 5.0, "Forbidden l=2: eta=20, rho=5"),
        (8, 20.0, 5.0, "Forbidden l=8: eta=20, rho=5"),
        (3, 50.0, 20.0, "Forbidden l=3: eta=50, rho=20"),
        (1, 1.0, 1.0, "Shallow forbidden l=1: eta=1, rho=1"),
        (3, 1.0, 1.0, "Shallow forbidden l=3: eta=1, rho=1"),
        (0, 10.0, 17.0, "Below turning point: eta=10, rho=17"),
        (2, 10.0, 17.0, "Below turning point l=2: eta=10, rho=17"),
        (0, 10.0, 19.0, "Just below turning point: eta=10, rho=19"),
        (0, 50.0, 85.0, "Below turning point: eta=50, rho=85"),
        (0, 3.0, 5.5, "Below turning point: eta=3, rho=5.5"),
        (4, 3.0, 5.5, "Below turning point l=4: eta=3, rho=5.5"),
        (0, 5.0, 9.5, "Just below turning point: eta=5, rho=9.5"),
        (0, 5.0, 10.5, "Just above turning point: eta=5, rho=10.5"),
        # Large eta between the series and Steed regimes (rho-shift path)
        (0, 100.0, 130.0, "Large eta below turning point: eta=100, rho=130"),
        (0, 100.0, 150.0, "Large eta below turning point: eta=100, rho=150"),
        (0, 100.0, 170.0, "Large eta below turning point: eta=100, rho=170"),
        (2, 100.0, 170.0, "Large eta below turning point l=2: eta=100, rho=170"),
        (0, 100.0, 185.0, "Large eta below turning point: eta=100, rho=185"),
        (0, 200.0, 300.0, "Very large eta: eta=200, rho=300"),
        (0, 200.0, 380.0, "Very large eta near turning point: eta=200, rho=380"),
    ]

    # === Small rho (power-series regime) ===
    small_rho = [
        (0, 0.0, 0.01, "Small rho=0.01, eta=0"),
        (0, 1.0, 0.01, "Small rho=0.01, eta=1"),
        (0, 10.0, 0.01, "Small rho=0.01, eta=10"),
        (2, 0.5, 0.01, "Small rho=0.01, eta=0.5, l=2"),
        (0, 5.0, 0.1, "Small rho=0.1, eta=5"),
        (5, 0.0, 0.1, "Small rho=0.1, eta=0, l=5"),
        (1, 2.0, 0.1, "Small rho=0.1, eta=2, l=1"),
        (0, 0.0, 0.5, "rho=0.5, eta=0"),
        (3, 1.0, 0.5, "rho=0.5, eta=1, l=3"),
        (0, 20.0, 0.5, "rho=0.5, eta=20"),
    ]

    # === Attractive Coulomb field (eta < 0) ===
    attractive = [
        (0, -1.0, 1.0, "Attractive eta=-1, rho=1"),
        (2, -5.0, 2.0, "Attractive eta=-5, rho=2, l=2"),
        (0, -10.0, 0.1, "Attractive eta=-10, rho=0.1"),
        (5, -2.0, 10.0, "Attractive eta=-2, rho=10, l=5"),
        (0, -0.5, 20.0, "Attractive eta=-0.5, rho=20"),
    ]

    all_points = [
        ("tier1", tier1),
        ("tier2", tier2),
        ("tier3", tier3),
        ("tier4_extreme", tier4),
        ("recurrence", extra),
        ("nuclear_physics", nuclear),
        ("forbidden_region", forbidden),
        ("small_rho", small_rho),
        ("attractive", attractive),
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
