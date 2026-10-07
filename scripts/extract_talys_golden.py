#!/usr/bin/env python3
"""Convert raw TALYS output files into the nucrust golden reference tables.

Usage (standard library only):

    uv run --no-project scripts/extract_talys_golden.py            # regenerate tables
    uv run --no-project scripts/extract_talys_golden.py --check    # verify tables are up to date
    uv run --no-project scripts/extract_talys_golden.py --collect fe56 <talys-run-dir>
                                                                   # copy raw files from a TALYS run

The script reads ONLY the raw TALYS output files committed under
``tests/reference_data/talys/<nuc>/raw/`` and the matching ``talys_input.txt``.
It never calls TALYS and has no fallback data: if a raw file is missing or does
not look like a completed TALYS run, it aborts with an error.

Raw files used per nucleus (Z, A = target; residual = Z, A+1):
    binary.tot           binary cross sections; column "gamma" = (n,gamma) first-step capture
    rpZZZAAA.tot         residual production cross section of the (A+1) nucleus
    all.tot              non-elastic / elastic / total / compound-elastic / shape-elastic / reaction
    transmission_n.out   neutron transmission coefficients T(L-1/2,L), T(L+1/2,L), T(L) per energy
    output.gz            TALYS main output (version banner, date, run status)

Numbers are copied as the original strings from the TALYS files (no reformatting),
so the tables are exactly as precise as TALYS prints them (7 significant digits).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "reference_data" / "talys"

NUCLEI = {
    "fe56": {"element": "Fe", "Z": 26, "A": 56},
    "u238": {"element": "U", "Z": 92, "A": 238},
}

ZERO = "0.000000E+00"


class ExtractError(RuntimeError):
    pass


def read_text(path: Path) -> str:
    if not path.is_file():
        raise ExtractError(f"missing raw TALYS file: {path.relative_to(ROOT)}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def read_columns(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a YANDF .tot table: (column names from the first '##' line, rows of strings)."""
    names: list[str] | None = None
    rows: list[list[str]] = []
    for line in read_text(path).splitlines():
        if line.startswith("##"):
            if names is None:
                names = line[2:].split()
            continue
        if line.startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    if names is None or not rows:
        raise ExtractError(f"no data table found in {path.name}")
    for r in rows:
        if len(r) != len(names):
            raise ExtractError(f"{path.name}: row has {len(r)} fields, header has {len(names)}")
    return names, rows


def col(names: list[str], rows: list[list[str]], name: str) -> list[str]:
    try:
        i = names.index(name)
    except ValueError as exc:
        raise ExtractError(f"column {name!r} not in header {names}") from exc
    return [r[i] for r in rows]


def read_transmission(path: Path):
    """Parse transmission_n.out into [(E_str, [(L, Tm, Tp, Tavg), ...]), ...] (strings)."""
    blocks = []
    cur = None
    particle = None
    for line in read_text(path).splitlines():
        if "energy [MeV]" in line:
            cur = []
            blocks.append((line.split(":", 1)[1].strip(), cur))
        elif line.startswith("#   particle:"):
            particle = line.split(":", 1)[1].strip()
        elif line.startswith("#") or not line.strip():
            continue
        else:
            p = line.split()
            if len(p) != 4 or cur is None:
                raise ExtractError(f"{path.name}: unexpected line {line!r}")
            cur.append((int(p[0]), p[1], p[2], p[3]))
    if particle != "neutron" or not blocks:
        raise ExtractError(f"{path.name}: not a neutron transmission file")
    for e, ls in blocks:
        if [l for l, *_ in ls] != list(range(len(ls))):
            raise ExtractError(f"{path.name}: L values at E={e} are not 0..Lmax")
    return blocks


def run_info(nuc: str) -> dict:
    d = DATA / nuc
    out = read_text(d / "raw" / "output.gz")
    inp = (d / "talys_input.txt").read_text()
    m = re.search(r"^\s*(TALYS-\S+)\s+\(Version: ([^)]+)\)", out, re.M)
    dm = re.search(r"^ Date: (\S+)", out, re.M)
    if not m or not dm:
        raise ExtractError("TALYS version banner / date not found in output.gz")
    if "The TALYS team congratulates you with this successful calculation." not in out:
        raise ExtractError("output.gz does not end in a successful TALYS calculation")
    cm = re.search(r"git ([0-9a-f]{10,40})", inp)
    if not cm:
        raise ExtractError("TALYS source commit not recorded in talys_input.txt comments")
    return {"version": m.group(1), "version_date": m.group(2), "git": cm.group(1), "date": dm.group(1)}


def header_block(nuc, info, title, sources, units, notes) -> list[str]:
    cfg = NUCLEI[nuc]
    h = [
        f"# {cfg['element']}-{cfg['A']}(n,x) {title}",
        f"# TALYS version: {info['version']} (git {info['git']}), version date {info['version_date']}",
        f"# Input: {nuc}/talys_input.txt (energy grid: energies_golden.txt)",
        f"# Generated: {info['date']} (date of the TALYS run); tables produced by scripts/extract_talys_golden.py",
        "# Raw TALYS outputs (committed in raw/):",
    ]
    h += [f"#   {s}" for s in sources]
    h.append(f"# Units: {units}")
    h += [f"# {n}" for n in notes]
    h.append("#")
    return h


def build_cross_sections(nuc: str, info: dict) -> str:
    cfg = NUCLEI[nuc]
    raw = DATA / nuc / "raw"
    resid = f"rp{cfg['Z']:03d}{cfg['A'] + 1:03d}.tot"
    bn, br = read_columns(raw / "binary.tot")
    rn, rr = read_columns(raw / resid)
    an, ar = read_columns(raw / "all.tot")
    energies = col(bn, br, "E")
    for label, (n_, r_) in (("residual", (rn, rr)), ("all.tot", (an, ar))):
        if col(n_, r_, "E") != energies:
            raise ExtractError(f"energy column of {label} file differs from binary.tot")
    cols = [
        ("E_lab", energies),
        ("sigma_ng_binary", col(bn, br, "gamma")),
        ("sigma_res", col(rn, rr, "xs")),
        ("sigma_nonel", col(an, ar, "Non-elastic")),
        ("sigma_el", col(an, ar, "Elastic")),
        ("sigma_tot", col(an, ar, "Total")),
        ("sigma_compel", col(an, ar, "Compound_elast.")),
        ("sigma_shapeel", col(an, ar, "Shape_elastic")),
        ("sigma_reac", col(an, ar, "Reaction")),
    ]
    A1 = cfg["A"] + 1
    sources = [
        "E_lab, sigma_ng_binary : binary.tot, columns 'E' and 'gamma' (binary (n,gamma) = first-step radiative capture)",
        f"sigma_res              : {resid}, column 'xs' (production of the A={A1} nucleus: (n,gamma) minus any "
        "further particle emission)",
        "sigma_nonel .. sigma_reac : all.tot, columns 'Non-elastic', 'Elastic', 'Total', 'Compound_elast.',"
        " 'Shape_elastic', 'Reaction'",
    ]
    notes = [
        "E_lab is the incident neutron energy in the LABORATORY frame, as the TALYS 'energy' keyword defines it.",
        "sigma_ng_binary ignores decay of the compound nucleus after the first gamma; sigma_res accounts for it",
        "  (they differ once (n,gamma n') etc. open up).",
        "sigma_reac = TALYS reaction cross section (= sigma_nonel + sigma_compel, i.e. total minus shape elastic).",
    ]
    lines = header_block(
        nuc, info, "capture, elastic, non-elastic, total and reaction cross sections", sources, "MeV, mb", notes
    )
    lines.append("# " + "\t".join(n for n, _ in cols))
    for i in range(len(energies)):
        lines.append("\t".join(c[i] for _, c in cols))
    return "\n".join(lines) + "\n"


def build_transmission(nuc: str, info: dict):
    blocks = read_transmission(DATA / nuc / "raw" / "transmission_n.out")
    lmax = max(len(ls) for _, ls in blocks) - 1
    first_e = float(blocks[0][0])
    ratio = first_e / 1.0e-3
    src = ["transmission_n.out, per-energy blocks: 'energy [MeV]' and columns L, T(L-1/2,L), T(L+1/2,L), T(L)"]
    notes = [
        "Energy is the neutron LAB energy printed by TALYS: E_lab = E_grid * (M + m_n) / M, where E_grid is the TALYS",
        "  basic emission grid 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, ... MeV (relative-motion energy).",
        f"For this target E_lab(first) / 0.001 MeV = {ratio:.6f}, i.e. E_cm(first) = 0.001 MeV.",
        "TALYS lists L only up to the largest L with non-negligible T; unlisted L are printed as 0 in the wide table.",
        "T(L-1/2,L) for L=0 does not exist (j=-1/2) and is printed as 0 by TALYS.",
    ]
    jl = header_block(
        nuc, info, "neutron transmission coefficients T_lj (spin-orbit split), long format", src,
        "E in MeV; T dimensionless", notes,
    )
    jl.append("# E_lab\tL\tT(L-1/2,L)\tT(L+1/2,L)\tT_avg(L)")
    for e, ls in blocks:
        for l, tm, tp, ta in ls:
            jl.append(f"{e}\t{l}\t{tm}\t{tp}\t{ta}")
    wl = header_block(
        nuc, info, "neutron transmission coefficients T_l (spin-averaged), wide format",
        src + ["T_l column = TALYS column 'T(L)'"], "E in MeV; T dimensionless", notes,
    )
    wl.append("# E_lab\t" + "\t".join(f"T_l={l}" for l in range(lmax + 1)))
    for e, ls in blocks:
        vals = [t for _, _, _, t in ls] + [ZERO] * (lmax + 1 - len(ls))
        wl.append(e + "\t" + "\t".join(vals))
    meta = {
        "n_transmission_energies": len(blocks),
        "l_max_listed": lmax,
        "transmission_E_lab_over_E_grid": round(ratio, 6),
        "transmission_E_lab_first_MeV": first_e,
        "transmission_E_lab_last_MeV": float(blocks[-1][0]),
    }
    return "\n".join(jl) + "\n", "\n".join(wl) + "\n", meta


def build_metadata(nuc, info, tmeta, n_xs, emin, emax) -> str:
    cfg = NUCLEI[nuc]
    resid = f"rp{cfg['Z']:03d}{cfg['A'] + 1:03d}.tot"
    md = {
        "target": {"element": cfg["element"], "Z": cfg["Z"], "A": cfg["A"]},
        "projectile": "n",
        "talys": {
            "version": info["version"],
            "version_date": info["version_date"],
            "source_git_commit": info["git"],
            "run_date": info["date"],
        },
        "input_file": "talys_input.txt",
        "energy_grid_file": "../energies_golden.txt",
        "energy_frame": "laboratory (TALYS 'energy' keyword); E_cm = E_lab * M/(M+m_n)",
        "cross_sections": {
            "n_energies": n_xs,
            "E_lab_min_MeV": float(emin),
            "E_lab_max_MeV": float(emax),
            "units": "mb",
            "file": f"{nuc}_cross_sections.dat",
        },
        "transmission": {**tmeta, "files": [f"{nuc}_transmission.dat", f"{nuc}_transmission_jsplit.dat"]},
        "non_default_keywords": {"outinverse": "y", "outtransenergy": "y", "transpower": 20, "transeps": "1.e-14"},
        "raw_files": ["raw/binary.tot", f"raw/{resid}", "raw/all.tot", "raw/transmission_n.out", "raw/output.gz"],
        "generator": "scripts/extract_talys_golden.py",
    }
    return json.dumps(md, indent=2) + "\n"


def generate(nuc: str) -> dict[str, str]:
    info = run_info(nuc)
    xs = build_cross_sections(nuc, info)
    jsplit, wide, tmeta = build_transmission(nuc, info)
    es = [l.split("\t")[0] for l in xs.splitlines() if not l.startswith("#")]
    return {
        f"{nuc}_cross_sections.dat": xs,
        f"{nuc}_transmission.dat": wide,
        f"{nuc}_transmission_jsplit.dat": jsplit,
        "metadata.json": build_metadata(nuc, info, tmeta, len(es), es[0], es[-1]),
    }


def collect(nuc: str, rundir: Path) -> None:
    cfg = NUCLEI[nuc]
    resid = f"rp{cfg['Z']:03d}{cfg['A'] + 1:03d}.tot"
    dest = DATA / nuc / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("binary.tot", resid, "all.tot", "transmission_n.out"):
        src = rundir / name
        if not src.is_file():
            raise ExtractError(f"{src} not found (is the TALYS run finished?)")
        shutil.copyfile(src, dest / name)
    for name in ("output", "talys_input.txt"):
        if not (rundir / name).is_file():
            raise ExtractError(f"{rundir / name} not found")
    with open(rundir / "output", "rb") as fi, open(dest / "output.gz", "wb") as fo:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fo, mtime=0, compresslevel=9) as gz:
            shutil.copyfileobj(fi, gz)
    shutil.copyfile(rundir / "talys_input.txt", DATA / nuc / "talys_input.txt")
    print(f"collected raw files for {nuc} into {dest.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; fail if committed tables differ")
    ap.add_argument("--collect", nargs=2, metavar=("NUC", "RUNDIR"), help="copy raw files from a TALYS run directory")
    args = ap.parse_args()
    try:
        if args.collect:
            nuc, rundir = args.collect
            if nuc not in NUCLEI:
                raise ExtractError(f"unknown nucleus {nuc!r}; choose from {sorted(NUCLEI)}")
            collect(nuc, Path(rundir))
            return 0
        bad = 0
        for nuc in NUCLEI:
            for name, text in generate(nuc).items():
                path = DATA / nuc / name
                if args.check:
                    if not path.is_file() or path.read_text() != text:
                        print(f"DIFFERS: {path.relative_to(ROOT)}")
                        bad += 1
                else:
                    path.write_text(text)
                    print(f"wrote {path.relative_to(ROOT)}")
        return 1 if bad else 0
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
