#!/usr/bin/env python3
"""Normalize a completed QE 7.6 SCF XML/output pair into ScfResult JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(parent: ET.Element, name: str) -> ET.Element:
    for value in parent:
        if local_name(value.tag) == name:
            return value
    raise ValueError(f"QE XML is missing {name} below {local_name(parent.tag)}")


def descendant(parent: ET.Element, name: str) -> ET.Element:
    for value in parent.iter():
        if local_name(value.tag) == name:
            return value
    raise ValueError(f"QE XML is missing {name}")


def scalar(parent: ET.Element, name: str) -> float:
    text = child(parent, name).text
    if text is None:
        raise ValueError(f"QE XML element {name} is empty")
    return float(text.replace("D", "E"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path)
    parser.add_argument("stdout", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "pseudopotential",
        type=Path,
        nargs="?",
        help="legacy single-species pseudopotential path",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--qe-revision", required=True)
    parser.add_argument("--element", default="Si")
    parser.add_argument(
        "--pseudo",
        action="append",
        default=[],
        metavar="ELEMENT=PATH",
        help="repeat for every species in a multi-species calculation",
    )
    args = parser.parse_args()

    pseudopotentials: dict[str, Path] = {}
    if args.pseudopotential is not None:
        pseudopotentials[args.element] = args.pseudopotential
    for specification in args.pseudo:
        if "=" not in specification:
            parser.error("--pseudo must have the form ELEMENT=PATH")
        element, raw_path = specification.split("=", 1)
        if not element or element in pseudopotentials:
            parser.error(f"duplicate or empty pseudopotential element: {element!r}")
        pseudopotentials[element] = Path(raw_path)
    if not pseudopotentials:
        parser.error("supply the legacy pseudopotential path or at least one --pseudo")
    missing = [str(path) for path in pseudopotentials.values() if not path.is_file()]
    if missing:
        parser.error(f"pseudopotential file does not exist: {', '.join(missing)}")

    root = ET.parse(args.xml).getroot()
    total_energy = descendant(root, "total_energy")
    band_structure = descendant(root, "band_structure")
    ks_energies = descendant(band_structure, "ks_energies")
    eigenvalues_text = child(ks_energies, "eigenvalues").text
    if eigenvalues_text is None:
        raise ValueError("QE XML eigenvalues are empty")
    eigenvalues_ry = [2.0 * float(value.replace("D", "E")) for value in eigenvalues_text.split()]

    stdout = args.stdout.read_text()
    iterations = re.findall(r"convergence has been achieved in\s+(\d+) iterations", stdout)
    if not iterations:
        raise ValueError("QE output has no successful SCF iteration count")
    output = descendant(root, "output")
    atomic_positions = descendant(descendant(output, "atomic_structure"), "atomic_positions")
    atom_count = len(
        [value for value in atomic_positions if local_name(value.tag) == "atom"]
    )
    if atom_count == 0:
        raise ValueError("QE XML contains no atoms")

    timing_total = next(
        (
            value
            for value in root.iter()
            if local_name(value.tag) == "total" and value.attrib.get("label") == "PWSCF"
        ),
        None,
    )
    if timing_total is None:
        raise ValueError("QE XML has no PWSCF total timing")

    result = {
        "schema_version": 2,
        "converged": True,
        "iterations": int(iterations[-1]),
        "atom_count": atom_count,
        "total_energy_ry": 2.0 * scalar(total_energy, "etot"),
        "energy_terms": {
            "band_energy_ry": 2.0 * scalar(total_energy, "eband"),
            "hartree_ry": 2.0 * scalar(total_energy, "ehart"),
            "integral_rho_vxc_ry": 2.0 * scalar(total_energy, "vtxc"),
            "xc_ry": 2.0 * scalar(total_energy, "etxc"),
            "ewald_ry": 2.0 * scalar(total_energy, "ewald"),
        },
        "eigenvalues_ry": eigenvalues_ry,
        "electron_count": scalar(band_structure, "nelec"),
        "residuals": [],
        "history": [],
        "history_complete": False,
        "timings": {"wall_seconds": scalar(timing_total, "wall")},
        "provenance": {
            "requested_backend": "cpu-f64",
            "actual_backend": "cpu-f64",
            "actual_device": "Quantum ESPRESSO 7.6 CPU reference",
            "precision_policy": {
                "storage": "complex-f64",
                "reductions": "f64",
                "eigensolve": "f64",
                "energy": "f64",
            },
            "input_sha256": sha256(args.input),
            "pseudopotential_sha256": {
                element: sha256(path)
                for element, path in sorted(pseudopotentials.items())
            },
            "git_revision": args.qe_revision,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
