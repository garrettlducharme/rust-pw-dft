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


def vector(element: ET.Element) -> list[float]:
    if element.text is None:
        raise ValueError(f"QE XML element {local_name(element.tag)} is empty")
    values = [float(value.replace("D", "E")) for value in element.text.split()]
    if len(values) != 3:
        raise ValueError(f"QE XML element {local_name(element.tag)} is not a vector")
    return values


def optional_descendant(parent: ET.Element, name: str) -> ET.Element | None:
    for value in parent.iter():
        if local_name(value.tag) == name:
            return value
    return None


def solve_reduced(reciprocal: list[list[float]], cartesian: list[float]) -> list[float]:
    # cartesian_j = sum_i reduced_i * reciprocal_i,j
    matrix = [
        [reciprocal[column][row] for column in range(3)] + [cartesian[row]]
        for row in range(3)
    ]
    for pivot in range(3):
        selected = max(range(pivot, 3), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[selected][pivot]) < 1.0e-14:
            raise ValueError("QE reciprocal lattice is singular")
        matrix[pivot], matrix[selected] = matrix[selected], matrix[pivot]
        scale = matrix[pivot][pivot]
        matrix[pivot] = [value / scale for value in matrix[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[pivot])
            ]
    return [wrap_reduced(matrix[row][3]) for row in range(3)]


def wrap_reduced(value: float) -> float:
    wrapped = value - int((value + 0.5) // 1)
    if abs(wrapped) <= 1.0e-12:
        return 0.0
    if abs(wrapped + 0.5) <= 1.0e-12:
        return -0.5
    return round(wrapped, 12)


def same_reduced(left: list[float], right: list[float]) -> bool:
    return all(abs(wrap_reduced(a - b)) <= 1.0e-10 for a, b in zip(left, right))


def reduce_time_reversal(
    points: list[dict[str, object]],
) -> tuple[list[dict[str, object]], float, float]:
    used = [False] * len(points)
    reduced: list[dict[str, object]] = []
    maximum_eigenvalue_error = 0.0
    maximum_occupation_error = 0.0
    for index, point in enumerate(points):
        if used[index]:
            continue
        used[index] = True
        coordinate = point["reduced"]
        assert isinstance(coordinate, list)
        opposite = [wrap_reduced(-value) for value in coordinate]
        pair = next(
            (
                candidate
                for candidate in range(index + 1, len(points))
                if not used[candidate]
                and same_reduced(points[candidate]["reduced"], opposite)  # type: ignore[arg-type]
            ),
            None,
        )
        if pair is None and not same_reduced(coordinate, opposite):
            raise ValueError(f"QE k point {coordinate} has no time-reversal partner")
        chosen = point.copy()
        if pair is not None:
            used[pair] = True
            paired = points[pair]
            eigenvalues = point["eigenvalues_ry"]
            paired_eigenvalues = paired["eigenvalues_ry"]
            occupations = point["occupations"]
            paired_occupations = paired["occupations"]
            assert isinstance(eigenvalues, list)
            assert isinstance(paired_eigenvalues, list)
            assert isinstance(occupations, list)
            assert isinstance(paired_occupations, list)
            if len(eigenvalues) != len(paired_eigenvalues):
                raise ValueError("QE time-reversal pair has different band counts")
            if len(occupations) != len(paired_occupations):
                raise ValueError("QE time-reversal pair has different occupation counts")
            eigenvalue_error = max(
                abs(a - b) for a, b in zip(eigenvalues, paired_eigenvalues)
            )
            occupation_error = max(
                abs(a - b) for a, b in zip(occupations, paired_occupations)
            )
            maximum_eigenvalue_error = max(maximum_eigenvalue_error, eigenvalue_error)
            maximum_occupation_error = max(maximum_occupation_error, occupation_error)
            if eigenvalue_error > 1.0e-5 or occupation_error > 1.0e-8:
                raise ValueError(
                    "QE time-reversal pair disagrees "
                    f"(eigenvalues {eigenvalue_error:e} Ry, occupations {occupation_error:e})"
                )
            point_weight = float(point["weight"])
            paired_weight = float(paired["weight"])
            combined_weight = point_weight + paired_weight
            chosen["eigenvalues_ry"] = [
                (point_weight * a + paired_weight * b) / combined_weight
                for a, b in zip(eigenvalues, paired_eigenvalues)
            ]
            chosen["occupations"] = [
                (point_weight * a + paired_weight * b) / combined_weight
                for a, b in zip(occupations, paired_occupations)
            ]
            paired_coordinate = paired["reduced"]
            assert isinstance(paired_coordinate, list)
            if tuple(paired_coordinate) > tuple(coordinate):
                chosen["reduced"] = paired_coordinate
            chosen["weight"] = combined_weight
        reduced.append(chosen)
    reduced.sort(key=lambda point: tuple(point["reduced"]))
    return reduced, maximum_eigenvalue_error, maximum_occupation_error


def normalize_k_points(
    raw_points: list[dict[str, object]], reduce_pairs: bool
) -> tuple[list[dict[str, object]], float, float]:
    if not raw_points:
        raise ValueError("QE XML contains no k-point eigenvalues")
    raw_weight_sum = sum(float(point["weight"]) for point in raw_points)
    if raw_weight_sum <= 0.0:
        raise ValueError("QE k-point weights have a nonpositive sum")
    for point in raw_points:
        point["weight"] = float(point["weight"]) / raw_weight_sum
        point["occupations"] = [
            raw_weight_sum * value for value in point["occupations"]  # type: ignore[union-attr]
        ]
    if reduce_pairs:
        return reduce_time_reversal(raw_points)
    return sorted(raw_points, key=lambda point: tuple(point["reduced"])), 0.0, 0.0


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
    parser.add_argument(
        "--reduce-time-reversal",
        action="store_true",
        help="combine verified k/-k pairs into the same convention as the Rust solver",
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
    output = descendant(root, "output")
    basis_set = descendant(output, "basis_set")
    reciprocal_element = child(basis_set, "reciprocal_lattice")
    reciprocal = [vector(child(reciprocal_element, f"b{axis}")) for axis in range(1, 4)]
    lsda_text = child(band_structure, "lsda").text
    if lsda_text is None:
        raise ValueError("QE XML band_structure/lsda is empty")
    is_collinear_spin = lsda_text.strip().lower() == "true"
    channel_band_counts: list[int] | None = None
    if is_collinear_spin:
        channel_band_counts = [
            int(scalar(band_structure, "nbnd_up")),
            int(scalar(band_structure, "nbnd_dw")),
        ]
        if any(count <= 0 for count in channel_band_counts):
            raise ValueError("QE XML has an invalid collinear-spin band count")

    raw_k_points: list[dict[str, object]] = []
    raw_spin_k_points: list[list[dict[str, object]]] = [[], []]
    for ks_energies in band_structure:
        if local_name(ks_energies.tag) != "ks_energies":
            continue
        k_point = child(ks_energies, "k_point")
        eigenvalues_text = child(ks_energies, "eigenvalues").text
        occupations_text = child(ks_energies, "occupations").text
        if eigenvalues_text is None or occupations_text is None:
            raise ValueError("QE XML eigenvalues or occupations are empty")
        eigenvalues = [
            2.0 * float(value.replace("D", "E"))
            for value in eigenvalues_text.split()
        ]
        occupations = [
            float(value.replace("D", "E"))
            for value in occupations_text.split()
        ]
        base = {
            "reduced": solve_reduced(reciprocal, vector(k_point)),
            "weight": float(k_point.attrib["weight"]),
            "residuals": [],
        }
        if is_collinear_spin:
            assert channel_band_counts is not None
            if len(eigenvalues) != sum(channel_band_counts) or len(occupations) != sum(
                channel_band_counts
            ):
                raise ValueError(
                    "QE collinear-spin eigenvalue dimensions disagree with nbnd_up/nbnd_dw"
                )
            offset = 0
            for channel, count in enumerate(channel_band_counts):
                raw_spin_k_points[channel].append(
                    base
                    | {
                        "eigenvalues_ry": eigenvalues[offset : offset + count],
                        "occupations": occupations[offset : offset + count],
                    }
                )
                offset += count
        else:
            raw_k_points.append(
                base
                | {
                    "eigenvalues_ry": eigenvalues,
                    "occupations": occupations,
                }
            )

    if is_collinear_spin:
        normalized_spin = [
            normalize_k_points(points, args.reduce_time_reversal)
            for points in raw_spin_k_points
        ]
        spin_k_points = [entry[0] for entry in normalized_spin]
        if [point["reduced"] for point in spin_k_points[0]] != [
            point["reduced"] for point in spin_k_points[1]
        ]:
            raise ValueError("QE collinear-spin channels use different k-point sets")
        k_points: list[dict[str, object]] = []
        eigenvalues_ry: list[float] = []
        maximum_time_reversal_eigenvalue_error_ry = max(
            entry[1] for entry in normalized_spin
        )
        maximum_time_reversal_occupation_error = max(
            entry[2] for entry in normalized_spin
        )
    else:
        (
            k_points,
            maximum_time_reversal_eigenvalue_error_ry,
            maximum_time_reversal_occupation_error,
        ) = normalize_k_points(raw_k_points, args.reduce_time_reversal)
        eigenvalues_ry = [
            value
            for point in k_points
            for value in point["eigenvalues_ry"]  # type: ignore[union-attr]
        ]

    stdout = args.stdout.read_text()
    iterations = re.findall(r"convergence has been achieved in\s+(\d+) iterations", stdout)
    if not iterations:
        raise ValueError("QE output has no successful SCF iteration count")
    atomic_positions = descendant(descendant(output, "atomic_structure"), "atomic_positions")
    atom_count = len(
        [value for value in atomic_positions if local_name(value.tag) == "atom"]
    )
    if atom_count == 0:
        raise ValueError("QE XML contains no atoms")
    forces_element = optional_descendant(output, "forces")
    forces_ry_bohr: list[list[float]] | None = None
    if forces_element is not None:
        if forces_element.text is None:
            raise ValueError("QE XML forces are empty")
        raw_forces = [
            2.0 * float(value.replace("D", "E"))
            for value in forces_element.text.split()
        ]
        if len(raw_forces) != 3 * atom_count:
            raise ValueError("QE XML force dimensions do not match the atom count")
        forces_ry_bohr = [
            raw_forces[offset : offset + 3]
            for offset in range(0, len(raw_forces), 3)
        ]

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

    energy_terms = {
        "band_energy_ry": 2.0 * scalar(total_energy, "eband"),
        "hartree_ry": 2.0 * scalar(total_energy, "ehart"),
        "integral_rho_vxc_ry": 2.0 * scalar(total_energy, "vtxc"),
        "xc_ry": 2.0 * scalar(total_energy, "etxc"),
        "ewald_ry": 2.0 * scalar(total_energy, "ewald"),
    }
    if args.reduce_time_reversal:
        energy_terms.update(
            {
                "time_reversal_pair_max_eigenvalue_error_ry": maximum_time_reversal_eigenvalue_error_ry,
                "time_reversal_pair_max_occupation_error": maximum_time_reversal_occupation_error,
            }
        )

    result = {
        "schema_version": 5 if is_collinear_spin else (4 if forces_ry_bohr is not None else 3),
        "converged": True,
        "iterations": int(iterations[-1]),
        "atom_count": atom_count,
        "total_energy_ry": 2.0 * scalar(total_energy, "etot"),
        "energy_terms": energy_terms,
        "eigenvalues_ry": eigenvalues_ry,
        "electron_count": scalar(band_structure, "nelec"),
        "residuals": [],
        "k_points": k_points,
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
    if is_collinear_spin:
        channel_electrons = [
            sum(
                float(point["weight"])
                * sum(float(value) for value in point["occupations"])  # type: ignore[union-attr]
                for point in points
            )
            for points in spin_k_points
        ]
        result["spin"] = {
            "magnetization_electrons": channel_electrons[0] - channel_electrons[1],
            "up": {
                "electron_count": channel_electrons[0],
                "k_points": spin_k_points[0],
            },
            "down": {
                "electron_count": channel_electrons[1],
                "k_points": spin_k_points[1],
            },
        }
    if forces_ry_bohr is not None:
        result["forces_ry_bohr"] = forces_ry_bohr
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
