"""Antibody sequence validation and IMGT CDR mapping."""

from __future__ import annotations

from functools import lru_cache


STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class AntibodySequenceError(ValueError):
    """Raised when a sequence cannot represent the configured antibody format."""


def validate_standard_sequence(sequence: str, name: str = "sequence") -> str:
    if not isinstance(sequence, str):
        raise AntibodySequenceError(f"{name} must be a string")
    seq = sequence.strip().upper()
    invalid = sorted(set(seq) - STANDARD_AAS)
    if not seq:
        raise AntibodySequenceError(f"{name} is empty")
    if invalid:
        raise AntibodySequenceError(
            f"{name} contains noncanonical residue(s): {', '.join(invalid)}"
        )
    return seq


def _fallback_cdrs(sequence: str) -> dict[str, list[int]]:
    """Conservative fallback for explicit non-production use only."""
    bounds = {"cdr1": (26, 37), "cdr2": (55, 64), "cdr3": (104, 116)}
    return {
        name: [start, min(end, len(sequence) - 1)]
        for name, (start, end) in bounds.items()
        if start < len(sequence)
    }


def _ranges_from_numbered(numbered, chain_type: str) -> dict[str, list[int]]:
    """Convert one AbNumber Chain to 0-based IMGT CDR ranges."""
    is_heavy = numbered.is_heavy_chain()
    if (chain_type in {"VH", "VHH"}) != is_heavy:
        expected = "heavy" if chain_type in {"VH", "VHH"} else "light"
        raise AntibodySequenceError(
            f"Configured {chain_type} sequence was numbered as a non-{expected} chain"
        )

    indices: dict[str, list[int]] = {"cdr1": [], "cdr2": [], "cdr3": []}
    for raw_index, (position, _aa) in enumerate(numbered.positions.items()):
        region = position.get_region().lower()
        if region in indices:
            indices[region].append(raw_index)

    missing = [name for name, values in indices.items() if not values]
    if missing:
        raise AntibodySequenceError(
            f"IMGT numbering did not identify: {', '.join(missing)}"
        )
    return {name: [values[0], values[-1]] for name, values in indices.items()}


@lru_cache(maxsize=None)
def annotate_cdrs(
    sequence: str,
    chain_type: str = "VHH",
    *,
    require_imgt: bool = True,
) -> dict[str, list[int]]:
    """Return 0-based CDR ranges after numbering one V-domain with AbNumber."""
    seq = validate_standard_sequence(sequence)
    kind = chain_type.upper()
    if kind not in {"VH", "VHH", "VL"}:
        raise AntibodySequenceError(f"Unsupported antibody chain type: {chain_type}")

    try:
        from abnumber import Chain

        numbered = Chain(
            seq,
            scheme="imgt",
            cdr_definition="imgt",
            use_anarcii=True,
            anarcii_args={"cpu": True},
        )
    except ImportError as exc:
        if not require_imgt:
            return _fallback_cdrs(seq)
        raise AntibodySequenceError(
            "AbNumber is required for IMGT CDR mapping; install the project dependencies"
        ) from exc
    except Exception as exc:
        raise AntibodySequenceError(f"IMGT numbering failed for {kind}: {exc}") from exc

    return _ranges_from_numbered(numbered, kind)


def get_cdr_indices(
    sequence: str,
    chain_type: str = "VHH",
    *,
    require_imgt: bool = True,
) -> list[int]:
    """Return 0-based indices belonging to any CDR in one V-domain."""
    ranges = annotate_cdrs(sequence, chain_type, require_imgt=require_imgt)
    return sorted(
        index
        for start, end in ranges.values()
        for index in range(start, end + 1)
    )


def _normalise_chain_types(chain_types) -> list[str]:
    if isinstance(chain_types, str):
        chain_types = [chain_types]
    values = [str(value).upper() for value in (chain_types or ["VHH"])]
    if values == ["SCFV"]:
        values = ["VH", "VL"]
    return values


def split_antibody_domains(
    sequence: str,
    chain_types,
    linker: str = "",
) -> list[tuple[str, str, int]]:
    """Return (chain_type, domain_sequence, offset) for VHH or scFv."""
    seq = validate_standard_sequence(sequence)
    kinds = _normalise_chain_types(chain_types)
    if len(kinds) == 1:
        return [(kinds[0], seq, 0)]
    if kinds != ["VH", "VL"]:
        raise AntibodySequenceError(f"Unsupported antibody domain layout: {kinds}")

    linker = validate_standard_sequence(linker, "scFv linker")
    if seq.count(linker) != 1:
        raise AntibodySequenceError(
            "scFv sequence must contain its configured linker exactly once"
        )
    vh, vl = seq.split(linker)
    if not vh or not vl:
        raise AntibodySequenceError(
            "scFv linker must separate non-empty VH and VL domains"
        )
    return [("VH", vh, 0), ("VL", vl, len(vh) + len(linker))]



def get_antibody_cdr_indices(
    sequence: str,
    chain_types,
    linker: str = "",
    *,
    require_imgt: bool = True,
) -> list[int]:
    """Return combined-sequence CDR indices for VHH or VH-linker-VL scFv."""
    indices: list[int] = []
    for kind, domain, offset in split_antibody_domains(sequence, chain_types, linker):
        indices.extend(
            offset + index
            for index in get_cdr_indices(
                domain, kind, require_imgt=require_imgt
            )
        )
    return sorted(indices)


def validate_antibody_sequence(
    sequence: str,
    chain_types,
    linker: str = "",
) -> str:
    """Validate alphabet, scFv layout, and conserved domain disulfides."""
    seq = validate_standard_sequence(sequence)
    for kind, domain, _offset in split_antibody_domains(seq, chain_types, linker):
        if domain.count("C") < 2:
            raise AntibodySequenceError(
                f"{kind} domain has fewer than two cysteines"
            )
    return seq
