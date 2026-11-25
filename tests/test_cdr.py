import importlib.util
import unittest

from src.utils.cdr import (
    AntibodySequenceError,
    annotate_cdrs,
    get_antibody_cdr_indices,
    split_antibody_domains,
    validate_antibody_sequence,
)


VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNVYYSSIHWVRQAPGKGLEWVAYISPSSGSTYY"
    "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARKQYSYWRDSYWAMDYWGQG"
)
VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQSVSSAVAWYQQKPGKAPKLLIYSASSLYSGVP"
    "SRFSGSGSGTDFTLTISSLQPEDFATYYCQQHQYNSLIFGQG"
)
LINKER = "GGGGSGGGGSGGGGS"


class AntibodySequenceTests(unittest.TestCase):
    def test_validates_vhh_and_scfv_layouts(self):
        self.assertEqual(
            validate_antibody_sequence(VH, ["VHH"]),
            VH,
        )
        scfv = VH + LINKER + VL
        self.assertEqual(
            validate_antibody_sequence(scfv, ["VH", "VL"], LINKER),
            scfv,
        )
        domains = split_antibody_domains(
            scfv, ["VH", "VL"], LINKER
        )
        self.assertEqual(
            [(kind, offset) for kind, _sequence, offset in domains],
            [("VH", 0), ("VL", len(VH) + len(LINKER))],
        )

    def test_rejects_noncanonical_and_lost_disulfides(self):
        with self.assertRaises(AntibodySequenceError):
            validate_antibody_sequence(VH.replace("A", "X", 1), ["VHH"])
        with self.assertRaises(AntibodySequenceError):
            validate_antibody_sequence(VH.replace("C", "A"), ["VHH"])

    def test_fallback_is_explicit(self):
        ranges = annotate_cdrs(
            VH, "VH", require_imgt=False
        )
        self.assertEqual(set(ranges), {"cdr1", "cdr2", "cdr3"})

    @unittest.skipUnless(
        importlib.util.find_spec("abnumber"),
        "AbNumber is not installed",
    )
    def test_imgt_maps_both_scfv_domains(self):
        indices = get_antibody_cdr_indices(
            VH + LINKER + VL,
            ["VH", "VL"],
            LINKER,
        )
        self.assertTrue(any(index < len(VH) for index in indices))
        self.assertTrue(
            any(index >= len(VH) + len(LINKER) for index in indices)
        )
        self.assertFalse(
            set(indices)
            & set(range(len(VH), len(VH) + len(LINKER)))
        )


if __name__ == "__main__":
    unittest.main()
