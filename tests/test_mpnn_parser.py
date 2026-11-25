import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_DEPS = all(
    importlib.util.find_spec(name)
    for name in ("abnumber", "Bio", "numpy", "scipy")
)

VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNVYYSSIHWVRQAPGKGLEWVAYISPSSGSTYY"
    "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARKQYSYWRDSYWAMDYWGQG"
)
VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQSVSSAVAWYQQKPGKAPKLLIYSASSLYSGVP"
    "SRFSGSGSGTDFTLTISSLQPEDFATYYCQQHQYNSLIFGQG"
)


@unittest.skipUnless(
    HAS_DEPS, "AbNumber/BioPython/SciPy are not installed"
)
class MPNNParserTests(unittest.TestCase):
    def test_invalid_native_pose_is_skipped(self):
        from src.models.mpnn_mutator import MPNNMutator

        mutator = MPNNMutator.__new__(MPNNMutator)
        mutator.linker = "GGGGSGGGGSGGGGS"
        mutator.chain_types = {"B": "VH", "C": "VL"}
        mutator.config = SimpleNamespace(chains_to_design=["B", "C"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.fa"
            path.write_text(">native\nXXXX/XXXX\n")
            records = mutator._parse_mpnn_fasta(
                path, 10, {"B": [], "C": []}, ["B", "C"]
            )
        self.assertEqual(records, [])
    def test_skips_native_invalid_and_duplicate_records(self):
        from src.models.mpnn_mutator import MPNNMutator

        mutator = MPNNMutator.__new__(MPNNMutator)
        mutator.linker = "GGGGSGGGGSGGGGS"
        mutator.chain_types = {"B": "VH", "C": "VL"}
        mutator.config = SimpleNamespace(
            chains_to_design=["B", "C"]
        )
        valid_vh = "Q" + VH[1:]
        invalid_vh = valid_vh[:20] + "X" + valid_vh[21:]
        fasta = (
            ">native, score=0.5\n"
            f"{VH}/{VL}\n"
            ">T=0.1, sample=1, score=-0.1234\n"
            f"{valid_vh}/{VL}\n"
            ">T=0.1, sample=2, score=-0.2\n"
            f"{invalid_vh}/{VL}\n"
            ">T=0.1, sample=3, score=-0.1234\n"
            f"{valid_vh}/{VL}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.fa"
            path.write_text(fasta)
            records = mutator._parse_mpnn_fasta(
                path,
                max_seqs=10,
                fixed_map={"B": [1], "C": [1]},
                design_chains=["B", "C"],
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["mpnn_score"], -0.1234)
        self.assertNotIn("X", records[0]["binder_sequence"])
        self.assertNotEqual(
            records[0]["binder_sequence"],
            VH + mutator.linker + VL,
        )
        self.assertTrue(records[0]["cdr_positions"])


if __name__ == "__main__":
    unittest.main()
