import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_DEPS = all(
    importlib.util.find_spec(name)
    for name in ("Bio", "numpy", "scipy")
)


def atom_line(serial, chain, residue, x):
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"  1.00 80.00           C\n"
    )


@unittest.skipUnless(HAS_DEPS, "BioPython/SciPy are not installed")
class GeometryContactTests(unittest.TestCase):
    def test_cdr_fraction_uses_residues_not_atom_counts(self):
        from src.utils.geometry import HotspotGeometry

        pdb = (
            atom_line(1, "A", 1, 0.0)
            + atom_line(2, "A", 2, 30.0)
            + atom_line(3, "B", 1, 4.0)
            + atom_line(4, "B", 2, 20.0)
            + "END\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.pdb"
            path.write_text(pdb)
            geometry = HotspotGeometry()
            whole_interface = geometry.hotspot_contact_metrics(
                path, "", [0], distance_threshold=5.0
            )
            wrong_hotspot = geometry.hotspot_contact_metrics(
                path, "2", [0], distance_threshold=5.0
            )

        self.assertEqual(
            whole_interface["hotspot_residue_contacts"], 1
        )
        self.assertEqual(
            whole_interface["cdr_hotspot_contacts"], 1
        )
        self.assertEqual(
            whole_interface["cdr_contact_fraction"], 1.0
        )
        self.assertEqual(
            wrong_hotspot["cdr_hotspot_contacts"], 0
        )


if __name__ == "__main__":
    unittest.main()
