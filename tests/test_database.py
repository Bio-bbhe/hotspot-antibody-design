import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_DEPS = all(
    importlib.util.find_spec(name)
    for name in ("numpy", "pandas")
)


@unittest.skipUnless(HAS_DEPS, "NumPy/Pandas are not installed")
class DatabaseTests(unittest.TestCase):
    def test_union_schema_json_paths_and_deduplication(self):
        import numpy as np
        import pandas as pd

        from src.data.database import BindSculptDatabase

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output_pdb = run_dir / "esmfold2_outputs" / "a.pdb"
            output_pdb.parent.mkdir()
            output_pdb.touch()
            database = BindSculptDatabase(run_dir)
            first = {
                "id": "a",
                "binder_sequence": "ACDC",
                "esmfold2_status": "success",
                "iptm": 0.8,
                "complex_pdb_path": str(output_pdb),
                "cdr_positions": [1, 2],
                "concat_embedding": np.array([1.0, 2.0]),
            }
            second = {
                "id": "b",
                "binder_sequence": "ACEC",
                "esmfold2_status": "success",
                "iptm": 0.9,
                "new_metric": 7,
                "fixed_interface_positions": {"B": [1, 4]},
                "concat_embedding": np.array([3.0, 4.0]),
            }
            database.save_evaluated_batch([first])
            database.save_evaluated_batch([second])
            database.save_evaluated_batch([
                {
                    **first,
                    "iptm": 0.85,
                    "concat_embedding": np.array([5.0, 6.0]),
                }
            ])

            frame = pd.read_csv(database.csv_path)
            self.assertEqual(len(frame), 2)
            self.assertIn("new_metric", frame.columns)
            records = {
                row["id"]: row
                for row in database.get_all_training_data()
            }
            self.assertEqual(records["a"]["cdr_positions"], [1, 2])
            self.assertEqual(
                records["b"]["fixed_interface_positions"],
                {"B": [1, 4]},
            )
            self.assertTrue(
                Path(records["a"]["complex_pdb_path"]).is_absolute()
            )
            np.testing.assert_array_equal(
                records["a"]["concat_embedding"],
                np.array([5.0, 6.0]),
            )

            database.save_evaluated_batch([{
                **first,
                "id": "c",
                "concat_embedding": np.array([7.0, 8.0]),
            }])
            frame = pd.read_csv(database.csv_path)
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame["binder_sequence"].is_unique)
            self.assertEqual(set(frame["id"]), {"b", "c"})



if __name__ == "__main__":
    unittest.main()
