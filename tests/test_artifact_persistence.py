import os
import tempfile
import torch
import pandas as pd
import pytest
from src.utils.artifact_persister import persist_experiment_artifacts, get_artifacts_root

def test_artifact_persistence_with_temp_files():
    """
    Verifies that persist_experiment_artifacts:
    1. Creates target directories automatically if they do not exist.
    2. Copies checkpoint (.pt) to <target_root>/checkpoints/
    3. Copies results CSV (.csv) to <target_root>/results/
    4. Preserves exact filenames and contents.
    """
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as target_root:
        # Create dummy checkpoint
        ckpt_filename = "DP-WHFedDG_seed42_eps3.0_final.pt"
        ckpt_src_path = os.path.join(src_dir, ckpt_filename)
        dummy_state = {"model_state": torch.tensor([1.0, 2.0, 3.0]), "round": 100}
        torch.save(dummy_state, ckpt_src_path)

        # Create dummy results CSV
        csv_filename = "phase10a_DP-WHFedDG_seed42_results.csv"
        csv_src_path = os.path.join(src_dir, csv_filename)
        dummy_df = pd.DataFrame([{"Method": "DP-WHFedDG", "Accuracy": 0.95, "AUROC": 0.98}])
        dummy_df.to_csv(csv_src_path, index=False)

        # Execute persistence
        persisted = persist_experiment_artifacts(
            checkpoint_path=ckpt_src_path,
            results_csv_path=csv_src_path,
            target_root=target_root
        )

        # 1. Verify destination paths and existence
        expected_ckpt_dest = os.path.join(target_root, "checkpoints", ckpt_filename)
        expected_csv_dest = os.path.join(target_root, "results", csv_filename)

        assert os.path.exists(expected_ckpt_dest), f"Checkpoint not found at {expected_ckpt_dest}"
        assert os.path.exists(expected_csv_dest), f"CSV not found at {expected_csv_dest}"
        assert persisted["checkpoint"] == expected_ckpt_dest
        assert persisted["results_csv"] == expected_csv_dest

        # 2. Verify filename preservation
        assert os.path.basename(expected_ckpt_dest) == ckpt_filename
        assert os.path.basename(expected_csv_dest) == csv_filename

        # 3. Verify content integrity
        loaded_ckpt = torch.load(expected_ckpt_dest)
        assert torch.equal(loaded_ckpt["model_state"], dummy_state["model_state"])
        assert loaded_ckpt["round"] == 100

        loaded_df = pd.read_csv(expected_csv_dest)
        assert loaded_df["Method"].iloc[0] == "DP-WHFedDG"
        assert loaded_df["Accuracy"].iloc[0] == 0.95

def test_artifact_persistence_creates_subdirs():
    """
    Verifies that nested subdirectories are created if they do not exist.
    """
    with tempfile.TemporaryDirectory() as base_dir:
        target_root = os.path.join(base_dir, "nested", "experiment_artifacts")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f_ckpt, \
             tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_csv:
            ckpt_path = f_ckpt.name
            csv_path = f_csv.name
            f_ckpt.write(b"dummy_pt_content")
            f_csv.write(b"dummy_csv_content")

        try:
            persisted = persist_experiment_artifacts(
                checkpoint_path=ckpt_path,
                results_csv_path=csv_path,
                target_root=target_root
            )
            assert os.path.isdir(os.path.join(target_root, "checkpoints"))
            assert os.path.isdir(os.path.join(target_root, "results"))
            assert os.path.exists(persisted["checkpoint"])
            assert os.path.exists(persisted["results_csv"])
        finally:
            os.remove(ckpt_path)
            os.remove(csv_path)

if __name__ == "__main__":
    test_artifact_persistence_with_temp_files()
    test_artifact_persistence_creates_subdirs()
    print("All artifact persistence tests passed!")
