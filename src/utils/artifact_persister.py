import os
import shutil

KAGGLE_ARTIFACTS_DIR = "/kaggle/working/experiment_artifacts"
LOCAL_FALLBACK_DIR = "./experiment_artifacts"

def get_artifacts_root():
    """
    Returns /kaggle/working/experiment_artifacts if /kaggle/working exists or can be created,
    otherwise falls back to ./experiment_artifacts for local execution/testing.
    """
    if os.path.exists("/kaggle/working"):
        return KAGGLE_ARTIFACTS_DIR
    elif os.environ.get("KAGGLE_WORKING_DIR"):
        return os.path.join(os.environ["KAGGLE_WORKING_DIR"], "experiment_artifacts")
    elif os.name != "nt" and os.path.exists("/kaggle"):
        return KAGGLE_ARTIFACTS_DIR
    else:
        try:
            os.makedirs("/kaggle/working/experiment_artifacts", exist_ok=True)
            return KAGGLE_ARTIFACTS_DIR
        except Exception:
            return LOCAL_FALLBACK_DIR

def persist_experiment_artifacts(checkpoint_path=None, results_csv_path=None, target_root=None):
    """
    Copies completed experiment artifacts:
      1. Checkpoint (.pt) -> <target_root>/checkpoints/
      2. Results CSV (.csv) -> <target_root>/results/
    
    Default target_root is /kaggle/working/experiment_artifacts (with local fallback).
    Creates directories automatically if they do not exist.
    """
    if target_root is None:
        target_root = get_artifacts_root()

    checkpoints_dest_dir = os.path.join(target_root, "checkpoints")
    results_dest_dir = os.path.join(target_root, "results")

    os.makedirs(checkpoints_dest_dir, exist_ok=True)
    os.makedirs(results_dest_dir, exist_ok=True)

    persisted = {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        dest_path = os.path.join(checkpoints_dest_dir, filename)
        if os.path.abspath(checkpoint_path) != os.path.abspath(dest_path):
            shutil.copy2(checkpoint_path, dest_path)
        persisted["checkpoint"] = dest_path
        print(f"[Artifact Persister] Checkpoint saved/copied to: {dest_path}")

    if results_csv_path and os.path.exists(results_csv_path):
        filename = os.path.basename(results_csv_path)
        dest_path = os.path.join(results_dest_dir, filename)
        if os.path.abspath(results_csv_path) != os.path.abspath(dest_path):
            shutil.copy2(results_csv_path, dest_path)
        persisted["results_csv"] = dest_path
        print(f"[Artifact Persister] Results CSV saved/copied to: {dest_path}")

    return persisted
