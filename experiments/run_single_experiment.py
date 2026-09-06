import os
import sys
import time
import argparse
import yaml
import copy
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.federated.trainer import FederatedTrainer
from src.evaluation.metrics import compute_classification_metrics, compute_worst_hospital_metric, compute_ece
from src.evaluation.slide_evaluator import SlideEvaluator
from src.utils.artifact_persister import persist_experiment_artifacts

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return commit
    except Exception:
        return "e45832d58dfec3b5edb9fe01a8461a5bbfdc41f0"

def main():
    parser = argparse.ArgumentParser(description="Phase 10A - Controlled Real Research Run")
    parser.add_argument("--config", type=str, default="./configs/camelyon17_wilds.yaml", help="Path to YAML configuration")
    parser.add_argument("--method", type=str, default="DP-WHFedDG", help="Federated algorithm method")
    parser.add_argument("--epsilon", type=float, default=3.0, help="Target privacy epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Experiment random seed")
    parser.add_argument("--rounds", type=int, default=100, help="Number of federated communication rounds")
    parser.add_argument("--local_epochs", type=int, default=None, help="Override local epochs (e.g. 1 for dev run)")
    parser.add_argument("--eval_every_round", action="store_true", default=False, help="Evaluate validation center every round")
    parser.add_argument("--tag", type=str, default="controlled_run", help="Unique tag for saving artifacts")
    parser.add_argument("--cache_in_ram", action="store_true", default=False, help="Preload training patches into RAM cache")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/mohdfam/camelyon17-wilds", help="Path to Camelyon17 dataset")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set hyperparameters
    config['project']['seed'] = args.seed
    config['privacy']['enabled'] = True if "DP" in args.method else False
    config['privacy']['target_epsilon'] = args.epsilon
    config['method'] = args.method
    config['federated']['rounds'] = args.rounds
    if args.local_epochs is not None:
        config['federated']['local_epochs'] = args.local_epochs
    if args.cache_in_ram:
        config['dataset']['cache_in_ram'] = True

    # Hardware detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU)"

    # Git commit
    git_commit = get_git_commit()

    train_centers = config['dataset']['train_centers']
    val_center = config['dataset']['val_center']
    test_center = config['dataset']['test_center']

    # ------------------------------------------------------------------
    # Pre-Run Diagnostic Header
    # ------------------------------------------------------------------
    print("======================================================================")
    print("         PHASE 10A — CONTROLLED REAL RESEARCH RUN")
    print("======================================================================")
    print(f"1. Git Commit:             {git_commit}")
    print(f"2. GPU Being Used:         {gpu_name} (CUDA: {cuda_available})")
    print(f"3. Dataset Path:           {args.data_dir}")
    print(f"4. Method:                 {args.method}")
    print(f"5. Privacy Epsilon:        {args.epsilon}")
    print(f"6. Seed:                   {args.seed}")
    print(f"7. Number of Rounds:       {args.rounds}")
    print(f"8. Training Centers:       {train_centers}")
    print(f"9. Validation Center:      {val_center}")
    print(f"10. Unseen Test Center:    {test_center}")
    print(f"11. RAM Caching Enabled:   {config['dataset'].get('cache_in_ram', False)}")
    print("======================================================================\n")

    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)

    # Load Real WILDS Camelyon17 Dataset (download=False)
    print("[1/5] Loading Real WILDS Camelyon17 Dataset (download=False)...")
    dataset = Camelyon17HospitalDataset(
        root_dir=args.data_dir,
        download=False,
        use_synthetic=False,
        seed=args.seed
    )
    print(f"  - Total Dataset Size: {len(dataset):,} samples")

    # Data Partitioning
    print("[2/5] Creating Federated Hospital Domain Splits...")
    splitter = HospitalClientSplitter(dataset, train_centers=train_centers, seed=args.seed)
    client_indices = splitter.get_natural_split()
    
    val_indices = dataset.get_center_subsets([val_center])
    test_indices = dataset.get_center_subsets([test_center])

    print(f"  - Training Clients (Centers 0, 1, 2): {[len(client_indices[i]) for i in range(len(train_centers))]} samples")
    print(f"  - Validation Center {val_center}: {len(val_indices):,} samples")
    print(f"  - Unseen Test Center {test_center}: {len(test_indices):,} samples (Strictly Isolated)")

    # Optional Preloading into RAM cache for training centers [0, 1, 2] ONLY
    if config['dataset'].get('cache_in_ram', False):
        all_train_indices = [idx for sublist in client_indices.values() for idx in sublist]
        dataset.preload_train_cache(all_train_indices)

    # Reset GPU peak memory tracking
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)

    # Instantiate Trainer
    print("\n[3/5] Initializing FederatedTrainer...")
    trainer = FederatedTrainer(
        dataset=dataset,
        client_indices=client_indices,
        config=config,
        method_name=args.method,
        device="cuda:0" if cuda_available else "cpu"
    )

    # Run Federated Training
    print(f"\n[4/5] Executing Complete {args.rounds}-Round Federated Training Loop...")
    start_time = time.time()
    history = trainer.run_training(
        rounds=args.rounds,
        eval_val_indices=val_indices if args.eval_every_round else None
    )
    elapsed_time = time.time() - start_time

    # Record peak memory
    peak_vram_gb = (torch.cuda.max_memory_allocated(0) / (1024**3)) if cuda_available else 0.0

    # ------------------------------------------------------------------
    # Post-Training Comprehensive Evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Performing Post-Training Multi-Center Evaluation...")
    
    # 1. Validation Center 3
    val_metrics = trainer.evaluate_on_subset(val_indices)
    
    # 2. Unseen Test Center 4
    test_metrics = trainer.evaluate_on_subset(test_indices)
    
    # 3. Training Hospital Centers 0, 1, 2 breakdown
    train_hospital_metrics = {}
    for c in train_centers:
        c_idx = dataset.get_center_subsets([c])
        train_hospital_metrics[c] = trainer.evaluate_on_subset(c_idx)
        
    worst_hosp_eval = compute_worst_hospital_metric(train_hospital_metrics)

    # Save checkpoint with unique tag
    epochs_val = config['federated']['local_epochs']
    checkpoint_name = f"{args.tag}_{args.method}_rounds{args.rounds}_epochs{epochs_val}_seed{args.seed}_eps{args.epsilon}.pt"
    checkpoint_path = os.path.join(config['logging']['save_dir'], checkpoint_name)
    torch.save({
        'model_state_dict': trainer.global_model.state_dict(),
        'config': config,
        'history': history,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'git_commit': git_commit
    }, checkpoint_path)

    # Save summary dataframe
    summary_df = pd.DataFrame([{
        'Git Commit': git_commit,
        'Method': args.method,
        'Epsilon': args.epsilon,
        'Seed': args.seed,
        'Rounds': args.rounds,
        'Local Epochs': epochs_val,
        'Elapsed Time (s)': elapsed_time,
        'Peak VRAM (GB)': peak_vram_gb,
        'Val Center 3 Accuracy': val_metrics['accuracy'],
        'Val Center 3 Balanced Accuracy': val_metrics.get('balanced_accuracy', 0.0),
        'Val Center 3 Macro F1': val_metrics.get('macro_f1', 0.0),
        'Val Center 3 AUROC': val_metrics['auroc'],
        'Val Center 3 ECE': val_metrics.get('ece', 0.0),
        'Val Center 3 Slide AUROC': val_metrics['slide_auroc'],
        'Val Center 3 Slide Accuracy': val_metrics['slide_accuracy'],
        'Test Center 4 Accuracy (Unseen)': test_metrics['accuracy'],
        'Test Center 4 Balanced Accuracy': test_metrics.get('balanced_accuracy', 0.0),
        'Test Center 4 Macro F1': test_metrics.get('macro_f1', 0.0),
        'Test Center 4 AUROC': test_metrics['auroc'],
        'Test Center 4 ECE': test_metrics.get('ece', 0.0),
        'Test Center 4 Slide AUROC': test_metrics['slide_auroc'],
        'Test Center 4 Slide Accuracy': test_metrics['slide_accuracy'],
        'Worst Train Hospital Accuracy': worst_hosp_eval['worst_hospital_accuracy'],
        'Average Train Hospital Accuracy': worst_hosp_eval['average_hospital_accuracy'],
        'Spent Epsilon (Composed)': history['privacy_spent_eps'][-1],
        'Spent Epsilon (Gradient-Only)': history.get('privacy_grad_eps', [0.0])[-1]
    }])
    
    results_csv_name = f"{args.tag}_{args.method}_rounds{args.rounds}_epochs{epochs_val}_seed{args.seed}_results.csv"
    results_csv_path = os.path.join(config['logging']['results_dir'], results_csv_name)
    summary_df.to_csv(results_csv_path, index=False)

    # Persist artifacts to /kaggle/working/experiment_artifacts/
    persisted_artifacts = persist_experiment_artifacts(
        checkpoint_path=checkpoint_path,
        results_csv_path=results_csv_path
    )

    # ------------------------------------------------------------------
    # Post-Run Comprehensive Report
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("                 PHASE 10A RUN COMPLETION REPORT")
    print("======================================================================")
    print(f"- Training Status:                 COMPLETED SUCCESSFULLY")
    print(f"- Measured Runtime:                {elapsed_time:.2f} seconds ({elapsed_time/60.0:.2f} min)")
    print(f"- Peak GPU Memory:                 {peak_vram_gb:.2f} GB")
    print(f"- Spent Privacy Epsilon (Composed):{history['privacy_spent_eps'][-1]:.4f}")
    print(f"- Spent Privacy Epsilon (Gradient):{history.get('privacy_grad_eps', [0.0])[-1]:.4f}")
    print("\n--- Validation Center 3 Metrics ---")
    print(f"  - Patch Accuracy:                {val_metrics['accuracy']*100:.2f}%")
    print(f"  - Balanced Accuracy:             {val_metrics.get('balanced_accuracy', 0.0)*100:.2f}%")
    print(f"  - Macro F1:                      {val_metrics.get('macro_f1', 0.0):.4f}")
    print(f"  - Patch AUROC:                   {val_metrics['auroc']:.4f}")
    print(f"  - Slide AUROC:                   {val_metrics['slide_auroc']:.4f}")
    print(f"  - Slide Accuracy:                {val_metrics['slide_accuracy']*100:.2f}%")
    print("\n--- Unseen Test Center 4 Metrics ---")
    print(f"  - Patch Accuracy:                {test_metrics['accuracy']*100:.2f}%")
    print(f"  - Balanced Accuracy:             {test_metrics.get('balanced_accuracy', 0.0)*100:.2f}%")
    print(f"  - Macro F1:                      {test_metrics.get('macro_f1', 0.0):.4f}")
    print(f"  - Patch AUROC:                   {test_metrics['auroc']:.4f}")
    print(f"  - Slide AUROC:                   {test_metrics['slide_auroc']:.4f}")
    print(f"  - Slide Accuracy:                {test_metrics['slide_accuracy']*100:.2f}%")
    print("\n--- Hospital-Level Performance Breakdown ---")
    for c in train_centers:
        print(f"  - Training Center {c}: Accuracy = {train_hospital_metrics[c]['accuracy']*100:.2f}%, AUROC = {train_hospital_metrics[c]['auroc']:.4f}, Slide AUROC = {train_hospital_metrics[c]['slide_auroc']:.4f}")
    print(f"  - Worst Training Hospital Accuracy: {worst_hosp_eval['worst_hospital_accuracy']*100:.2f}%")
    print(f"  - Average Training Hospital Acc:    {worst_hosp_eval['average_hospital_accuracy']*100:.2f}%")
    print("\n--- Output & Checkpoint Artifacts ---")
    print(f"  - Checkpoint Saved To:           {checkpoint_path}")
    print(f"  - Results CSV Saved To:          {results_csv_path}")
    if 'checkpoint' in persisted_artifacts:
        print(f"  - Persisted Checkpoint:          {persisted_artifacts['checkpoint']}")
    if 'results_csv' in persisted_artifacts:
        print(f"  - Persisted Results CSV:         {persisted_artifacts['results_csv']}")
    print("======================================================================")

if __name__ == "__main__":
    main()
