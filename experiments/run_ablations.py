import os
import yaml
import copy
import torch
import numpy as np
import pandas as pd
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.federated.trainer import FederatedTrainer
from src.utils.artifact_persister import persist_experiment_artifacts

def run_ablation_study(config_path='./configs/camelyon17_wilds.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    seeds = config.get('project', {}).get('seeds', [42, 123, 456, 789, 2026])
    
    ablation_variants = [
        ('DP-WHFedDG Full', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.05}),
        ('w/o Smooth DRO (Uniform Weighting)', {'method': 'FedAvg', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.05}),
        ('w/o Focal Loss (Standard CE)', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': False, 'sam_rho': 0.05}),
        ('w/o SAM Regularization (rho=0)', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.0}),
        ('w/o DP Noise (Non-Private)', {'method': 'DP-WHFedDG', 'privacy_enabled': False, 'focal': True, 'sam_rho': 0.05})
    ]
    
    results = []
    
    for name, params in ablation_variants:
        print(f"\n------------------------------------------")
        print(f"   ABLATION VARIANT: {name} across {len(seeds)} seeds: {seeds}")
        print(f"------------------------------------------")
        
        seed_accs = []
        seed_aurocs = []
        seed_slide_aurocs = []
        spent_epsilons = []
        
        for seed in seeds:
            cfg = copy_config(config)
            cfg['project']['seed'] = seed
            cfg['method'] = params['method']
            cfg['privacy']['enabled'] = params['privacy_enabled']
            cfg['federated']['sam_rho'] = params['sam_rho']
            if not params['focal']:
                cfg['federated']['focal_gamma'] = 0.0  # Equivalent to CE
                
            use_synthetic = cfg.get('dataset', {}).get('use_synthetic', False)
            dataset = Camelyon17HospitalDataset(
                root_dir=cfg['dataset'].get('root_dir', './data'),
                download=cfg['dataset'].get('download', True),
                use_synthetic=use_synthetic,
                seed=seed
            )
            splitter = HospitalClientSplitter(dataset, train_centers=cfg['dataset']['train_centers'], seed=seed)
            client_indices = splitter.get_natural_split()
            test_indices = dataset.get_center_subsets([cfg['dataset']['test_center']])
            
            trainer = FederatedTrainer(
                dataset=dataset,
                client_indices=client_indices,
                config=cfg,
                method_name=params['method'],
                device=cfg['project']['device']
            )
            
            rounds = cfg['federated']['rounds']
            history = trainer.run_training(rounds=rounds)
            test_metrics = trainer.evaluate_on_subset(test_indices)
            
            seed_accs.append(test_metrics['accuracy'])
            seed_aurocs.append(test_metrics['auroc'])
            seed_slide_aurocs.append(test_metrics['slide_auroc'])
            spent_epsilons.append(history['privacy_spent_eps'][-1])
            
        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        mean_auroc = np.mean(seed_aurocs)
        mean_slide_auroc = np.mean(seed_slide_aurocs)
        
        res = {
            'Ablation Variant': name,
            'Mean Test Center 4 Accuracy': mean_acc,
            'Std Test Center 4 Accuracy': std_acc,
            'Mean Test Center 4 AUROC': mean_auroc,
            'Mean Test Center 4 Slide AUROC': mean_slide_auroc,
            'Spent Epsilon': spent_epsilons[-1],
            'Seeds Evaluated': len(seeds)
        }
        results.append(res)
        print(f"Ablation '{name}': Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Slide AUROC = {mean_slide_auroc:.4f}")

    results_df = pd.DataFrame(results)
    save_path = os.path.join(config['logging']['results_dir'], 'ablation_results.csv')
    results_df.to_csv(save_path, index=False)
    persist_experiment_artifacts(results_csv_path=save_path)
    print(f"\nAblation study finished! Results saved to {save_path}")
    return results_df

def copy_config(cfg):
    import copy
    return copy.deepcopy(cfg)

if __name__ == "__main__":
    run_ablation_study()
