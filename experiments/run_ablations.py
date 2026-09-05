import os
import yaml
import copy
import torch
import numpy as np
import pandas as pd
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.federated.trainer import FederatedTrainer

def run_ablation_study(config_path='./configs/camelyon17_wilds.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    
    ablation_variants = [
        ('DP-WHFedDG Full', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.05}),
        ('w/o Smooth DRO (Uniform Weighting)', {'method': 'FedAvg', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.05}),
        ('w/o Focal Loss (Standard CE)', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': False, 'sam_rho': 0.05}),
        ('w/o SAM Regularization (rho=0)', {'method': 'DP-WHFedDG', 'privacy_enabled': True, 'focal': True, 'sam_rho': 0.0}),
        ('w/o DP Noise (Non-Private)', {'method': 'DP-WHFedDG', 'privacy_enabled': False, 'focal': True, 'sam_rho': 0.05})
    ]
    
    results = []
    use_synthetic = config.get('dataset', {}).get('use_synthetic', False)
    dataset = Camelyon17HospitalDataset(
        root_dir=config['dataset'].get('root_dir', './data'),
        download=config['dataset'].get('download', True),
        use_synthetic=use_synthetic,
        seed=config['project']['seed']
    )
    splitter = HospitalClientSplitter(dataset, train_centers=config['dataset']['train_centers'], seed=config['project']['seed'])
    client_indices = splitter.get_natural_split()
    test_indices = dataset.get_center_subsets([config['dataset']['test_center']])
    
    for name, params in ablation_variants:
        print(f"\n------------------------------------------")
        print(f"   ABLATION VARIANT: {name}")
        print(f"------------------------------------------")
        
        cfg = copy_config(config)
        cfg['method'] = params['method']
        cfg['privacy']['enabled'] = params['privacy_enabled']
        cfg['federated']['sam_rho'] = params['sam_rho']
        if not params['focal']:
            cfg['federated']['focal_gamma'] = 0.0  # Equivalent to CE
            
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
        
        res = {
            'Ablation Variant': name,
            'Target Center 4 Patch Accuracy': test_metrics['accuracy'],
            'Target Center 4 AUROC': test_metrics['auroc'],
            'Target Center 4 Slide AUROC': test_metrics['slide_auroc'],
            'Spent Epsilon': history['privacy_spent_eps'][-1]
        }
        results.append(res)
        print(f"Ablation '{name}': Accuracy = {test_metrics['accuracy']:.4f}, Slide AUROC = {test_metrics['slide_auroc']:.4f}")

    results_df = pd.DataFrame(results)
    save_path = os.path.join(config['logging']['results_dir'], 'ablation_results.csv')
    results_df.to_csv(save_path, index=False)
    print(f"\nAblation study finished! Results saved to {save_path}")
    return results_df

def copy_config(cfg):
    import copy
    return copy.deepcopy(cfg)

if __name__ == "__main__":
    run_ablation_study()
