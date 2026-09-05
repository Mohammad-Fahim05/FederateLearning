import os
import yaml
import copy
import torch
import numpy as np
import pandas as pd
from src.data.dataset import Camelyon17HospitalDataset
from src.data.client_splitter import HospitalClientSplitter
from src.federated.trainer import FederatedTrainer

def run_all_baselines(config_path='./configs/camelyon17_wilds.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Load dataset and create natural hospital client splits
    use_synthetic = config.get('dataset', {}).get('use_synthetic', False)
    dataset = Camelyon17HospitalDataset(
        root_dir=config['dataset'].get('root_dir', './data'),
        download=config['dataset'].get('download', True),
        use_synthetic=use_synthetic,
        seed=config['project']['seed']
    )
    splitter = HospitalClientSplitter(dataset, train_centers=config['dataset']['train_centers'], seed=config['project']['seed'])
    client_indices = splitter.get_natural_split()
    
    val_indices = dataset.get_center_subsets([config['dataset']['val_center']])
    test_indices = dataset.get_center_subsets([config['dataset']['test_center']])
    
    baselines = ['FedAvg', 'FedProx', 'GroupDRO', 'DP-FedAvg']
    results = []
    
    for method in baselines:
        print(f"\n==========================================")
        print(f"   RUNNING BASELINE: {method}")
        print(f"==========================================")
        
        cfg = copy_config(config)
        if method == 'DP-FedAvg':
            cfg['privacy']['enabled'] = True
            cfg['method'] = 'FedAvg'
        else:
            cfg['privacy']['enabled'] = False
            cfg['method'] = method
            
        trainer = FederatedTrainer(
            dataset=dataset,
            client_indices=client_indices,
            config=cfg,
            method_name=method,
            device=config['project']['device']
        )
        
        history = trainer.run_training(rounds=20)
        
        # Evaluate on validation center (Center 3) and unseen test center (Center 4)
        val_metrics = trainer.evaluate_on_subset(val_indices)
        test_metrics = trainer.evaluate_on_subset(test_indices)
        
        # Hospital metrics breakdown for Centers 0, 1, 2
        hospital_metrics = {}
        for c in config['dataset']['train_centers']:
            c_indices = dataset.get_center_subsets([c])
            hospital_metrics[c] = trainer.evaluate_on_subset(c_indices)
            
        train_hosp_accs = [m['accuracy'] for m in hospital_metrics.values()]
        worst_train_acc = min(train_hosp_accs)
        
        res = {
            'Method': method,
            'Privacy Enabled': cfg['privacy']['enabled'],
            'Spent Epsilon': history['privacy_spent_eps'][-1],
            'Val Center 3 Accuracy': val_metrics['accuracy'],
            'Val Center 3 AUROC': val_metrics['auroc'],
            'Test Center 4 Accuracy (Unseen Target)': test_metrics['accuracy'],
            'Test Center 4 AUROC (Unseen Target)': test_metrics['auroc'],
            'Test Center 4 Slide AUROC': test_metrics['slide_auroc'],
            'Worst Train Hospital Accuracy': worst_train_acc
        }
        results.append(res)
        print(f"Results for {method}: Test Center 4 Acc = {test_metrics['accuracy']:.4f}, Slide AUROC = {test_metrics['slide_auroc']:.4f}")

    results_df = pd.DataFrame(results)
    save_path = os.path.join(config['logging']['results_dir'], 'baseline_results.csv')
    results_df.to_csv(save_path, index=False)
    print(f"\nBaseline experiments finished! Results saved to {save_path}")
    return results_df

def copy_config(cfg):
    import copy
    return copy.deepcopy(cfg)

if __name__ == "__main__":
    run_all_baselines()
