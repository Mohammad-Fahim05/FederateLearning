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

def run_all_baselines(config_path='./configs/camelyon17_wilds.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    seeds = config.get('project', {}).get('seeds', [42, 123, 456, 789, 2026])
    baselines = ['FedAvg', 'FedProx', 'GroupDRO', 'DP-FedAvg']
    results = []
    
    for method in baselines:
        print(f"\n==========================================")
        print(f"   RUNNING BASELINE: {method} across {len(seeds)} seeds: {seeds}")
        print(f"==========================================")
        
        seed_accs = []
        seed_bal_accs = []
        seed_macro_f1s = []
        seed_aurocs = []
        seed_eces = []
        seed_slide_aurocs = []
        seed_slide_accs = []
        seed_worst_train_accs = []
        spent_epsilons = []
        
        for seed in seeds:
            cfg = copy_config(config)
            cfg['project']['seed'] = seed
            if method == 'DP-FedAvg':
                cfg['privacy']['enabled'] = True
                cfg['method'] = 'FedAvg'
            else:
                cfg['privacy']['enabled'] = False
                cfg['method'] = method
                
            use_synthetic = cfg.get('dataset', {}).get('use_synthetic', False)
            dataset = Camelyon17HospitalDataset(
                root_dir=cfg['dataset'].get('root_dir', './data'),
                download=cfg['dataset'].get('download', True),
                use_synthetic=use_synthetic,
                seed=seed
            )
            splitter = HospitalClientSplitter(dataset, train_centers=cfg['dataset']['train_centers'], seed=seed)
            client_indices = splitter.get_natural_split()
            
            val_indices = dataset.get_center_subsets([cfg['dataset']['val_center']])
            test_indices = dataset.get_center_subsets([cfg['dataset']['test_center']])
            
            trainer = FederatedTrainer(
                dataset=dataset,
                client_indices=client_indices,
                config=cfg,
                method_name=method,
                device=cfg['project']['device']
            )
            
            rounds = cfg['federated']['rounds']
            history = trainer.run_training(rounds=rounds)
            
            val_metrics = trainer.evaluate_on_subset(val_indices)
            test_metrics = trainer.evaluate_on_subset(test_indices)
            
            # Hospital metrics breakdown for training centers
            hospital_metrics = {}
            for c in cfg['dataset']['train_centers']:
                c_indices = dataset.get_center_subsets([c])
                hospital_metrics[c] = trainer.evaluate_on_subset(c_indices)
                
            train_hosp_accs = [m['accuracy'] for m in hospital_metrics.values()]
            worst_train_acc = min(train_hosp_accs)
            
            seed_accs.append(test_metrics['accuracy'])
            seed_bal_accs.append(test_metrics.get('balanced_accuracy', 0.0))
            seed_macro_f1s.append(test_metrics.get('macro_f1', 0.0))
            seed_aurocs.append(test_metrics['auroc'])
            seed_eces.append(test_metrics.get('ece', 0.0))
            seed_slide_aurocs.append(test_metrics['slide_auroc'])
            seed_slide_accs.append(test_metrics.get('slide_accuracy', 0.0))
            seed_worst_train_accs.append(worst_train_acc)
            spent_epsilons.append(history['privacy_spent_eps'][-1])
            
        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        mean_bal_acc = np.mean(seed_bal_accs)
        std_bal_acc = np.std(seed_bal_accs)
        mean_macro_f1 = np.mean(seed_macro_f1s)
        std_macro_f1 = np.std(seed_macro_f1s)
        mean_auroc = np.mean(seed_aurocs)
        std_auroc = np.std(seed_aurocs)
        mean_ece = np.mean(seed_eces)
        std_ece = np.std(seed_eces)
        mean_slide_auroc = np.mean(seed_slide_aurocs)
        std_slide_auroc = np.std(seed_slide_aurocs)
        mean_slide_acc = np.mean(seed_slide_accs)
        std_slide_acc = np.std(seed_slide_accs)
        mean_worst_train = np.mean(seed_worst_train_accs)
        std_worst_train = np.std(seed_worst_train_accs)
        
        res = {
            'Method': method,
            'Privacy Enabled': (method == 'DP-FedAvg'),
            'Spent Epsilon': spent_epsilons[-1],
            'Mean Test Center 4 Accuracy': mean_acc,
            'Std Test Center 4 Accuracy': std_acc,
            'Mean Test Center 4 Balanced Accuracy': mean_bal_acc,
            'Std Test Center 4 Balanced Accuracy': std_bal_acc,
            'Mean Test Center 4 Macro F1': mean_macro_f1,
            'Std Test Center 4 Macro F1': std_macro_f1,
            'Mean Test Center 4 AUROC': mean_auroc,
            'Std Test Center 4 AUROC': std_auroc,
            'Mean Test Center 4 ECE': mean_ece,
            'Std Test Center 4 ECE': std_ece,
            'Mean Test Center 4 Slide AUROC': mean_slide_auroc,
            'Std Test Center 4 Slide AUROC': std_slide_auroc,
            'Mean Test Center 4 Slide Accuracy': mean_slide_acc,
            'Std Test Center 4 Slide Accuracy': std_slide_acc,
            'Mean Worst Train Hospital Accuracy': mean_worst_train,
            'Std Worst Train Hospital Accuracy': std_worst_train,
            'Seeds Evaluated': len(seeds)
        }
        results.append(res)
        print(f"Results for {method}: Test Center 4 Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Slide AUROC = {mean_slide_auroc:.4f}")

    results_df = pd.DataFrame(results)
    save_path = os.path.join(config['logging']['results_dir'], 'baseline_results.csv')
    results_df.to_csv(save_path, index=False)
    persist_experiment_artifacts(results_csv_path=save_path)
    print(f"\nBaseline experiments finished! Results saved to {save_path}")
    return results_df

def copy_config(cfg):
    import copy
    return copy.deepcopy(cfg)

if __name__ == "__main__":
    run_all_baselines()
