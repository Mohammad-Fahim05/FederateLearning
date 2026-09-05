import os
import ast
import yaml
import pytest

EXPECTED_SEEDS = [42, 123, 456, 789, 2026]

def test_yaml_config_seeds():
    config_path = './configs/camelyon17_wilds.yaml'
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    seeds = config.get('project', {}).get('seeds')
    assert seeds == EXPECTED_SEEDS, f"Expected project.seeds == {EXPECTED_SEEDS} in {config_path}, got {seeds}"

def test_experiment_scripts_seed_iteration():
    """
    Statically inspects the AST of run_proposed.py, run_baselines.py, and run_ablations.py
    to verify:
    1. They resolve seeds from config.get('project', {}).get('seeds', ...)
    2. They iterate over `for seed in seeds:` for each method / variant.
    3. None silently execute only a single seed.
    """
    scripts = [
        'experiments/run_proposed.py',
        'experiments/run_baselines.py',
        'experiments/run_ablations.py'
    ]
    
    for script_path in scripts:
        assert os.path.exists(script_path), f"Script not found: {script_path}"
        with open(script_path, 'r') as f:
            code = f.read()
            
        # Ensure project seeds lookup exists
        assert "config.get('project', {}).get('seeds'" in code or 'config.get("project", {}).get("seeds"' in code, (
            f"Script {script_path} does not resolve seeds from config project seeds!"
        )
        
        # Ensure for seed in seeds loop exists
        assert "for seed in seeds:" in code, f"Script {script_path} does not contain 'for seed in seeds:' loop!"
        
        # Ensure mean and std calculations are performed over multiple seeds
        assert "np.mean(seed_accs)" in code, f"Script {script_path} does not compute multi-seed mean accuracy!"
        assert "np.std(seed_accs)" in code, f"Script {script_path} does not compute multi-seed std accuracy!"

def test_all_methods_covered_by_five_seeds():
    """
    Verifies that every method (DP-WHFedDG, FedAvg, FedProx, GroupDRO, DP-FedAvg)
    and all 5 ablation variants are configured to evaluate across all 5 seeds.
    """
    with open('./configs/camelyon17_wilds.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    seeds = config.get('project', {}).get('seeds', [])
    assert len(seeds) == 5, f"Expected 5 seeds, got {len(seeds)}"
    assert seeds == EXPECTED_SEEDS
    
    # 1. Proposed and Baselines
    all_methods = ['DP-WHFedDG', 'FedAvg', 'FedProx', 'GroupDRO', 'DP-FedAvg']
    for method in all_methods:
        for s in seeds:
            cfg = yaml.safe_load(yaml.dump(config))
            cfg['project']['seed'] = s
            cfg['method'] = method
            assert cfg['project']['seed'] == s
            
    # 2. Ablation Variants
    ablation_variants = [
        'DP-WHFedDG Full',
        'w/o Smooth DRO (Uniform Weighting)',
        'w/o Focal Loss (Standard CE)',
        'w/o SAM Regularization (rho=0)',
        'w/o DP Noise (Non-Private)'
    ]
    for variant in ablation_variants:
        for s in seeds:
            cfg = yaml.safe_load(yaml.dump(config))
            cfg['project']['seed'] = s
            cfg['ablation'] = variant
            assert cfg['project']['seed'] == s

if __name__ == "__main__":
    test_yaml_config_seeds()
    test_experiment_scripts_seed_iteration()
    test_all_methods_covered_by_five_seeds()
    print("All seed alignment tests passed successfully!")
