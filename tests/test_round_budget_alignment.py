import os
import ast
import yaml
import pytest

def test_yaml_config_rounds():
    config_path = './configs/camelyon17_wilds.yaml'
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rounds = config.get('federated', {}).get('rounds')
    assert rounds == 100, f"Expected federated.rounds == 100 in {config_path}, got {rounds}"

def test_experiment_scripts_static_rounds():
    """
    Statically inspects the AST of run_proposed.py, run_baselines.py, and run_ablations.py
    to verify none have hardcoded rounds (e.g. rounds=20) and all dynamically use cfg['federated']['rounds'].
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
            
        tree = ast.parse(code)
        
        # Ensure no Call to run_training has a Constant number (like 20) hardcoded
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # check if function attribute or name is run_training
                is_run_training = False
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'run_training':
                    is_run_training = True
                elif isinstance(node.func, ast.Name) and node.func.id == 'run_training':
                    is_run_training = True
                    
                if is_run_training:
                    # Check keyword arguments
                    for kw in node.keywords:
                        if kw.arg == 'rounds':
                            assert not isinstance(kw.value, ast.Constant), (
                                f"Found hardcoded literal rounds={kw.value.value} in {script_path}! "
                                "Must dynamically use cfg['federated']['rounds']."
                            )
                            
        # Ensure 'rounds = cfg['federated']['rounds']' is present in code
        assert "cfg['federated']['rounds']" in code or 'cfg["federated"]["rounds"]' in code, (
            f"Script {script_path} does not resolve rounds from cfg['federated']['rounds']!"
        )

def test_all_methods_and_ablations_resolve_100_rounds():
    """
    Verifies that all 5 benchmark methods and 5 ablation variants resolve to 100 rounds from config.
    """
    with open('./configs/camelyon17_wilds.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    expected_rounds = config['federated']['rounds']
    assert expected_rounds == 100
    
    # 1. Benchmark methods
    methods = ['DP-WHFedDG', 'FedAvg', 'FedProx', 'GroupDRO', 'DP-FedAvg']
    for method in methods:
        cfg = yaml.safe_load(yaml.dump(config))
        cfg['method'] = method
        assert cfg['federated']['rounds'] == 100, f"Method {method} resolved to {cfg['federated']['rounds']}"
        
    # 2. Ablation variants
    ablation_variants = [
        'DP-WHFedDG Full',
        'w/o Smooth DRO (Uniform Weighting)',
        'w/o Focal Loss (Standard CE)',
        'w/o SAM Regularization (rho=0)',
        'w/o DP Noise (Non-Private)'
    ]
    for variant in ablation_variants:
        cfg = yaml.safe_load(yaml.dump(config))
        assert cfg['federated']['rounds'] == 100, f"Ablation {variant} resolved to {cfg['federated']['rounds']}"

if __name__ == "__main__":
    test_yaml_config_rounds()
    test_experiment_scripts_static_rounds()
    test_all_methods_and_ablations_resolve_100_rounds()
    print("All round budget alignment tests passed!")
