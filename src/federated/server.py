import torch
import copy
import numpy as np

class CentralServer:
    """
    Central Coordinator for Federated Learning.
    Supports DP-WHFedDG (Noise-Bounded Smooth DRO), FedAvg, FedProx, and Group DRO aggregation.
    """
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config
        self.algorithm = config.get('method', 'DP-WHFedDG')
        self.eta = config['dro']['eta']
        self.historical_losses = {}
        
    def aggregate(self, client_weights_list, client_losses_dict, client_sample_counts):
        """
        client_weights_list: List of state_dicts from clients
        client_losses_dict: Dict of {client_id: local_loss}
        client_sample_counts: Dict of {client_id: n_samples}
        """
        client_ids = list(client_losses_dict.keys())
        num_clients = len(client_ids)
        
        # Determine aggregation weights
        if self.algorithm == 'DP-WHFedDG':
            # Noise-Bounded Smooth DRO Weighting via exponential softmax
            losses = np.array([client_losses_dict[cid] for cid in client_ids])
            # Exponential softmax for smooth risk reweighting
            exp_losses = np.exp(self.eta * (losses - np.max(losses)))  # Numerical stability
            agg_weights = exp_losses / np.sum(exp_losses)
            
        elif self.algorithm in ['FedAvg', 'FedProx']:
            # Standard sample-count proportional weighting
            total_samples = sum([client_sample_counts[cid] for cid in client_ids])
            agg_weights = np.array([client_sample_counts[cid] / total_samples for cid in client_ids])
            
        elif self.algorithm == 'GroupDRO':
            # Standard Group DRO exponentiated gradient weighting
            losses = np.array([client_losses_dict[cid] for cid in client_ids])
            agg_weights = np.exp(self.eta * losses)
            agg_weights = agg_weights / np.sum(agg_weights)
            
        else:
            # Uniform fallback
            agg_weights = np.ones(num_clients) / num_clients

        # Compute weighted average of state_dict parameters
        new_state_dict = copy.deepcopy(global_model_state_dict(self.global_model))
        first_client_dict = client_weights_list[0]
        
        for key in new_state_dict.keys():
            if first_client_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
                new_state_dict[key] = torch.zeros_like(first_client_dict[key])
                for idx, cid in enumerate(client_ids):
                    w = agg_weights[idx]
                    new_state_dict[key] += w * client_weights_list[idx][key]
            else:
                # Keep original non-float parameter (e.g. num_batches_tracked)
                new_state_dict[key] = first_client_dict[key]
                
        # Load aggregated parameters into global model
        self.global_model.load_state_dict(new_state_dict)
        return agg_weights

def global_model_state_dict(model):
    return model.state_dict()
