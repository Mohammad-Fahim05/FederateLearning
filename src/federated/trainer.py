import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from src.models.resnet_backbone import ResNet18Backbone
from src.federated.client import HospitalClient
from src.federated.server import CentralServer
from src.privacy.privacy_accountant import RDPPrivacyAccountant
from src.evaluation.metrics import compute_classification_metrics, compute_worst_hospital_metric
from src.evaluation.slide_evaluator import SlideEvaluator

class FederatedTrainer:
    """
    Coordinator for Federated Learning experiments.
    Manages clients, communication rounds, privacy accounting, server aggregation, and hospital evaluation.
    """
    def __init__(self, dataset, client_indices, config, method_name='DP-WHFedDG', device=None):
        self.dataset = dataset
        self.client_indices = client_indices
        self.config = config
        self.method_name = method_name
        
        # Auto-detect device if 'auto' or None
        if device is None:
            config_device = config.get('project', {}).get('device', 'auto')
        else:
            config_device = device
            
        if config_device == 'auto' or config_device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config_device)
            
        # Initialize Global ResNet-18 Model
        self.global_model = ResNet18Backbone(
            num_classes=config['dataset']['num_classes'],
            pretrained=False
        ).to(self.device)
        
        # Config algorithm
        config['method'] = method_name
        self.server = CentralServer(self.global_model, config)
        
        # Instantiate Hospital Clients
        self.clients = {}
        for cid, indices in client_indices.items():
            self.clients[cid] = HospitalClient(
                client_id=cid,
                dataset=dataset,
                indices=indices,
                config=config,
                device=self.device
            )
            
        # RDP Privacy Accountant
        self.privacy_accountant = RDPPrivacyAccountant(
            target_delta=config['privacy']['target_delta']
        )
        
        self.slide_evaluator = SlideEvaluator()

    def evaluate_on_subset(self, indices):
        """
        Evaluates the global model on a specific subset of data.
        Returns patch-level metrics and slide-level metrics.
        """
        self.global_model.eval()
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=self.config['federated']['batch_size'], shuffle=False)
        
        all_targets = []
        all_preds = []
        all_probs = []
        all_slide_ids = []
        
        with torch.no_grad():
            for images, labels, _, slide_ids in loader:
                images = images.to(self.device)
                outputs = self.global_model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_targets.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_slide_ids.extend(slide_ids.numpy())
                
        all_probs = np.array(all_probs)
        patch_metrics = compute_classification_metrics(all_targets, all_probs, all_preds)
        
        tumor_probs = all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]
        slide_metrics = self.slide_evaluator.evaluate_slides(tumor_probs, all_targets, all_slide_ids)
        
        metrics = {**patch_metrics, **slide_metrics}
        return metrics

    def run_training(self, rounds=None, eval_val_indices=None):
        """
        Executes T communication rounds.
        """
        import time
        if rounds is None:
            rounds = self.config['federated']['rounds']
            
        history = {
            'rounds': [],
            'client_losses': [],
            'privacy_spent_eps': [],
            'train_center_metrics': {},
            'val_center_metrics': [],
            'test_center_metrics': [],
            'round_times': []
        }
        
        print(f"--- Starting Federated Training ({self.method_name}) for {rounds} Rounds ---")
        
        for r in range(1, rounds + 1):
            r_start_time = time.time()
            client_weights = []
            client_losses = {}
            client_sample_counts = {}
            
            # Local training at each hospital client
            for cid, client in self.clients.items():
                weights, loss = client.local_train(self.global_model)
                client_weights.append(weights)
                client_losses[cid] = loss
                client_sample_counts[cid] = client.num_samples
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    c_mem = torch.cuda.memory_allocated() / (1024**2)
                    c_res = torch.cuda.memory_reserved() / (1024**2)
                    c_max = torch.cuda.max_memory_allocated() / (1024**2)
                    print(f"  [Client {cid} Complete] CUDA Mem: {c_mem:.1f} MB (Reserved: {c_res:.1f} MB, Peak: {c_max:.1f} MB)")
                
            # Central server aggregation
            agg_weights = self.server.aggregate(client_weights, client_losses, client_sample_counts)
            
            # Compute exact cumulative DP Epsilon per client
            batch_size = self.config['federated']['batch_size']
            local_epochs = self.config['federated']['local_epochs']
            noise_mult = self.config['privacy']['noise_multiplier'] if self.config['privacy']['enabled'] else 0.0
            loss_noise_std = self.config.get('privacy', {}).get('loss_noise_std', 0.05)
            
            client_dp_reports = {}
            for cid, client in self.clients.items():
                cdp = self.privacy_accountant.get_client_privacy_spent(
                    num_samples=client.num_samples,
                    batch_size=batch_size,
                    local_epochs=local_epochs,
                    rounds=r,
                    noise_multiplier=noise_mult,
                    loss_noise_std=loss_noise_std,
                    compose_loss=True
                )
                client_dp_reports[cid] = cdp

            worst_total_eps = max(cdp['total_epsilon'] for cdp in client_dp_reports.values())
            worst_grad_eps = max(cdp['grad_epsilon'] for cdp in client_dp_reports.values())
            
            # Evaluate per-round progress
            avg_loss = sum(client_losses.values()) / len(client_losses)
            r_elapsed = time.time() - r_start_time
            history['rounds'].append(r)
            history['client_losses'].append(client_losses)
            history['privacy_spent_eps'].append(worst_total_eps)
            history['round_times'].append(r_elapsed)
            if 'privacy_grad_eps' not in history:
                history['privacy_grad_eps'] = []
            history['privacy_grad_eps'].append(worst_grad_eps)
            
            # Optional per-round validation evaluation (e.g. for development runs)
            if eval_val_indices is not None:
                val_metrics = self.evaluate_on_subset(eval_val_indices)
                history['val_center_metrics'].append(val_metrics)
                acc = val_metrics.get('accuracy', 0.0)
                bacc = val_metrics.get('balanced_accuracy', 0.0)
                f1 = val_metrics.get('macro_f1', 0.0)
                auc = val_metrics.get('auroc', 0.0)
                ece = val_metrics.get('ece', 0.0)
                print(f"Round [{r:02d}/{rounds:02d}] | Train Loss: {avg_loss:.4f} | Val Acc: {acc*100:.2f}% | Val BalAcc: {bacc*100:.2f}% | Val F1: {f1:.4f} | Val AUROC: {auc:.4f} | Val ECE: {ece:.4f} | Time: {r_elapsed:.1f}s")
            elif r % 10 == 0 or r == 1 or r == rounds:
                eps_str = f"{worst_total_eps:.2f} (grad: {worst_grad_eps:.2f})" if worst_total_eps < float('inf') else "inf (No DP)"
                print(f"Round [{r}/{rounds}] - Avg Loss: {avg_loss:.4f} | Spent DP Epsilon: {eps_str} | Time: {r_elapsed:.1f}s")
                
        return history
