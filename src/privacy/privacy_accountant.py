import numpy as np

class RDPPrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) accountant.
    Tracks privacy loss composition across local epochs and federated rounds.
    """
    def __init__(self, target_delta=1e-5):
        self.target_delta = float(target_delta)
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
    def compute_rdp(self, q, noise_multiplier, steps):
        """
        Computes RDP for sampled Gaussian mechanism.
        q: sampling ratio (batch_size / total_samples)
        noise_multiplier: sigma
        steps: total gradient steps
        """
        if noise_multiplier == 0 or noise_multiplier is None:
            return [float('inf')] * len(self.orders)
            
        rdp = []
        for alpha in self.orders:
            # Analytical RDP bound approximation for Gaussian mechanism
            r = (alpha * (q ** 2)) / (2 * (noise_multiplier ** 2)) * steps
            rdp.append(r)
        return rdp

    def get_privacy_spent(self, q, noise_multiplier, steps):
        """
        Converts RDP bounds to standard (Epsilon, Delta)-DP.
        """
        if noise_multiplier == 0 or noise_multiplier is None:
            return float('inf'), self.target_delta
            
        rdp = self.compute_rdp(q, noise_multiplier, steps)
        epsilons = []
        for alpha, r in zip(self.orders, rdp):
            eps = r + np.log(1 / self.target_delta) / (alpha - 1)
            epsilons.append(eps)
            
        best_eps = min(epsilons)
        return float(best_eps), self.target_delta
