import numpy as np

class RDPPrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) accountant for Federated Learning.
    Tracks exact privacy loss composition across local minibatch optimization steps and federated rounds
    for both per-sample clipped DP-SGD gradient releases and scalar statistic/loss releases.
    """
    def __init__(self, target_delta=1e-5):
        self.target_delta = float(target_delta)
        # Orders alpha from 1.1 to 64
        self.orders = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
    def compute_rdp_subsampled_gaussian(self, q, noise_multiplier, steps):
        """
        Computes analytical RDP for the subsampled Gaussian mechanism across `steps` minibatch releases.
        
        Args:
            q (float): Client-local subsampling ratio q = B / N_client.
            noise_multiplier (float): Noise multiplier sigma = sigma_raw / C.
            steps (int): Total number of local minibatch gradient steps = R * E * ceil(N_client / B).
            
        Returns:
            list of float: RDP epsilon at each order alpha.
        """
        if noise_multiplier is None or noise_multiplier <= 0:
            return [float('inf')] * len(self.orders)
            
        rdp = []
        for alpha in self.orders:
            # Analytical RDP bound for subsampled Gaussian mechanism
            # Under subsampling ratio q, RDP(alpha) <= (alpha * q^2) / (2 * sigma^2) * steps
            r = (alpha * (q ** 2)) / (2.0 * (noise_multiplier ** 2)) * steps
            rdp.append(r)
        return rdp

    def compute_rdp_gaussian_scalar(self, sensitivity, noise_std, num_releases):
        """
        Computes RDP for scalar Gaussian mechanism releases (e.g., noisy client loss statistics).
        
        Args:
            sensitivity (float): L2 sensitivity of the scalar release (e.g., max_loss / N_client).
            noise_std (float): Standard deviation of additive Gaussian noise.
            num_releases (int): Number of scalar releases (federated rounds R).
            
        Returns:
            list of float: RDP epsilon at each order alpha.
        """
        if noise_std is None or noise_std <= 0:
            return [float('inf')] * len(self.orders)
            
        rdp = []
        for alpha in self.orders:
            # Gaussian Mechanism RDP: RDP(alpha) = (alpha * Delta^2) / (2 * sigma^2) * num_releases
            r = (alpha * (sensitivity ** 2)) / (2.0 * (noise_std ** 2)) * num_releases
            rdp.append(r)
        return rdp

    def rdp_to_dp(self, rdp_list):
        """
        Converts RDP bounds over orders alpha to standard (Epsilon, Delta)-DP:
        epsilon(delta) = min_{alpha > 1} { RDP(alpha) + log(1/delta) / (alpha - 1) }
        """
        if any(np.isinf(r) for r in rdp_list):
            return float('inf'), self.target_delta
            
        epsilons = []
        for alpha, r in zip(self.orders, rdp_list):
            eps = r + np.log(1.0 / self.target_delta) / (alpha - 1.0)
            epsilons.append(eps)
            
        best_eps = min(epsilons)
        return float(best_eps), self.target_delta

    def get_privacy_spent(self, q, noise_multiplier, steps):
        """
        Legacy direct RDP conversion for a single subsampled Gaussian mechanism sequence.
        """
        if noise_multiplier is None or noise_multiplier <= 0:
            return float('inf'), self.target_delta
            
        rdp = self.compute_rdp_subsampled_gaussian(q, noise_multiplier, steps)
        return self.rdp_to_dp(rdp)

    def get_client_privacy_spent(self, num_samples, batch_size, local_epochs, rounds, noise_multiplier,
                                 loss_noise_std=0.05, max_loss=5.0, compose_loss=True):
        """
        Calculates exact per-client privacy expenditure taking into account:
        1. Exact client dataset size N_client
        2. Client-local sampling ratio q = B / N_client
        3. Exact minibatch optimizer steps per local epoch: S = ceil(N_client / B)
        4. Total gradient releases: T = rounds * local_epochs * S
        5. Scalar loss release privacy with additive noise (if compose_loss=True)
        
        Returns:
            dict with detailed diagnostic metrics and DP bounds.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        steps_per_epoch = int(np.ceil(num_samples / float(batch_size)))
        total_grad_steps = int(rounds * local_epochs * steps_per_epoch)
        q = min(1.0, float(batch_size) / float(num_samples))
        
        if noise_multiplier is None or noise_multiplier <= 0:
            return {
                'q': q,
                'steps_per_epoch': steps_per_epoch,
                'total_grad_steps': total_grad_steps,
                'grad_epsilon': float('inf'),
                'loss_epsilon': float('inf'),
                'total_epsilon': float('inf'),
                'delta': self.target_delta
            }

        # 1. DP-SGD Gradient RDP
        grad_rdp = self.compute_rdp_subsampled_gaussian(q, noise_multiplier, total_grad_steps)
        grad_eps, _ = self.rdp_to_dp(grad_rdp)

        # 2. Scalar Loss Release RDP (Sensitivity = max_loss / num_samples)
        loss_sensitivity = float(max_loss) / float(num_samples)
        loss_rdp = self.compute_rdp_gaussian_scalar(loss_sensitivity, loss_noise_std, num_releases=rounds)
        loss_eps, _ = self.rdp_to_dp(loss_rdp)

        # 3. Composed RDP via RDP additivity: RDP_total(alpha) = RDP_grad(alpha) + RDP_loss(alpha)
        if compose_loss and loss_noise_std is not None and loss_noise_std > 0:
            total_rdp = [g + l for g, l in zip(grad_rdp, loss_rdp)]
            total_eps, _ = self.rdp_to_dp(total_rdp)
        else:
            total_eps = grad_eps

        return {
            'q': q,
            'steps_per_epoch': steps_per_epoch,
            'total_grad_steps': total_grad_steps,
            'grad_epsilon': grad_eps,
            'loss_epsilon': loss_eps,
            'total_epsilon': total_eps,
            'delta': self.target_delta
        }
