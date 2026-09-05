import torch

class DPGradientClipper:
    """
    Applies mathematically rigorous per-sample gradient clipping and Gaussian DP noise injection.
    Ensures formal Differential Privacy guarantees for sampled Gaussian mechanism.
    """
    def __init__(self, max_grad_norm=1.0, noise_multiplier=0.8, enabled=True):
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.enabled = enabled

    def clip_and_noise_sample_grad(self, model, criterion, images, labels, device='cpu'):
        """
        Computes per-sample gradients, clips each sample's gradient to max_grad_norm C:
        g_i' = g_i * min(1, C / ||g_i||_2)
        Then averages clipped gradients across batch and adds Gaussian DP noise:
        \\bar{g} = (1/B) * \\sum_{i=1}^B g_i' + N(0, (sigma * C / B)^2 * I)
        """
        if not self.enabled:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss.item()

        batch_size = len(images)
        params = [p for p in model.parameters() if p.requires_grad]
        clipped_grads = {p: torch.zeros_like(p.data) for p in params}
        total_loss = 0.0

        # Per-sample forward, loss, backward, and clipping loop
        for i in range(batch_size):
            model.zero_grad()
            img_i = images[i:i+1]
            lbl_i = labels[i:i+1]

            out_i = model(img_i)
            loss_i = criterion(out_i, lbl_i)
            if loss_i.dim() > 0:
                loss_i = loss_i.mean()

            total_loss += loss_i.item()
            loss_i.backward()

            # Compute L2 norm of sample i gradient vector across all trainable parameters
            sample_norm_sq = 0.0
            for p in params:
                if p.grad is not None:
                    sample_norm_sq += p.grad.data.norm(2).item() ** 2
            sample_norm = sample_norm_sq ** 0.5

            # Scaling factor min(1, C / ||g_i||_2)
            clip_coef = min(1.0, self.max_grad_norm / (sample_norm + 1e-6))

            # Accumulate clipped gradient
            for p in params:
                if p.grad is not None:
                    clipped_grads[p] += p.grad.data * clip_coef

        # Set model parameter gradients to averaged clipped gradient + Gaussian DP noise
        model.zero_grad()
        for p in params:
            avg_clipped_grad = clipped_grads[p] / batch_size
            if self.noise_multiplier > 0:
                std = (self.noise_multiplier * self.max_grad_norm) / batch_size
                noise = torch.randn_like(p.data) * std
                p.grad = avg_clipped_grad + noise
            else:
                p.grad = avg_clipped_grad

        return total_loss / max(1, batch_size)


