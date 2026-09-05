import torch

class DPGradientClipper:
    """
    Applies mathematically rigorous, vectorized per-sample gradient clipping and Gaussian DP noise injection.
    Ensures formal Differential Privacy guarantees for sampled Gaussian mechanism with high throughput.
    """
    def __init__(self, max_grad_norm=1.0, noise_multiplier=0.8, enabled=True):
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.enabled = enabled

    def clip_and_noise_sample_grad(self, model, criterion, images, labels, device='cpu'):
        """
        Computes vectorized per-sample gradients using torch.func / vmap,
        clips each sample's gradient norm to C = max_grad_norm:
        g_i' = g_i * min(1, C / ||g_i||_2)
        Averages across batch and injects Gaussian DP noise:
        \\bar{g} = (1/B) * \\sum_{i=1}^B g_i' + N(0, (sigma * C / B)^2 * I)
        """
        if not self.enabled:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss.item()

        batch_size = len(images)
        if batch_size == 0:
            return 0.0

        # Extract trainable parameters and buffers
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        buffers = dict(model.named_buffers())

        # Vectorized per-sample loss function
        def compute_single_loss(p_dict, b_dict, img_single, lbl_single):
            out = torch.func.functional_call(model, (p_dict, b_dict), img_single.unsqueeze(0))
            loss = criterion(out, lbl_single.unsqueeze(0))
            if loss.dim() > 0:
                loss = loss.mean()
            return loss

        # Fast vectorized per-sample gradient computation
        ft_grad = torch.func.grad(compute_single_loss)
        ft_vmap = torch.func.vmap(ft_grad, in_dims=(None, None, 0, 0))

        # Model evaluation mode during per-sample functional evaluation ensures stable buffer running stats
        was_training = model.training
        model.eval()
        try:
            sample_grads = ft_vmap(params, buffers, images, labels)
        finally:
            if was_training:
                model.train()

        # Compute per-sample gradient L2 norms across all parameters
        sample_norms_sq = torch.zeros(batch_size, device=images.device)
        for p_name, s_grad in sample_grads.items():
            sample_norms_sq += s_grad.flatten(start_dim=1).pow(2).sum(dim=1)
        sample_norms = sample_norms_sq.sqrt()

        # Scaling factor min(1, C / ||g_i||_2)
        clip_coefs = torch.clamp(self.max_grad_norm / (sample_norms + 1e-6), max=1.0)  # Shape: (B,)

        # Compute average loss for tracking
        with torch.no_grad():
            full_out = model(images)
            loss_val = criterion(full_out, labels)
            total_loss = loss_val.item() if loss_val.dim() == 0 else loss_val.mean().item()

        # Set parameter gradients: averaged clipped gradient + Gaussian noise
        model.zero_grad()
        for name, p in model.named_parameters():
            if p.requires_grad and name in sample_grads:
                sg = sample_grads[name]  # Shape: (B, *param_shape)
                view_shape = [batch_size] + [1] * (sg.dim() - 1)
                clipped_sg = sg * clip_coefs.view(*view_shape)
                avg_clipped_grad = clipped_sg.mean(dim=0)

                if self.noise_multiplier > 0:
                    std = (self.noise_multiplier * self.max_grad_norm) / batch_size
                    noise = torch.randn_like(p.data) * std
                    p.grad = avg_clipped_grad + noise
                else:
                    p.grad = avg_clipped_grad

        return total_loss
