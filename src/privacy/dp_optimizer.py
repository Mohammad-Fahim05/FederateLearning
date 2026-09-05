import torch

class DPGradientClipper:
    """
    Applies mathematically rigorous, chunked vectorized per-sample gradient clipping and Gaussian DP noise injection.
    Computes per-sample gradients in memory-safe micro-chunks via torch.func.vmap to eliminate GPU OOM spikes
    while maintaining exact mathematical equivalence.
    """
    def __init__(self, max_grad_norm=1.0, noise_multiplier=0.8, enabled=True, chunk_size=64):
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.enabled = enabled
        self.chunk_size = int(chunk_size) if chunk_size is not None else 64

    def clip_and_noise_sample_grad(self, model, criterion, images, labels, device='cpu'):
        """
        Computes vectorized per-sample gradients in micro-chunks of size chunk_size using torch.func / vmap,
        clips each sample's gradient norm to C = max_grad_norm:
        g_i' = g_i * min(1, C / ||g_i||_2)
        Averages across full batch B and injects Gaussian DP noise:
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

        # Fast vectorized per-sample gradient computation operator
        ft_grad = torch.func.grad(compute_single_loss)
        ft_vmap = torch.func.vmap(ft_grad, in_dims=(None, None, 0, 0))

        # Accumulator for clipped gradients: sum_{i=1}^B g_i'
        accum_clipped_grads = {name: torch.zeros_like(p.data) for name, p in params.items()}
        total_loss_accum = 0.0

        # Model evaluation mode during per-sample functional evaluation ensures stable buffer running stats
        was_training = model.training
        model.eval()
        try:
            chunk_size = max(1, self.chunk_size)
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunk_imgs = images[start_idx:end_idx]
                chunk_lbls = labels[start_idx:end_idx]
                curr_chunk_size = end_idx - start_idx

                # 1. Compute per-sample gradients for this chunk
                chunk_sample_grads = ft_vmap(params, buffers, chunk_imgs, chunk_lbls)

                # All clipping, norm calculation, and accumulation strictly detached in no_grad
                with torch.no_grad():
                    # 2. Compute per-sample gradient L2 norms for the chunk
                    chunk_norms_sq = torch.zeros(curr_chunk_size, device=images.device)
                    for p_name, s_grad in chunk_sample_grads.items():
                        chunk_norms_sq += s_grad.detach().flatten(start_dim=1).pow(2).sum(dim=1)
                    chunk_norms = chunk_norms_sq.sqrt()

                    # 3. Per-sample scaling factor min(1, C / ||g_i||_2)
                    chunk_clip_coefs = torch.clamp(self.max_grad_norm / (chunk_norms + 1e-6), max=1.0)

                    # 4. Accumulate clipped gradients (strictly detached)
                    for name, s_grad in chunk_sample_grads.items():
                        view_shape = [curr_chunk_size] + [1] * (s_grad.dim() - 1)
                        clipped_chunk_sg = s_grad.detach() * chunk_clip_coefs.view(*view_shape)
                        accum_clipped_grads[name] += clipped_chunk_sg.sum(dim=0)

                    # 5. Track loss
                    chunk_out = model(chunk_imgs)
                    c_loss = criterion(chunk_out, chunk_lbls)
                    c_val = c_loss.item() if c_loss.dim() == 0 else c_loss.mean().item()
                    total_loss_accum += c_val * curr_chunk_size

                # Explicitly clean up chunk memory immediately
                del chunk_sample_grads, chunk_norms_sq, chunk_norms, chunk_clip_coefs
        finally:
            if was_training:
                model.train()

        total_loss = total_loss_accum / batch_size

        # Set parameter gradients: averaged clipped gradient (sum / B) + Gaussian noise
        # Strictly detached with requires_grad=False and grad_fn=None to prevent autograd graph retention
        model.zero_grad()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in accum_clipped_grads:
                    avg_clipped_grad = accum_clipped_grads[name] / batch_size
                    if self.noise_multiplier > 0:
                        std = (self.noise_multiplier * self.max_grad_norm) / batch_size
                        noise = torch.randn_like(p.data) * std
                        p.grad = (avg_clipped_grad + noise).detach()
                    else:
                        p.grad = avg_clipped_grad.detach()

        return total_loss
