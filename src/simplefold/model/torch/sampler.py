#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from tqdm import tqdm
from einops import repeat
from utils.boltz_utils import center_random_augmentation


class EMSampler():
    """
    A Euler-Maruyama solver for SDEs.
    """
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    def diffusion_coefficient(self, t, eps=0.01, w_is_zero=None):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if w_is_zero is None:
            w_is_zero = t >= self.w_cutoff
        if w_is_zero:
            w = 0.0
        return w

    @torch.no_grad()
    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y, 
        t, 
        t_next, 
        batch, 
        static=None,
        w_is_zero=None,
    ):
        dt = t_next - t
        # draw on the CPU generator so a given seed yields the same noise on cpu/mps/cuda
        eps = torch.randn(y.shape, dtype=y.dtype).to(y.device)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0])
        model_kwargs = dict(noised_pos=y, t=batched_t, feats=batch)
        if static is not None:
            model_kwargs["static"] = static
        velocity = model_fn(**model_kwargs)['predict_velocity']
        score = flow.compute_score_from_velocity(velocity, y, t)

        diff_coeff = self.diffusion_coefficient(t, w_is_zero=w_is_zero)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch):
        sampling_timesteps = self.num_timesteps
        steps = self.steps.to(noise.device)
        y_sampled = noise
        feats = batch

        # Everything in the model that depends only on `feats` is constant across the steps of one
        # protein: compute it once (same ops, same inputs -> bitwise identical results).
        # `model_fn` is either the model or a bound `forward` (e.g. model_ema.module.forward)
        owner = getattr(model_fn, "__self__", model_fn)
        precompute_static = getattr(owner, "precompute_static", None)
        static = precompute_static(feats) if precompute_static is not None else None
        # Evaluate the w_cutoff branch for all steps at once (one host sync instead of one per step).
        w_is_zero = (steps >= self.w_cutoff).tolist()

        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn,
                flow,
                y_sampled,
                t,
                t_next,
                feats,
                static=static,
                w_is_zero=w_is_zero[i],
            )

        return {
            "denoised_coords": y_sampled
        }
