#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.core as mx
from tqdm import tqdm
from einops.array_api import repeat
from utils.mlx_utils import center_random_augmentation


def logspace(start, end, steps, base=10.0, dtype=mx.float32):
    # create a linear space between start and end
    lin = mx.linspace(start, end, steps, dtype=dtype)
    # raise base to that power
    return mx.power(mx.array(base, dtype=dtype), lin)


class EMSampler:
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
            t = 1.0 - logspace(-2, 0, steps=self.num_timesteps + 1)[::-1, ...]
            t = t - mx.min(t)
            t = t / mx.max(t)
            self.steps = mx.clip(t, a_min=self.t_start, a_max=1.0)
        else:
            self.steps = mx.linspace(self.t_start, 1.0, num=self.num_timesteps + 1)

    def diffusion_coefficient(self, t, eps=0.01, w_is_zero=None):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if w_is_zero is None:
            w_is_zero = t >= self.w_cutoff
        if w_is_zero:
            w = 0.0
        return w

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
        eps = mx.random.normal(y.shape)

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
        velocity = model_fn(**model_kwargs)["predict_velocity"]

        score = flow.compute_score_from_velocity(velocity, y, t)

        diff_coeff = self.diffusion_coefficient(t, w_is_zero=w_is_zero)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + mx.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    def sample(self, model_fn, flow, noise, batch):
        sampling_timesteps = self.num_timesteps
        steps = self.steps
        y_sampled = noise
        feats = batch

        # Everything in the model that depends only on `feats` is constant across the steps of one
        # protein: compute it once (same ops, same inputs -> bitwise identical results).
        # `model_fn` is either the model or a bound `forward` (e.g. model_ema.module.forward)
        owner = getattr(model_fn, "__self__", model_fn)
        precompute_static = getattr(owner, "precompute_static", None)
        static = precompute_static(feats) if precompute_static is not None else None
        if static is not None:
            # materialize once, so every step reuses the arrays instead of re-evaluating their graphs
            mx.eval(*[v for v in static.values() if isinstance(v, mx.array)])
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
            # Bound the graph every step like mx.eval did, but let Python build step i+1 while the
            # GPU runs step i. Same graph and same RNG call order, so results are unchanged.
            mx.async_eval(y_sampled)
        mx.eval(y_sampled)

        return {"denoised_coords": y_sampled}
