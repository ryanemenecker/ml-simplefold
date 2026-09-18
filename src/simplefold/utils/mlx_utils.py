#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def load_mlx_state_dict(model, mlx_state_dict):
    """``model.update`` with the strictness of ``torch.nn.Module.load_state_dict(strict=True)``.

    ``mlx.nn.Module.update`` silently ignores parameters that are absent from the given tree, which
    would leave them at their random initialization. Raise instead if the checkpoint does not
    cover the model exactly.
    """
    param_keys = {k for k, _ in tree_flatten(model.parameters())}
    sd_keys = set(mlx_state_dict)
    missing = sorted(param_keys - sd_keys)
    unexpected = sorted(sd_keys - param_keys)
    if missing or unexpected:
        raise RuntimeError(
            "MLX state dict does not match the model: "
            f"{len(missing)} missing (e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]})."
        )
    model.update(tree_unflatten(list(mlx_state_dict.items())))


# We redefine the centering function here using mlx primitives
def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates.
    atom_mask : Tensor
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True

    Returns
    -------
    Tensor
        The augmented atom coordinates.

    """
    if centering:
        atom_mean = mx.sum(
            atom_coords * atom_mask[:, :, None], axis=1, keepdims=True
        ) / mx.sum(atom_mask[:, :, None], axis=1, keepdims=True)
        atom_coords = atom_coords - atom_mean

    return atom_coords


def map_torch_to_mlx(key, value):
    if 'atom2latent_proj.' in key:
        key = key.replace('atom2latent_proj.', 'atom2latent_proj.layers.')
    if 'adaLN_modulation.' in key:
        key = key.replace('adaLN_modulation.', 'adaLN_modulation.layers.')
    if 'time_embedder.mlp.' in key:
        key = key.replace('mlp.', 'mlp.layers.')
    if 'k_norm.scale' in key:
        key = key.replace('k_norm.scale', 'k_norm.weight')
    if 'q_norm.scale' in key:
        key = key.replace('q_norm.scale', 'q_norm.weight')
    if 'atom_dec_cond_proj.' in key:
        key = key.replace('atom_dec_cond_proj.', 'atom_dec_cond_proj.layers.')
    if 'atom_enc_cond_proj.' in key:
        key = key.replace('atom_enc_cond_proj.', 'atom_enc_cond_proj.layers.')
    if 'atom_feat_proj.' in key:
        key = key.replace('atom_feat_proj.', 'atom_feat_proj.layers.')
    if 'context2atom_proj.' in key:
        key = key.replace('context2atom_proj.', 'context2atom_proj.layers.')
    if 'latent2atom_proj.' in key:
        key = key.replace('latent2atom_proj.', 'latent2atom_proj.layers.')
    if 'esm_s_proj.proj.' in key:
        key = key.replace("esm_s_proj.proj.", "esm_s_proj.proj.layers.")
    if 'length_embedder.' in key:
        key = key.replace("length_embedder.", "length_embedder.layers.")
    if "embed_tokens.weight" in key:
        key = key.replace("embed_tokens.weight", "embed_tokens")
    return key, value.numpy()


def map_plddt_torch_to_mlx(key, value):
    if 'k_norm.scale' in key:
        key = key.replace('k_norm.scale', 'k_norm.weight')
    if 'q_norm.scale' in key:
        key = key.replace('q_norm.scale', 'q_norm.weight')
    if 'to_plddt_logits.' in key:
        key = key.replace('to_plddt_logits.', 'to_plddt_logits.layers.')
    return key, value.numpy()
