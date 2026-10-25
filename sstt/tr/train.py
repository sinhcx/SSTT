from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .model import NeighborhoodGNN
from ._utils import (
    _set_seed,
    _prepare_inputs,
    _directional_reconstruction_loss,
    _reconstruction_cosine,
    _neighbor_infonce,
)

EMBEDDING_KEY = "niche_embedding"


def train_model(
    adata,
    x_key="X_pca",
    base_network_key="sstt_neighbor_network",
    spatial_key="spatial",
    hidden_dim=96,
    dropout=0.2,
    lr=1e-3,
    batch_size=2048,
    num_epochs=500,
    lambda_self_fidelity=0.5,
    lambda_first_environment=0.8,
    lambda_second_environment=0.2,
    similarity_power=2.0,
    second_order_scale=0.5,
    n_pcs=None,
    early_stopping=True,
    min_training_steps=100,
    patience_steps=50,
    relative_delta=1e-3,
    l2_normalize=True,
    random_state=0,
    deterministic=True,
    device=None,
    verbose=True,
    show_loss=False,
    lambda_contrastive=1.0,
    contrastive_tau=1.0,
    k_pos=2,
    k_neg=8,
    use_similarity_weighted_contrastive=True,
):
    if num_epochs < 1 or batch_size < 1:
        raise ValueError("num_epochs and batch_size must be positive")
    weights = (
        lambda_self_fidelity,
        lambda_first_environment,
        lambda_second_environment,
        lambda_contrastive,
    )
    if (
        not np.isfinite(weights).all()
        or min(weights) < 0
        or not any(weight > 0 for weight in weights)
    ):
        raise ValueError(
            "Loss weights must be finite and nonnegative, with at least one positive weight"
        )
    scales = (contrastive_tau, similarity_power, second_order_scale)
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError(
            "Contrastive temperature and spatial scales must be finite and positive"
        )
    if k_pos < 1 or k_neg < 1:
        raise ValueError("InfoNCE requires positive k_pos and k_neg")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_seed(random_state, deterministic)
    inputs = _prepare_inputs(
        adata,
        x_key,
        base_network_key,
        spatial_key,
        n_pcs,
        second_order_scale,
        device,
    )
    x = inputs["x"]
    with torch.no_grad():
        first_environment_target = torch.sparse.mm(inputs["first_weights"], x)
        second_environment_target = torch.sparse.mm(inputs["second_weights"], x)
    model = NeighborhoodGNN(
        x.shape[1],
        hidden_dim,
        dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    steps_per_epoch = int(np.ceil(n / batch_size))

    history = {
        "epoch": [],
        "optimizer_step": [],
        "total_loss": [],
        "self_fidelity_loss": [],
        "first_environment_loss": [],
        "second_environment_loss": [],
        "reconstruction_loss": [],
        "contrast_loss": [],
    }
    best_loss = np.inf
    best_state = None
    reference_loss = None
    steps_without_improvement = 0
    optimizer_steps = 0

    iterator = range(num_epochs)
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training SSTT G3 niche GNN")

    for epoch in iterator:
        model.train()
        order = torch.randperm(n, device=device)
        total_sum = self_sum = first_sum = second_sum = 0.0
        reconstruction_sum = contrast_sum = 0.0

        for start in range(0, n, batch_size):
            nodes = order[start : start + batch_size]
            optimizer.zero_grad()

            embedding = model(
                x,
                inputs["first_weights"],
                inputs["second_weights"],
            )
            embedding_nodes = F.normalize(embedding[nodes], dim=1)

            if lambda_self_fidelity > 0:
                self_prediction = model.decode_self_state(embedding_nodes)
                self_loss = _directional_reconstruction_loss(
                    self_prediction,
                    x[nodes],
                )
            else:
                self_loss = embedding_nodes.new_zeros(())
            if lambda_first_environment > 0:
                first_prediction = model.decode_first_environment(embedding_nodes)
                first_environment_loss = _directional_reconstruction_loss(
                    first_prediction,
                    first_environment_target[nodes],
                )
            else:
                first_environment_loss = embedding_nodes.new_zeros(())
            if lambda_second_environment > 0:
                second_prediction = model.decode_second_environment(embedding_nodes)
                second_environment_loss = _directional_reconstruction_loss(
                    second_prediction,
                    second_environment_target[nodes],
                )
            else:
                second_environment_loss = embedding_nodes.new_zeros(())
            reconstruction_loss = (
                lambda_self_fidelity * self_loss
                + lambda_first_environment * first_environment_loss
                + lambda_second_environment * second_environment_loss
            )
            if lambda_contrastive > 0:
                contrast_loss = _neighbor_infonce(
                    embedding,
                    x,
                    inputs["neighbor_ptr"],
                    inputs["neighbor_edges"],
                    nodes,
                    contrastive_tau,
                    k_pos,
                    k_neg,
                    similarity_power,
                    use_similarity_weighted_contrastive,
                )
            else:
                contrast_loss = embedding.new_zeros(())
            loss = reconstruction_loss + lambda_contrastive * contrast_loss
            loss.backward()
            optimizer.step()

            size = nodes.numel()
            total_sum += loss.item() * size
            self_sum += self_loss.item() * size
            first_sum += first_environment_loss.item() * size
            second_sum += second_environment_loss.item() * size
            reconstruction_sum += reconstruction_loss.item() * size
            contrast_sum += contrast_loss.item() * size
            optimizer_steps += 1

        total_mean = total_sum / n
        self_mean = self_sum / n
        first_mean = first_sum / n
        second_mean = second_sum / n
        contrast_mean = contrast_sum / n
        history["epoch"].append(epoch + 1)
        history["optimizer_step"].append(optimizer_steps)
        history["total_loss"].append(total_mean)
        history["self_fidelity_loss"].append(self_mean)
        history["first_environment_loss"].append(first_mean)
        history["second_environment_loss"].append(second_mean)
        history["reconstruction_loss"].append(reconstruction_sum / n)
        history["contrast_loss"].append(contrast_mean)

        if verbose:
            iterator.set_postfix(
                loss=f"{total_mean:.4f}",
                self_loss=f"{self_mean:.4f}",
                env1=f"{first_mean:.4f}",
                env2=f"{second_mean:.4f}",
                contrast=f"{contrast_mean:.4f}",
            )

        if total_mean < best_loss:
            best_loss = total_mean
            best_state = copy.deepcopy(model.state_dict())

        if early_stopping:
            if reference_loss is None:
                reference_loss = total_mean
            else:
                improvement = (reference_loss - total_mean) / (
                    abs(reference_loss) + 1e-12
                )
                if improvement > relative_delta:
                    reference_loss = total_mean
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += steps_per_epoch
            if (
                optimizer_steps >= min_training_steps
                and steps_without_improvement >= patience_steps
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    for key in history:
        dtype = np.int64 if key in {"epoch", "optimizer_step"} else np.float64
        history[key] = np.asarray(history[key], dtype=dtype)
    history["best_loss"] = float(best_loss)
    history["optimizer_steps"] = int(optimizer_steps)
    diagnostic_values = {}

    model.eval()
    with torch.no_grad():
        embedding = model(
            x,
            inputs["first_weights"],
            inputs["second_weights"],
        )
    embedding = embedding.cpu().numpy()
    if l2_normalize:
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    adata.obsm[EMBEDDING_KEY] = embedding.astype(np.float32)
    with torch.no_grad():
        embedding_unit = F.normalize(
            torch.tensor(embedding, dtype=torch.float32, device=device),
            dim=1,
        )
        neighbor_embedding = torch.sparse.mm(inputs["first_weights"], embedding_unit)
        local_disagreement = 1.0 - (embedding_unit * neighbor_embedding).sum(dim=1)
        local_disagreement[
            torch.as_tensor(
                np.diff(inputs["first_weights_np"].indptr) == 0,
                device=device,
            )
        ] = torch.nan
        if lambda_self_fidelity > 0:
            self_cosine = _reconstruction_cosine(
                model.decode_self_state(embedding_unit),
                x,
            )
        else:
            self_cosine = torch.full(
                (x.shape[0],),
                torch.nan,
                dtype=x.dtype,
                device=x.device,
            )
        if lambda_first_environment > 0:
            first_environment_cosine = _reconstruction_cosine(
                model.decode_first_environment(embedding_unit),
                first_environment_target,
            )
        else:
            first_environment_cosine = torch.full(
                (x.shape[0],),
                torch.nan,
                dtype=x.dtype,
                device=device,
            )
        if lambda_second_environment > 0:
            second_environment_cosine = _reconstruction_cosine(
                model.decode_second_environment(embedding_unit),
                second_environment_target,
            )
        else:
            second_environment_cosine = torch.full(
                (x.shape[0],),
                torch.nan,
                dtype=x.dtype,
                device=x.device,
            )
    diagnostic_values["local_embedding_disagreement"] = local_disagreement.cpu().numpy()
    diagnostic_values["self_reconstruction_cosine"] = self_cosine.cpu().numpy()
    diagnostic_values["first_environment_reconstruction_cosine"] = (
        first_environment_cosine.cpu().numpy()
    )
    diagnostic_values["second_environment_reconstruction_cosine"] = (
        second_environment_cosine.cpu().numpy()
    )

    config = {
        "model": "sstt_g3_self_first_second_mean_weighted_infonce",
        "x_key": x_key,
        "base_network_key": base_network_key,
        "spatial_key": spatial_key,
        "n_pcs": n_pcs,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "lambda_self_fidelity": float(lambda_self_fidelity),
        "lambda_first_environment": float(lambda_first_environment),
        "lambda_second_environment": float(lambda_second_environment),
        "active_branches": ["self", "first_mean", "second_mean"],
        "lambda_contrastive": float(lambda_contrastive),
        "contrastive_tau": float(contrastive_tau),
        "k_pos": int(k_pos),
        "k_neg": int(k_neg),
        "use_similarity_weighted_contrastive": bool(
            use_similarity_weighted_contrastive
        ),
        "similarity_power": float(similarity_power),
        "second_order_scale": float(second_order_scale),
        "first_distance_scale": float(inputs["sigma1"]),
        "second_distance_scale": float(inputs["sigma2"]),
        "optimizer_steps": int(optimizer_steps),
        "early_stopping": bool(early_stopping),
        "min_training_steps": int(min_training_steps),
        "patience_steps": int(patience_steps),
        "relative_delta": float(relative_delta),
        "l2_normalize": bool(l2_normalize),
        "random_state": int(random_state),
        "deterministic": bool(deterministic),
        "device": str(device),
    }
    result = {
        "model": model,
        "history": history,
        "config": config,
        "diagnostics": diagnostic_values,
        "graphs": {"second_order_context_weights": inputs["second_weights_np"]},
    }
    if show_loss:
        plot_training_loss(result)
    return result


def plot_training_loss(result, save_path=None, show=True):
    history = result["history"]
    config = result["config"]
    epoch = history["epoch"]
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.plot(epoch, history["total_loss"], label="Total", linewidth=2)
    ax.plot(
        epoch,
        config["lambda_self_fidelity"] * history["self_fidelity_loss"],
        label=r"$\lambda_f$ × clean self-fidelity",
    )
    ax.plot(
        epoch,
        config["lambda_first_environment"] * history["first_environment_loss"],
        label=r"$\lambda_1$ × first environment",
    )
    ax.plot(
        epoch,
        config["lambda_second_environment"] * history["second_environment_loss"],
        label=r"$\lambda_2$ × second environment",
    )
    ax.plot(
        epoch,
        config["lambda_contrastive"] * history["contrast_loss"],
        label=r"$\lambda_c$ × neighbor InfoNCE",
    )
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax


__all__ = ["train_model", "plot_training_loss"]
