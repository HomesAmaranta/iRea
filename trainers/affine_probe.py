"""Train-only affine regression and exact full-catalogue cosine retrieval."""

import math

import torch
from torch.nn import functional as F


DEFAULT_LR = 1e-3
DEFAULT_MAX_EPOCHS = 3000
DEFAULT_PATIENCE = 50
PROTOCOL = "raw_hidden_affine_cosine_plus_mse_v1"
RECALL_KS = (10, 20, 50, 100)


def cosine_mse(prediction, target, loss_type="cosine_mse"):
    if loss_type not in ("cosine", "cosine_mse"):
        raise ValueError("Unknown probe loss")
    mse = F.mse_loss(prediction, target)
    cosine_loss = 1 - F.cosine_similarity(prediction, target, dim=-1).mean()
    total = cosine_loss if loss_type == "cosine" else mse + cosine_loss
    return total, mse, cosine_loss


def validate_fit_config(learning_rate, max_epochs, patience):
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0
            or isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0):
        raise ValueError("Probe learning rate and integer training budgets must be positive")


@torch.enable_grad()
def fit_probe(hidden, target, splits, *, learning_rate=DEFAULT_LR,
              max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE,
              seed=42, log_every=0, loss_type="cosine_mse"):
    """Only train has gradients; valid selects weights; test is evaluated afterwards."""
    validate_fit_config(learning_rate, max_epochs, patience)
    x, y = hidden.detach().float(), target.detach().to(hidden.device).float()
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y)
            or x.shape[1] == 0 or y.shape[1] == 0
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
        raise ValueError("Expected finite aligned feature matrices")
    rows = {s: torch.as_tensor(splits[s], device=x.device, dtype=torch.long)
            for s in ("train", "valid", "test")}
    all_rows = torch.cat(list(rows.values()))
    if (any(len(v) == 0 for v in rows.values()) or len(all_rows) != len(x)
            or len(all_rows.unique()) != len(x) or all_rows.min() < 0 or all_rows.max() >= len(x)):
        raise ValueError("Splits must be nonempty, disjoint and cover the catalogue")
    # Initialize on CPU without consuming either the training CPU or CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        linear = torch.nn.Linear(x.shape[1], y.shape[1], device="cpu", dtype=torch.float32)
    linear = linear.to(x.device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate, weight_decay=0)
    train_x, train_y = x[rows["train"]], y[rows["train"]]
    valid_x, valid_y = x[rows["valid"]], y[rows["valid"]]
    best_loss, wait, history = math.inf, 0, []
    stop_reason = "max_epochs"
    with torch.autocast(device_type=x.device.type, enabled=False):
        for epoch in range(1, max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, mse, cosine_loss = cosine_mse(linear(train_x), train_y, loss_type)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite train loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                valid_loss, valid_mse, valid_cosine_loss = cosine_mse(
                    linear(valid_x), valid_y, loss_type
                )
            value = valid_loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite valid loss at epoch {epoch}")
            history.append({
                "epoch": epoch, "train_loss": loss.item(), "train_mse": mse.item(),
                "train_cosine_loss": cosine_loss.item(), "valid_loss": value,
                "valid_mse": valid_mse.item(), "valid_cosine_loss": valid_cosine_loss.item(),
            })
            if value < best_loss:
                best_loss, best_epoch, wait = value, epoch, 0
                best_state = {k: v.detach().clone() for k, v in linear.state_dict().items()}
            else:
                wait += 1
            if log_every and (epoch == 1 or epoch % log_every == 0):
                print(f"probe epoch={epoch} train={loss.item():.8f} valid={value:.8f} "
                      f"best_epoch={best_epoch} wait={wait}", flush=True)
            if wait >= patience:
                stop_reason = "patience"
                break
        linear.load_state_dict(best_state)
        metrics = {}
        with torch.no_grad():
            for split, indices in rows.items():
                prediction = linear(x[indices])
                total, mse, _ = cosine_mse(prediction, y[indices], loss_type)
                cosine = F.cosine_similarity(prediction, y[indices], dim=-1).mean()
                metrics[split] = {"loss": total.item(), "mse": mse.item(), "cosine": cosine.item()}
                if not all(math.isfinite(v) for v in metrics[split].values()):
                    raise RuntimeError(f"Nonfinite {split} probe metrics")
    return best_state, {
        "protocol": "raw_hidden_affine_cosine_only_v1" if loss_type == "cosine" else PROTOCOL,
        "loss": ("mean(1-cosine)" if loss_type == "cosine" else
                 "mean(1-cosine) + mean squared error over all items and coordinates"),
        "loss_weights": {"cosine": 1, "mse": int(loss_type != "cosine")}, "weight_decay": 0,
        "input_standardization": False, "target_normalization_for_loss": False,
        "optimizer": "Adam", "learning_rate": learning_rate, "batch": "full train split",
        "max_epochs": max_epochs, "patience": patience, "seed": seed,
        "selection": ("minimum valid cosine loss; strict improvement; restore best, no refit"
                      if loss_type == "cosine" else
                      "minimum valid cosine+MSE; strict improvement; restore best, no refit"),
        "epoch_numbering": "1-based; train loss before update, valid loss after update",
        "best_epoch": best_epoch, "best_valid_loss": best_loss,
        "epochs_trained": epoch, "stop_reason": stop_reason, "history": history,
        "splits": metrics,
    }


@torch.no_grad()
def retrieval_metrics(prediction, catalogue, query_rows, batch_size=128,
                      top_ks=(1, 5, 10), item_ids=None):
    """Exact cosine ranks; ties use ascending ItemID, or catalogue row if omitted."""
    if (prediction.ndim != 2 or catalogue.ndim != 2
            or prediction.shape[1] != catalogue.shape[1] or catalogue.shape[1] == 0
            or len(prediction) == 0 or len(catalogue) == 0 or batch_size <= 0
            or not top_ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0
                                 for k in top_ks)):
        raise ValueError("Invalid retrieval shapes, batch size or k values")
    catalogue = F.normalize(catalogue.detach().double(), dim=1)
    queries = F.normalize(prediction.detach().double(), dim=1)
    query_rows = torch.as_tensor(query_rows, device=catalogue.device, dtype=torch.long)
    if (query_rows.ndim != 1 or len(queries) != len(query_rows)
            or (query_rows < 0).any() or (query_rows >= len(catalogue)).any()):
        raise ValueError("Each query must have one valid matching catalogue row")
    if not torch.isfinite(queries).all() or not torch.isfinite(catalogue).all():
        raise ValueError("Retrieval vectors must be finite")
    queries = queries.to(catalogue.device)
    candidates = (torch.arange(len(catalogue), device=catalogue.device) if item_ids is None
                  else torch.as_tensor(item_ids, device=catalogue.device))
    if (candidates.ndim != 1 or len(candidates) != len(catalogue)
            or candidates.unique().numel() != len(catalogue)):
        raise ValueError("Catalogue ItemIDs must be unique and aligned with rows")
    ranks = []
    for start in range(0, len(queries), batch_size):
        scores = queries[start:start + batch_size] @ catalogue.T
        truth = query_rows[start:start + len(scores)]
        positive = scores[torch.arange(len(scores), device=scores.device), truth, None]
        better = scores > positive
        tied_before = (scores == positive) & (candidates[None, :] < candidates[truth, None])
        ranks.append(1 + (better | tied_before).sum(dim=1))
    ranks = torch.cat(ranks)
    return {
        f"Recall@{k}": {"hits": int((ranks <= k).sum()),
                       "value": float((ranks <= k).double().mean())}
        for k in top_ks
    }, ranks
