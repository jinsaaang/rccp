from typing import Optional
import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nan

from time import time
from tqdm import tqdm, trange
from tsl import logger

from tsl.utils import ensure_list

from reservoir_conformal_prediction.src.torch_reservoir_computing.modules import (
    RC_forecaster_torch,
)


def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        if emb.ndim == 3 and x.ndim == 4:
            emb = emb.unsqueeze(1)
        else:
            emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)


def adj_to_fc_edge_index(adjs):
    num_nodes = adjs.shape[-1]
    adjs = adjs.transpose(-2, -1)
    edge_weight = adjs.flatten()
    idx = torch.arange(num_nodes, device=adjs.device)
    edge_index = torch.cartesian_prod(idx, idx).T
    if adjs.dim() == 3:
        edge_index = [edge_index + num_nodes * i for i in range(adjs.size(0))]
        edge_index = torch.cat(edge_index, dim=-1)
    return edge_index, edge_weight


def self_normalizing_activation(x: Tensor, r: float = 1.0):
    return r * F.normalize(x, p=2, dim=-1)


def encode_dataset(
    dataset,
    encoder_class,
    encoder_kwargs,
    start_at=0,
    hidden_size=None,
    encode_exogenous=False,
    append_exogenous=False,
    encode_both=False,
    encode_residuals=False,
    keep_raw=False,
    l2_normalize=True,
    device="cpu",
    save_path=None,
):
    # if preprocess_exogenous is True, preprocess all exogenous
    if isinstance(encode_exogenous, bool) and "u" in dataset:
        preprocess_exogenous = ["u"]
    else:
        preprocess_exogenous = []
    preprocess_exogenous = ensure_list(preprocess_exogenous)

    if encode_both:
        logger.info("Encoding both target and residuals.")
        to_encode = ["target", "residuals_input"] + preprocess_exogenous
    else:
        if not encode_residuals:
            logger.info("Encoding target only.")
            to_encode = ["target"] + preprocess_exogenous
        else:
            logger.info("Encoding residuals only.")
            to_encode = ["residuals_input"] + preprocess_exogenous

    xs = []
    for key in to_encode:
        x, _ = dataset.get_tensor(key, preprocess=True)
        if x.dim() == 2:
            # x is [t, f] -> transform it to [t, n, f]
            x = x.unsqueeze(1)
            x = x.expand(-1, xs[0].shape[1], -1)
        xs.append(x)
    x = torch.cat(xs, dim=-1)

    encoded_x = torch.zeros(
        x.shape[0], x.shape[1], hidden_size, device=x.device
    )  # [n_steps, n_nodes, hidden_size]

    x = x[start_at:]
    x = torch.where(x.isnan(), torch.tensor(0.0, device=x.device), x)

    encoder = RC_forecaster_torch(**encoder_kwargs)
    encoder = encoder.to(device=device)

    start = time()
    x = x.permute(1, 0, 2)  # [n_nodes, n_steps, n_features]
    new_x = encoder._reservoir.get_states(x.to(device=device), bidir=False)
    new_x = new_x.permute(1, 0, 2)  # [n_steps, n_nodes, hidden_size]

    encoded_x[start_at:] = new_x

    if l2_normalize:
        norm = torch.linalg.norm(encoded_x, dim=-1, keepdim=True)
        encoded_x = torch.where(
            ~torch.isclose(norm, torch.zeros_like(norm)), encoded_x / norm, encoded_x
        )
        new_norm = torch.where(
            ~torch.isclose(norm, torch.zeros_like(norm)),
            torch.linalg.norm(encoded_x, dim=-1, keepdim=True),
            torch.tensor(1.0, device=encoded_x.device),
        ).squeeze()
        assert torch.isclose(
            new_norm, torch.ones_like(new_norm)
        ).all(), f"States are not normalized. Got norms {new_norm[~torch.isclose(new_norm, torch.ones_like(new_norm))]}, {norm[~torch.isclose(new_norm, torch.ones_like(new_norm))]} instead."
    elapsed = int(time() - start)

    # encoded_x = torch.zeros(n_steps, n_nodes, hidden_size, device=x.device)
    # chunk_size = 5000
    # start = time()
    # # split dataset encoding in chunks to avoid OOM
    # for ts_i in range(n_nodes):
    #     if n_steps > chunk_size:
    #         logger.info(f"Encoding dataset in chunks of {chunk_size} samples.")
    #         encoded_x[:, ts_i : ts_i + 1] = encode_by_chunks(
    #             chunk_size, x[:, ts_i : ts_i + 1], encoder, hidden_size, l2_normalize
    #         )
    #     else:
    #         logger.info("Encoding dataset in one go.")
    #         encoded_x[:, ts_i : ts_i + 1] = encoder(x[:, ts_i : ts_i + 1])

    #         if l2_normalize:
    #             norm = torch.linalg.norm(
    #                 encoded_x[:, ts_i : ts_i + 1], dim=-1, keepdim=True
    #             )
    #             encoded_x[:, ts_i : ts_i + 1] = encoded_x[:, ts_i : ts_i + 1] / norm
    #             new_norm = torch.linalg.norm(encoded_x[:, ts_i : ts_i + 1], dim=-1)
    #             assert torch.isclose(
    #                 new_norm, torch.tensor(1.0)
    #             ).all(), f"Calibration states are not normalized. Got norms {new_norm} instead."

    # elapsed = int(time() - start)

    if append_exogenous:
        logger.info("Appending exogenous variables to the encoding.")
        for key in preprocess_exogenous:
            x, _ = dataset.get_tensor(key, preprocess=True)
            if x.dim() == 2:
                x = x.unsqueeze(1)
                x = x.expand(-1, xs[0].shape[1], -1)
            encoded_x = torch.cat([encoded_x, x], dim=-1)

    if save_path is not None:
        torch.save(encoded_x, save_path)

    logger.info(f"Dataset encoded in {elapsed // 60}:{elapsed % 60:02d} minutes.")

    dataset.add_exogenous(
        "encoded_x",
        encoded_x,
        add_to_input_map=False,
        synch_mode="window",
    )

    input_map = {"x": ["encoded_x"]}
    # Control exogenous variables availability for the model
    # If encode_exogenous=True but append_exogenous=False, exogenous was used for encoding but not for model
    u = []
    if "u" in dataset and not encode_exogenous:
        # If exogenous wasn't used for encoding, keep it available for the model
        u = ["u"]
    elif "u" in dataset and encode_exogenous and append_exogenous:
        # If exogenous was both encoded and should be appended, keep it for the model
        u = ["u"]
    # If encode_exogenous=True and append_exogenous=False, exogenous is not available for model

    if keep_raw:
        u += ["residuals_input"] if encode_residuals else ["target"]
    if len(u):
        input_map["u"] = u
    dataset.set_input_map(input_map)
    return dataset


def encode_by_chunks(chunk_size, x, encoder, hidden_size, l2_normalize=True):
    n_steps, n_nodes, _ = x.shape
    encoded_x = torch.zeros(n_steps, n_nodes, hidden_size, device=x.device)
    first = True
    t = trange(0, n_steps, chunk_size, desc="Encoding dataset...", leave=True)
    for i in t:
        if first:
            encoded_chunk = encoder(x[i : i + chunk_size])
            first = False
        else:
            encoded_chunk = encoder(
                x[i : i + chunk_size],
                h0=encoded_x[i - 1].unsqueeze(0),
            )

        if l2_normalize:
            norm = torch.linalg.norm(encoded_chunk, dim=-1, keepdim=True)
            encoded_chunk = encoded_chunk / norm
            new_norm = torch.linalg.norm(encoded_chunk, dim=-1)
            assert torch.isclose(
                new_norm, torch.tensor(1.0)
            ).all(), (
                f"Calibration states are not normalized. Got norms {new_norm} instead."
            )

        encoded_x[i : i + chunk_size] = encoded_chunk

    assert encoded_x.shape == (n_steps, n_nodes, hidden_size), (
        f"Encoded shape mismatch: expected {(n_steps, n_nodes, hidden_size)}, "
        f"got {encoded_x.shape}."
    )
    return encoded_x
