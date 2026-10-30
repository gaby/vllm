# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash indexer head gate: fp32 logits straight from bf16 activations.

The gate logits stay fp32 (bf16 rounding can flip near-tie pool rankings), but
they used to come from an fp32 copy of ``hidden_states`` and an fp32 matmul of
a cached fp32 weight. The fp32 epilogue of the half-precision GEMM computes the
same product without the copy. CPU torch has no such epilogue, so ``torch.mm``
is spied on and emulates it.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.config import AttentionConfig, CacheConfig
from vllm.models.glm5next.common import attention
from vllm.utils.torch_utils import set_default_torch_dtype

HIDDEN, N_HEAD, HEAD_DIM, NUM_TOKENS = 64, 4, 128, 5
TOPK_WIDTH = 256


@pytest.fixture(autouse=True)
def _parallel_state(default_vllm_config, dist_init):
    """A real single-rank TP/PP state and current config for the layers."""


def _make_indexer(
    monkeypatch,
    *,
    cuda_alike: bool = True,
    batch_invariant: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> attention.Indexer:
    monkeypatch.setattr(attention.current_platform, "is_cuda_alike", lambda: cuda_alike)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if batch_invariant else "0")
    # The top-k op needs DeepGEMM or AITER kernels and plays no part in the gate.
    monkeypatch.setattr(
        attention, "SparseAttnIndexerKpool", lambda *a, **k: nn.Module()
    )
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=128,
        vocab_size=32,
        pad_token_id=0,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=4,
        layer_types=["deepseek_sparse_attention", "linear_attention"],
        mhc=False,
        mla_use_nope=True,
        q_lora_rank=32,
        index_topk=128,
        index_kpool=4,
        index_n_heads=N_HEAD,
        index_head_dim=HEAD_DIM,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024, dtype=dtype),
        attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=5),
    )
    with set_default_torch_dtype(dtype):
        return attention.Indexer(
            vllm_config,
            config,
            HIDDEN,
            config.q_lora_rank,
            None,
            CacheConfig(),
            torch.empty(NUM_TOKENS, TOPK_WIDTH, dtype=torch.int32),
            prefix="model.layers.0.self_attn.indexer",
        )


def _load_weights(indexer: attention.Indexer) -> torch.Tensor:
    weight = indexer.wk_weights_proj.weight
    assert isinstance(weight, torch.Tensor)
    assert weight.shape == (HEAD_DIM + N_HEAD, HIDDEN)
    torch.manual_seed(0)
    weight.data.copy_(torch.randn(HEAD_DIM + N_HEAD, HIDDEN))
    return weight


def _reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate = weight[HEAD_DIM:].double()
    return (hidden.double() @ gate.t()).float()


@pytest.fixture
def mm_calls(monkeypatch):
    calls: list[SimpleNamespace] = []
    real_mm = torch.mm

    def spy_mm(a, b, *, out_dtype=None):
        calls.append(SimpleNamespace(a=a, b=b, out_dtype=out_dtype))
        return real_mm(a.float(), b.float()) if out_dtype else real_mm(a, b)

    monkeypatch.setattr(torch, "mm", spy_mm)
    return calls


def test_gate_uses_the_fp32_epilogue_without_copying_the_activations(
    monkeypatch, mm_calls
):
    indexer = _make_indexer(monkeypatch)
    weight = _load_weights(indexer)
    hidden = torch.randn(NUM_TOKENS, HIDDEN).bfloat16()

    gate = indexer._head_gate(hidden)

    (call,) = mm_calls
    assert call.a is hidden
    assert call.a.dtype == call.b.dtype == torch.bfloat16
    assert call.out_dtype == torch.float32
    assert gate.dtype == torch.float32
    torch.testing.assert_close(gate, _reference(hidden, weight))


def test_gate_only_reads_the_weights_proj_rows(monkeypatch, mm_calls):
    indexer = _make_indexer(monkeypatch)
    weight = _load_weights(indexer)
    weight.data[:HEAD_DIM] = float("nan")
    hidden = torch.randn(NUM_TOKENS, HIDDEN).bfloat16()

    gate = indexer._head_gate(hidden)

    assert gate.shape == (NUM_TOKENS, N_HEAD)
    assert torch.isfinite(gate).all()


def test_gate_follows_in_place_weight_updates(monkeypatch, mm_calls):
    # A weight refit writes the parameter in place; a cached copy goes stale.
    indexer = _make_indexer(monkeypatch)
    weight = _load_weights(indexer)
    hidden = torch.randn(NUM_TOKENS, HIDDEN).bfloat16()
    before = indexer._head_gate(hidden)

    weight.data[HEAD_DIM:].mul_(2)

    torch.testing.assert_close(indexer._head_gate(hidden), 2 * before)


@pytest.mark.parametrize(
    "build_kwargs",
    [
        # CPU, or any platform without the cuBLAS/hipBLASLt epilogue.
        {"cuda_alike": False},
        # Batch-invariant mode relies on its own deterministic mm override.
        {"batch_invariant": True},
        # The epilogue takes half-precision operands only.
        {"dtype": torch.float32},
    ],
    ids=["no-epilogue-platform", "batch-invariant", "fp32-model"],
)
def test_gate_keeps_the_fp32_matmul_where_the_epilogue_does_not_apply(
    monkeypatch, mm_calls, build_kwargs
):
    indexer = _make_indexer(monkeypatch, **build_kwargs)
    weight = _load_weights(indexer)
    hidden = torch.randn(NUM_TOKENS, HIDDEN).to(weight.dtype)

    gate = indexer._head_gate(hidden)

    (call,) = mm_calls
    assert call.a.dtype == call.b.dtype == torch.float32
    assert call.out_dtype is None
    torch.testing.assert_close(gate, _reference(hidden, weight))


def test_gate_falls_back_when_the_activations_do_not_match_the_weight(
    monkeypatch, mm_calls
):
    indexer = _make_indexer(monkeypatch)
    weight = _load_weights(indexer)
    hidden = torch.randn(NUM_TOKENS, HIDDEN)

    gate = indexer._head_gate(hidden)

    (call,) = mm_calls
    assert call.out_dtype is None
    torch.testing.assert_close(gate, _reference(hidden, weight))
