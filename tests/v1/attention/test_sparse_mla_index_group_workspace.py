# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index-group workspace allocation of the sparse MLA impls.

Every index-producing sparse MLA layer registers a group that owns a
``(max_decode_rows + 1, topk_width)`` int32 workspace for converted top-k
indices. A backend whose ``forward_mqa`` converts the indices itself never reads
it, and a layer without a shared builder gets a private one sized by
``max_num_batched_tokens``; GLM-5.3-Flash paid that per layer (816 MiB at 8192
tokens, 1.6 GiB at 16384 for its 12 sparse MLA layers).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.attention import HiSparseConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Impl,
)
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

NUM_TOKENS, TOPK_WIDTH, DECODE_ROWS, NUM_LAYERS = 64, 256, 8, 3
NUM_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, V_HEAD_DIM = 16, 512, 256, 256


@pytest.fixture(autouse=True)
def _parallel_state(default_vllm_config, dist_init):
    """A real single-rank TP/PP state and current config for the impls."""


def _make_impl(impl_cls, buffer: torch.Tensor, *, has_indexer: bool = True, **kwargs):
    kv_b_proj = ColumnParallelLinear(
        KV_LORA_RANK,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        bias=False,
        prefix="kv_b_proj",
    )
    return impl_cls(
        num_heads=NUM_HEADS,
        head_size=KV_LORA_RANK,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=1536,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=0,
        qk_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
        # The impl only reads the buffer from its indexer.
        indexer=SimpleNamespace(topk_indices_buffer=buffer) if has_indexer else None,
        topk_indices_buffer=buffer,
        **kwargs,
    )


def _workspace_bytes(impls) -> int:
    return sum(
        t.numel() * t.element_size()
        for impl in impls
        if impl.index_group is not None
        for t in (
            impl.index_group.physical_topk_indices,
            impl.index_group.valid_topk_counts,
        )
    )


def _buffer() -> torch.Tensor:
    return torch.zeros(NUM_TOKENS, TOPK_WIDTH, dtype=torch.int32)


def test_impl_that_converts_inline_allocates_no_workspace():
    impls = [
        _make_impl(FlashInferMLASparseSM90Impl, _buffer()) for _ in range(NUM_LAYERS)
    ]

    assert all(impl.index_group is None for impl in impls)


def test_impl_that_reads_the_group_without_a_builder_pays_for_full_batches():
    # The fallback builder has no decode-row bound: this is what a model that
    # passes no shared builder gets per layer.
    impls = [_make_impl(FlashInferMLASparseImpl, _buffer()) for _ in range(NUM_LAYERS)]

    group = impls[0].index_group
    assert group is not None
    assert group.physical_topk_indices.shape == (NUM_TOKENS + 1, TOPK_WIDTH)
    assert _workspace_bytes(impls) == NUM_LAYERS * (
        (NUM_TOKENS + 1) * TOPK_WIDTH * 4 + (NUM_TOKENS + 1) * 4
    )


def test_shared_builder_bounds_the_workspace_by_decode_rows():
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)
    impls = [
        _make_impl(FlashInferMLASparseImpl, buffer, index_group_builder=builder)
        for _ in range(NUM_LAYERS)
    ]

    for impl in impls:
        assert impl.index_group is not None
        assert impl.index_group.physical_topk_indices.shape == (
            DECODE_ROWS + 1,
            TOPK_WIDTH,
        )
    assert _workspace_bytes(impls) == NUM_LAYERS * (
        (DECODE_ROWS + 1) * TOPK_WIDTH * 4 + (DECODE_ROWS + 1) * 4
    )


def test_opted_out_impl_ignores_an_explicit_builder():
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    impl = _make_impl(FlashInferMLASparseSM90Impl, buffer, index_group_builder=builder)

    assert impl.index_group is None
    assert builder.current_group is None


def test_opted_out_producer_ends_the_group_of_later_skip_layers():
    # The opted-out layer's indexer rewrites the shared top-k buffer, so a later
    # skip layer must not reuse the conversion of the first layer's indices.
    buffer = _buffer()
    builder = SparseMLAIndexGroupBuilder(buffer, DECODE_ROWS)

    producer = _make_impl(FlashInferMLASparseImpl, buffer, index_group_builder=builder)
    _make_impl(FlashInferMLASparseSM90Impl, buffer, index_group_builder=builder)
    skip = _make_impl(
        FlashInferMLASparseImpl,
        buffer,
        has_indexer=False,
        index_group_builder=builder,
    )

    assert skip.index_group is not None
    assert skip.index_group is not producer.index_group
    assert not skip.index_group.has_indexer


class _RecordingBuilder:
    """Stands in for the builder: a real HiSparse group needs a GPU runtime."""

    def __init__(self) -> None:
        self.registered: list[bool] = []

    def register_layer(self, is_index_producing_layer, *args, **kwargs):
        self.registered.append(is_index_producing_layer)
        return object(), 0


def test_hisparse_keeps_the_group_of_an_opted_out_impl(
    monkeypatch, default_vllm_config
):
    # HiSparse binds each layer's cache handle from its group, so a configured
    # HiSparse must not lose it. Platform validation of the field is bypassed.
    monkeypatch.setattr(
        default_vllm_config.attention_config, "hisparse_config", HiSparseConfig()
    )
    builder = _RecordingBuilder()

    _make_impl(FlashInferMLASparseSM90Impl, _buffer(), index_group_builder=builder)

    assert builder.registered == [True]
