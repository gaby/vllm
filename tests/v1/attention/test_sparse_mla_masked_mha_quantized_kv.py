# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked-MHA sparse MLA prefill on quantized KV caches.

The masked path reads the new tokens' K/V from activations and only touches the
cache for cached-context chunks, which it gathers with a reader that cannot
decode the packed fp8/nvfp4 layouts. Quantized caches may therefore use it for
batches without cached context, and only when the user opts in.
"""

import pytest

import vllm.model_executor.layers.attention.sparse_mla_attention as sparse_mla
from vllm.model_executor.layers.attention.mla_attention import _use_masked_mha
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    _is_masked_mha_available,
)

GLM5_DIMS = (64, 512, 192, 64, 256)
QUANTIZED_KV_CACHES = ("fp8", "fp8_ds_mla", "nvfp4_ds_mla")


@pytest.fixture
def sm100_fa4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sparse_mla.current_platform, "is_device_capability_family", lambda family: True
    )
    monkeypatch.setattr(sparse_mla, "get_flash_attn_version", lambda **kwargs: 4)


@pytest.mark.parametrize("kv_cache_dtype", QUANTIZED_KV_CACHES)
def test_quantized_kv_masked_mha_requires_opt_in(sm100_fa4, kv_cache_dtype) -> None:
    assert not _is_masked_mha_available(*GLM5_DIMS, kv_cache_dtype)
    assert _is_masked_mha_available(*GLM5_DIMS, kv_cache_dtype, allow_quantized_kv=True)


def test_opt_in_does_not_widen_the_geometry_or_kernel_requirements(
    sm100_fa4, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _is_masked_mha_available(*GLM5_DIMS, "auto", allow_quantized_kv=True)
    wrong_heads = (128, 512, 256, 0, 256)
    assert not _is_masked_mha_available(
        *wrong_heads, "nvfp4_ds_mla", allow_quantized_kv=True
    )
    monkeypatch.setattr(sparse_mla, "get_flash_attn_version", lambda **kwargs: 3)
    assert not _is_masked_mha_available(
        *GLM5_DIMS, "nvfp4_ds_mla", allow_quantized_kv=True
    )


# TP8 FLASHMLA_SPARSE routes both of these to masked MHA for an unquantized cache.
ROUTED = dict(
    backend_name="FLASHMLA_SPARSE",
    tensor_parallel_size=8,
    qk_head_dim=256,
    v_head_dim=256,
)


@pytest.mark.parametrize(
    ("query_len", "seq_len", "has_context"),
    [(8 * 1024, 8 * 1024, False), (8 * 1024, 16 * 1024, True)],
    ids=["context_free", "cached_context"],
)
def test_quantized_kv_masked_mha_serves_only_context_free_batches(
    query_len, seq_len, has_context
) -> None:
    batch = dict(query_len=query_len, seq_len=seq_len, has_context=has_context)
    assert _use_masked_mha(**ROUTED, **batch)
    assert _use_masked_mha(**ROUTED, **batch, kv_cache_quantized=False)
    assert _use_masked_mha(**ROUTED, **batch, kv_cache_quantized=True) is (
        not has_context
    )
