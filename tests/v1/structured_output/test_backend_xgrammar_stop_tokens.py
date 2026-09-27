# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #42403.

A request's stop-token set (the generation_config eos list plus any
user-supplied ``stop_token_ids``) is invisible to xgrammar, which only knows
the tokenizer's single eos. Such tokens can therefore escape the grammar
bitmask while the FSM is still mid-object and truncate structured output.
``compile_grammar`` now forwards ``all_stop_token_ids`` to the matcher as
``override_stop_tokens`` so xgrammar masks them until the grammar completes.
"""

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257

# gpt2 token ids used to drive a `{"type": "string"}` grammar deterministically.
EOS = 50256  # <|endoftext|> -- the tokenizer's only default stop token
QUOTE = 1  # standalone `"`; opens then closes the JSON string
LETTER = 55  # `X`: valid string content, not a special/stop token by default

GLM_TOKENIZER = "zai-org/GLM-4.7"
GLM_VOCAB_SIZE = 151552
# generation_config eos ids: <|endoftext|>, <|user|>, <|observation|>. xgrammar
# only knows <|endoftext|> as a stop token; the other two are plain text to it.
GLM_STOP_TOKENS = {151329, 151336, 151338}


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


def _make_backend(tokenizer: str, vocab_size: int) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=AutoTokenizer.from_pretrained(tokenizer),
        vocab_size=vocab_size,
    )


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    return _make_backend(TOKENIZER, VOCAB_SIZE)


def test_request_stop_tokens_gated_to_grammar_terminal(backend: XgrammarBackend):
    schema = '{"type": "string"}'
    default = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    override = backend.compile_grammar(
        StructuredOutputOptions.JSON, schema, stop_token_ids={EOS, LETTER}
    )

    # Open the string: both grammars are now in a non-terminal state.
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])

    bm_default = backend.allocate_token_bitmask(1)
    bm_override = backend.allocate_token_bitmask(1)
    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # Mid-string, the plain token is valid content, so the default grammar
    # leaves it samplable -- this is the leak. Registering it as a stop token
    # masks it until the grammar can terminate.
    assert _token_allowed(bm_default[0], LETTER)
    assert not _token_allowed(bm_override[0], LETTER)

    # Close the string -> accepting state (grammar complete, not yet terminated).
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])
        assert not grammar.is_terminated()

    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # The extra stop token may now terminate under the override, never under
    # the default grammar -- and the tokenizer's own eos still terminates both,
    # so default termination is preserved.
    assert not _token_allowed(bm_default[0], LETTER)
    assert _token_allowed(bm_override[0], LETTER)
    assert _token_allowed(bm_default[0], EOS)
    assert _token_allowed(bm_override[0], EOS)


def test_text_stop_tokens_masked_until_grammar_completes():
    """GLM ends turns with <|user|>/<|observation|>, which xgrammar treats as
    text. Mid-grammar the matcher still rejects them as stop tokens, so the
    mask must exclude them too; sampling one would otherwise fail the request
    with "Failed to advance FSM".
    """
    backend = _make_backend(GLM_TOKENIZER, GLM_VOCAB_SIZE)
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON,
        '{"type": "string"}',
        stop_token_ids=GLM_STOP_TOKENS,
    )
    quote = backend.tokenizer.encode('"', add_special_tokens=False)
    bitmask = backend.allocate_token_bitmask(1)

    assert grammar.accept_tokens("req", quote)
    grammar.fill_bitmask(bitmask, 0)
    for token_id in GLM_STOP_TOKENS:
        assert not grammar.validate_tokens([token_id])
        assert not _token_allowed(bitmask[0], token_id)

    assert grammar.accept_tokens("req", quote)
    grammar.fill_bitmask(bitmask, 0)
    for token_id in GLM_STOP_TOKENS:
        assert _token_allowed(bitmask[0], token_id)
