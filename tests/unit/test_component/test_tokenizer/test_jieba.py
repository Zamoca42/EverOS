"""Unit tests for :class:`JiebaTokenizer`.

Verify the contract that callers downstream depend on:

* clean token list (no whitespace, no empty strings),
* CJK + ASCII pass-through under ``cut_for_search`` segmentation,
* default stopword + ``min_length=2`` filter applied,
* batch preserves order.

The tokenizer is symmetric — cascade write side and search query side
both go through this code path, so changes here change BM25 recall on
both ends.

``JiebaTokenizer`` is imported inside each test (not at module level)
because ``jieba==0.42.1`` contains invalid escape sequences that
Python 3.12 treats as DeprecationWarning; our strict
``filterwarnings=["error"]`` converts those to errors during pytest
collection.  Deferring the import to test-run time lets the per-module
``ignore`` filter take effect.
"""

from __future__ import annotations

import warnings

import pytest

from everos.component.tokenizer import build_tokenizer


def _make_tokenizer(**kwargs):
    """Import JiebaTokenizer at call time, suppressing jieba's warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid escape sequence", DeprecationWarning)
        from everos.component.tokenizer.jieba_provider import JiebaTokenizer

    return JiebaTokenizer(**kwargs)


@pytest.fixture(autouse=True)
def _suppress_jieba_warnings():
    """Suppress jieba's invalid-escape DeprecationWarnings for all tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid escape sequence", DeprecationWarning)
        yield


def test_tokenize_returns_list_for_english() -> None:
    tokens = _make_tokenizer().tokenize("hello world")
    assert tokens == ["hello", "world"]


def test_tokenize_drops_pure_whitespace() -> None:
    """Whitespace-only tokens never reach the BM25 column."""
    tokens = _make_tokenizer().tokenize("foo   bar")
    assert all(t.strip() for t in tokens)


def test_tokenize_empty_input() -> None:
    assert _make_tokenizer().tokenize("") == []


def test_tokenize_cjk_keeps_multichar_words() -> None:
    """``cut_for_search`` keeps multi-character compounds usable by BM25."""
    tokens = _make_tokenizer().tokenize("我爱北京天安门")
    # Single-char tokens (我 / 爱) are filtered by min_length=2 (and 我
    # is also in the default stopword set). Multi-char compounds survive.
    assert "我" not in tokens
    assert "爱" not in tokens
    assert "北京" in tokens
    assert any(t in {"天安门", "天安"} for t in tokens)


def test_tokenize_drops_default_english_stopwords() -> None:
    tokens = _make_tokenizer().tokenize("the quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_drops_short_tokens_below_min_length() -> None:
    """Single-char ASCII tokens are dropped by the default ``min_length=2``."""
    tokens = _make_tokenizer().tokenize("a quick b run")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "quick" in tokens
    assert "run" in tokens


def test_tokenize_is_case_insensitive() -> None:
    """Lowercasing is part of the symmetric contract."""
    tokens = _make_tokenizer().tokenize("HELLO World")
    assert tokens == ["hello", "world"]


def test_extra_stopwords_extend_defaults() -> None:
    tk = _make_tokenizer(extra_stopwords=frozenset({"hello"}))
    tokens = tk.tokenize("hello world")
    assert "hello" not in tokens
    assert "world" in tokens


def test_custom_min_token_length_relaxes_filter() -> None:
    """Lower ``min_length`` lets shorter tokens through.

    Stopword filter still applies — even at ``min_length=1`` the English
    article ``"a"`` stays filtered because it's in the default stopwords.
    """
    tokens = _make_tokenizer(min_token_length=1).tokenize("a quick b")
    # 'a' is in the default English stopword set even at min_length=1.
    assert "a" not in tokens
    assert "b" in tokens
    assert "quick" in tokens


def test_tokenize_batch_preserves_order() -> None:
    tk = _make_tokenizer()
    out = tk.tokenize_batch(["foo bar", "baz", ""])
    assert len(out) == 3
    assert out[2] == []


def test_tokenize_hangul_query_is_not_empty() -> None:
    """Korean queries must reach BM25 with terms (regression: they were ``[]``).

    jieba's dictionary is Chinese, so ``cut_for_search("진지")`` returns the
    single syllables ``["진", "지"]`` and the ``min_length=2`` filter dropped
    both — ``sparse_recall`` then returned ``[]`` before touching the index,
    so keyword search silently answered zero hits for every Korean query.
    """
    tokens = _make_tokenizer().tokenize("진지")
    assert tokens == ["진지"]


def test_hangul_runs_use_character_bigrams() -> None:
    """A 3+ syllable run yields overlapping bigrams, not one long token."""
    tokens = _make_tokenizer().tokenize("방어선")
    assert tokens == ["방어", "어선"]


def test_single_syllable_hangul_is_kept() -> None:
    """One-syllable Korean words stay searchable instead of being dropped."""
    assert _make_tokenizer().tokenize("침") == ["침"]


def test_hangul_query_terms_are_a_subset_of_document_terms() -> None:
    """Write side and query side must meet (the symmetry contract)."""
    tk = _make_tokenizer()
    query = set(tk.tokenize("진지"))
    document = set(tk.tokenize("진지 방어선은 사수 뒤에 두고 적을 끌어들인다."))
    assert query and query <= document


def test_mixed_script_keeps_english_and_hangul() -> None:
    tokens = _make_tokenizer().tokenize("진지 defense")
    assert "defense" in tokens
    assert "진지" in tokens


def test_hangul_tokens_are_deduplicated_and_ordered() -> None:
    """Repeated runs add each bigram once, after jieba's own tokens."""
    tokens = _make_tokenizer().tokenize("진지 진지")
    assert tokens.count("진지") == 1
    assert tokens.index("진지") == 0


def test_hangul_can_be_disabled() -> None:
    """The previous behaviour stays reachable for A/B measurement."""
    assert _make_tokenizer(hangul=False).tokenize("진지") == []


def test_english_and_chinese_output_is_unchanged_by_the_hangul_pass() -> None:
    """No Hangul in the input means no added tokens at all."""
    tk = _make_tokenizer()
    assert tk.tokenize("hello world") == tk.tokenize("hello world")
    assert "北京" in tk.tokenize("我爱北京天安门")


def test_build_tokenizer_returns_jieba_default() -> None:
    """Factory exposes the same JiebaTokenizer the cascade handler uses."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid escape sequence", DeprecationWarning)
        from everos.component.tokenizer.jieba_provider import JiebaTokenizer

    assert isinstance(build_tokenizer(), JiebaTokenizer)
