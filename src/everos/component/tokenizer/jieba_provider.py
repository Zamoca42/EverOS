"""Jieba-based tokenizer — covers CJK + English mixed content.

Uses ``jieba.cut_for_search`` (search-mode segmentation: yields both the
greedy max-match segment and its finer sub-segments for compound CJK
words). Same mode as the legacy enterprise keyword-search path uses on
the query side — keeping cascade write and search query symmetric is
the hard contract for BM25 recall to work.

Hangul is handled separately (see :func:`_hangul_runs`): jieba's
dictionary is Chinese, so Korean text is not segmented — ``cut_for_search``
yields one character per syllable and the ``min_token_length`` filter then
drops every one of them. That made Korean queries tokenise to ``[]`` and
``sparse_recall`` return ``[]`` before touching the index (BM25 keyword
search silently returned nothing for Korean while English/Chinese worked).
Hangul runs are therefore tokenised here with character bigrams, the
standard segmentation for languages without word delimiters and the same
shape jieba produces for CJK compounds.

After segmentation we drop:

* whitespace / empty tokens (so the join-on-space output stays clean),
* tokens shorter than ``min_token_length`` (default 2 — same threshold
  enterprise's ``filter_stopwords(min_length=2)`` uses; single-char
  fragments mostly hurt BM25 precision). A **one-syllable Hangul run** is a
  whole word rather than a fragment, so it is kept regardless,
* tokens in a small bilingual stopword set (Chinese function words +
  English articles / prepositions / aux verbs),
* exact duplicates introduced by the Hangul pass (jieba's own output is
  passed through unchanged).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Final

with warnings.catch_warnings():
    # jieba 0.42.1 (its last release; unmaintained) ships regex string
    # literals with invalid escape sequences. Python >= 3.12 flags these as
    # SyntaxWarning at first-import compile time (before the .pyc is cached),
    # leaking noise on a user's first run. The warnings are harmless and out
    # of our control, so we suppress them at the single jieba import site.
    # See https://github.com/EverMind-AI/EverOS/issues/304.
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba

# Modern Korean is written in precomposed Hangul syllables; jamo blocks are
# left out on purpose so decomposed spellings stay out of the BM25 index.
_HANGUL_RUN: Final[re.Pattern[str]] = re.compile(r"[\uac00-\ud7a3]+")

# Small bilingual stopword set. Intentionally tight (not a full
# Chinese stopword list) so the behaviour is predictable; callers
# tuning recall can subclass / extend.
_DEFAULT_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # English — articles / prepositions / aux verbs that dominate BM25
        # idf-noise but add no recall value.
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        # Chinese — function words / particles. ``cut_for_search`` emits
        # these as single-char tokens anyway, and the min_length=2 floor
        # would drop most; listing them explicitly makes the intent clear
        # and is a no-op when min_length filtering also kicks in.
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "你",
        "他",
        "她",
        "它",
        "也",
        "都",
        "就",
        "还",
        "或",
        "及",
        "与",
        "对",
        "把",
        "被",
        "有",
        "没",
        "不",
        "啊",
        "吗",
        "呢",
        "吧",
        "哦",
    }
)

_DEFAULT_MIN_TOKEN_LENGTH: Final[int] = 2


def _hangul_runs(text: str) -> list[str]:
    """Tokenise Hangul runs by character bigram (single-syllable runs as-is).

    Korean is not in jieba's dictionary, so ``cut_for_search`` splits it into
    single syllables and ``min_token_length`` discards them all — a Korean
    query then reaches BM25 with zero terms. Character bigrams are the
    standard segmentation for scripts without word delimiters: the query
    ``진지`` yields ``진지``, and a document containing ``진지 방어선`` yields
    ``진지``, ``방어``, ``어선``, so the two sides meet on the same terms.
    A one-syllable run (``침``) is kept whole — dropping it would make
    single-syllable words unsearchable, the exact defect this fixes.
    """
    tokens: list[str] = []
    for run in _HANGUL_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class JiebaTokenizer:
    """Tokenizer that calls into ``jieba.cut_for_search`` and filters."""

    def __init__(
        self,
        *,
        min_token_length: int = _DEFAULT_MIN_TOKEN_LENGTH,
        extra_stopwords: frozenset[str] | None = None,
        hangul: bool = True,
    ) -> None:
        # Touching ``jieba.initialize()`` here would force eager dict load
        # at import time and balloon test-collection latency. ``jieba.cut*``
        # lazy-loads on first call instead.
        self._min_len = min_token_length
        self._hangul = hangul
        self._stopwords = (
            _DEFAULT_STOPWORDS | extra_stopwords
            if extra_stopwords
            else _DEFAULT_STOPWORDS
        )

    def _keep(self, raw: str) -> str | None:
        """Normalise one candidate token, or return ``None`` when filtered."""
        tok = raw.strip().lower()
        if not tok or tok.isspace():
            return None
        if len(tok) < self._min_len:
            return None
        if tok in self._stopwords:
            return None
        return tok

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        for raw in jieba.cut_for_search(text):
            tok = self._keep(raw)
            if tok is not None:
                out.append(tok)
        if self._hangul:
            # Append only new Hangul terms so jieba's own output (and its
            # duplicates) keeps the exact shape existing callers expect.
            seen = set(out)
            for tok in _hangul_runs(text):
                if len(tok) == 1:
                    # A one-syllable run is a whole word (침/spike, 적/enemy),
                    # not a fragment of a longer one: the min_length filter is
                    # about fragments, so it does not apply here.
                    kept = tok.strip().lower()
                    if not kept or kept in self._stopwords:
                        continue
                else:
                    kept = self._keep(tok)
                    if kept is None:
                        continue
                if kept not in seen:
                    seen.add(kept)
                    out.append(kept)
        return out

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]
