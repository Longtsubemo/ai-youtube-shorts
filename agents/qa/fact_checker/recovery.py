"""Deterministic near-verbatim evidence recovery (recall backstop).

Stage 3's golden rule is that a **source is not evidence**: only `Evidence`
objects the provider actually obtained from the source's retrieved content may
back a claim. This module does not change that rule - it widens *recall* of
genuinely present text, never the integrity bar.

Why it exists
-------------
`EvidenceProvider.get_evidence(source)` is claim-blind: it asks the extractor
for whatever excerpts it judges worth keeping. When the single sentence that
happens to support a claim is not among the chosen excerpts, the source is
silently absent from that claim's evidence pool. The verifier then cannot
credit it no matter how good the source is, and the claim falls to REVIEW even
though the supporting text was on the page all along.

That is what happened to claim `cl_e552015c60` in run `e2e-20261002-192946`:
Wikipedia was bound, was eligible (`high_quality_secondary`), and was
retrieved, and evidence was extracted from it - but not the particular
supporting sentence, so only Britannica was credited.

What recovery does
------------------
Given a claim and the source's *already retrieved* content, find the one
sentence that is a near-verbatim restatement of the claim and hand it back as a
candidate. The caller still has to satisfy every existing rule before it counts
(verbatim containment, `Evidence.validate()`, the verifier, the gates).

What recovery is NOT
--------------------
- It is not keyword matching. Matching is a longest-common-subsequence ratio
  over word tokens, so it is order-sensitive: an unrelated sentence that
  happens to reuse a few claim words scores low and is rejected.
- It never invents text. A candidate is a span of the retrieved content.
- It never adds a source, changes a tier, or promotes a discovery source.
- It does not bypass the verifier. Recovered evidence simply joins the pool the
  verifier already judges; whether it counts is still the verifier's call.

Determinism: pure function of (claim text, content). No model, no network, no
randomness, no I/O.
"""

from __future__ import annotations

import re
import unicodedata

# Minimum share of the claim's tokens that must appear in the retrieved
# sentence, *in the claim's own order*, for the sentence to count as a
# near-verbatim restatement. Deliberately strict: the point is to recover a
# sentence that says the claim, not to find a loosely related one.
RECOVERY_MIN_LCS_RATIO = 0.75

# Minimum length of the longest contiguous shared token run. Bag-of-words
# coverage alone would accept keyword salad; requiring a real contiguous run
# is what keeps "related wording" from being mistaken for "the claim".
RECOVERY_MIN_RUN = 4

# Guards against trivially short claims, where a single shared token would be a
# large fraction of the claim and therefore easy to match by accident.
RECOVERY_MIN_CLAIM_TOKENS = 4

# A "sentence" longer than this is almost certainly a page dump or a table, not
# a restatement. Candidates longer than the cap are not recovered.
RECOVERY_MAX_SENTENCE_TOKENS = 120

# Sentence terminators.
_TERMINATORS = ".!?"

# Hard structural breaks in markdown: blank lines, ATX headings, list bullets,
# blockquote/table pipes. These end a block outright.
_HARD_BREAK = re.compile(r"\n\s*\n|^\s{0,3}#{1,6}\s|^\s*[-*+]\s|^\s*>", re.MULTILINE)

# Sentence boundary inside a block: terminator followed by whitespace.
_SENTENCE_BREAK = re.compile(rf"(?<=[{_TERMINATORS}])\s+")

# Smart punctuation folded to ASCII so a typographic apostrophe or en dash
# cannot break an otherwise verbatim match.
_PUNCTUATION_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "\u200b": "",
}

_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_for_recovery(text: str) -> str:
    """Fold Unicode and punctuation so near-verbatim text compares cleanly.

    NFKC handles full-width forms, ligatures and compatibility characters;
    the explicit table handles smart quotes, dashes and non-breaking space,
    which NFKC leaves alone and which would otherwise make an otherwise
    verbatim sentence look different. Case and whitespace are folded last.

    This is intentionally more conservative than the verifier's `_normalize`:
    the verifier compares token *sets* for coverage, whereas recovery compares
    token *sequences* for order, so folding punctuation away is required rather
    than merely convenient.
    """
    folded = unicodedata.normalize("NFKC", text)
    for source, target in _PUNCTUATION_FOLD.items():
        folded = folded.replace(source, target)
    return " ".join(folded.strip().lower().split())


def recovery_tokens(text: str) -> list[str]:
    """Word tokens in order, after Unicode/punctuation folding."""
    return _WORD.findall(normalize_for_recovery(text))


def _longest_common_subsequence_length(left: list[str], right: list[str]) -> int:
    """Length of the longest common subsequence of two token sequences.

    Order-sensitive by construction, which is exactly the property that
    separates "the claim, said again" from "a few of the claim's words, in some
    other order".
    """
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def _longest_shared_run(left: list[str], right: list[str]) -> int:
    """Length of the longest *contiguous* run shared by both token sequences."""
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for left_index, left_token in enumerate(left):
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
            else:
                current[index] = 0
        previous = current
    return best


def _split_blocks(content: str) -> list[str]:
    """Split markdown into prose blocks on hard structural breaks."""
    parts = [part.strip() for part in _HARD_BREAK.split(content or "")]
    return [part for part in parts if part]


def candidate_sentences(content: str) -> list[str]:
    """Deterministic sentence candidates from retrieved content.

    Sentences come from splitting each prose block on sentence terminators.
    Adjacent sentence pairs are also offered, because a claim is sometimes
    spread across two short sentences ("X is Y. It was found in 1971.") and the
    support only exists once they are read together.

    Text is returned as it appears in the content - never reworded - so the
    caller can run the existing verbatim containment check on it unchanged.
    """
    candidates: list[str] = []

    def add(sentence: str) -> None:
        stripped = sentence.strip()
        if not stripped:
            return
        token_count = len(recovery_tokens(stripped))
        if token_count == 0 or token_count > RECOVERY_MAX_SENTENCE_TOKENS:
            return
        if stripped not in candidates:
            candidates.append(stripped)

    for block in _split_blocks(content):
        pieces = [piece for piece in _SENTENCE_BREAK.split(block) if piece.strip()]
        for piece in pieces:
            add(piece)
        for left, right in zip(pieces, pieces[1:]):
            add(f"{left} {right}")
    return candidates


def near_verbatim_score(claim_text: str, sentence: str) -> float:
    """Share of the claim's tokens that the sentence restates *in order*.

    1.0 means the sentence contains every claim token in the claim's order.
    A sentence that merely reuses the claim's vocabulary scores lower because
    reordered and dropped tokens break the subsequence.
    """
    claim_tokens = recovery_tokens(claim_text)
    if not claim_tokens:
        return 0.0
    sentence_tokens = recovery_tokens(sentence)
    return _longest_common_subsequence_length(claim_tokens, sentence_tokens) / len(claim_tokens)


def is_near_verbatim(claim_text: str, sentence: str) -> bool:
    """Whether `sentence` is a near-verbatim restatement of `claim_text`.

    Requires both an order-preserving match over most of the claim and a real
    contiguous run, so neither a keyword salad nor a short accidental overlap
    qualifies.
    """
    claim_tokens = recovery_tokens(claim_text)
    if len(claim_tokens) < RECOVERY_MIN_CLAIM_TOKENS:
        return False
    sentence_tokens = recovery_tokens(sentence)
    if not sentence_tokens:
        return False
    ratio = _longest_common_subsequence_length(claim_tokens, sentence_tokens) / len(claim_tokens)
    if ratio < RECOVERY_MIN_LCS_RATIO:
        return False
    return _longest_shared_run(claim_tokens, sentence_tokens) >= RECOVERY_MIN_RUN


def find_near_verbatim(claim_text: str, content: str) -> str:
    """Best near-verbatim sentence in `content` for `claim_text`, or "".

    "Best" is the highest order-preserving score; ties are broken by position in
    the content, so the result is stable for identical inputs. Returns the
    sentence exactly as it appears in the content.
    """
    claim_tokens = recovery_tokens(claim_text)
    if len(claim_tokens) < RECOVERY_MIN_CLAIM_TOKENS:
        return ""

    best_sentence = ""
    best_score = 0.0
    for sentence in candidate_sentences(content):
        sentence_tokens = recovery_tokens(sentence)
        score = _longest_common_subsequence_length(claim_tokens, sentence_tokens) / len(claim_tokens)
        if score <= best_score:
            continue
        if not is_near_verbatim(claim_text, sentence):
            continue
        best_sentence, best_score = sentence, score
    return best_sentence
