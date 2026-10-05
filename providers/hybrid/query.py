"""Deterministic research-query planning: video title -> factual search queries.

Why this exists
---------------
Stage 1 produces a *video* title for humans ("Do Black Holes Actually Shrink?").
That string is a poor search query: it carries a rhetorical frame ("Do ... ?"),
an intensifier ("Actually"), and an implied audience reaction. Feeding it to a
search engine verbatim — as the Firecrawl retriever does — biases discovery
toward video-essay listicles and social/UGC pages rather than the factual
sources Stage 3 needs for corroboration.

`TopicDescriptor.description`, by contrast, is written as prose about the
subject and usually names the actual mechanism ("Hawking radiation", "lose
mass"). This module turns that available text into a small ranked set of
*factual* queries, in order of expected precision.

Design rules
------------
1. **Deterministic.** No LLM call, no randomness, no network. The same topic
   always yields exactly the same queries, so a run is reproducible and an
   artifact can be re-derived from the topic alone.
2. **Generic.** Every rule operates on structure (interrogative framing, speech
   acts, stopwords, meta-sentences), never on subject matter. Nothing here
   knows what "black holes" or "Hawking radiation" are — those terms enter only
   because the Stage-1 description contains them.
3. **Nothing invented.** Queries are assembled from substrings of the topic's
   own title/description. The planner never adds a term, entity, or domain that
   the topic did not supply, so it cannot smuggle in an assumed subject.
4. **Traceable.** Every emitted query records the `rule` that produced it, and
   `ResearchQueryPlan.as_trace()` renders that as an auditable one-line summary.
   This is the whole audit story: a reader can see exactly which words came from
   the title, which from the description, and which rule fired.

Multiple queries, not one
-------------------------
The plan emits up to four variants rather than a single "best" query. Recall
across several phrasings is what widens the candidate pool enough for Stage 3's
`corroboration_min_sources = 2` to be satisfiable; the retriever merges and
deduplicates the results. This widens *recall* only — a source still has to win
fetch, extraction, and the model's own source-tiering to be used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from agents.research.provider import TopicDescriptor

DEFAULT_MAX_QUERIES = 4

# Leading tokens that signal a question rather than a subject. Stripped from the
# front of a title to recover the noun phrase underneath.
_QUESTION_LEADERS = frozenset(
    {
        "do",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "will",
        "would",
        "should",
        "has",
        "have",
        "am",
        "why",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "which",
        "how",
    }
)

# Emphasis words that carry no retrieval signal but strongly bias a search
# engine toward clickbait and opinion content.
_INTENSIFIERS = frozenset(
    {
        "actually",
        "really",
        "truly",
        "honestly",
        "basically",
        "literally",
        "simply",
        "just",
        "ever",
    }
)

# Vocabulary that describes the *artifact* rather than the subject: the medium,
# the format, the genre, or the production direction. A Stage-1 description is a
# brief for a scriptwriter, so it is full of these ("animation-friendly
# explainer", "cinematic deep dive"), and they describe the video, not the world.
#
# Left in a search query they are actively harmful: "animation-friendly
# explainer ..." retrieves animation-tool marketing pages (Adobe Express,
# flipanim.com, brush.ninja) instead of the factual sources Stage 3 must
# corroborate. That is a whole-category problem, not one bad word, so it is fixed
# as a category.
#
# Deliberately excluded are words that can legitimately denote subject matter:
# "film" (a thin surface layer), "motion", "visual" ("visual cortex"),
# "graphics", "narrative", "score", "cut", "story", "series" ("Taylor series"),
# "segment" ("body segment"), "script" ("Python script"), and "editing" ("gene
# editing"). Filtering those would delete real topical terms, which is the failure
# this change exists to prevent.
_PRODUCTION_VOCABULARY = frozenset(
    {
        # medium / format
        "animation",
        "animations",
        "animated",
        "animate",
        "animator",
        "animators",
        "explainer",
        "explainers",
        "video",
        "videos",
        "vlog",
        "vlogs",
        "documentary",
        "documentaries",
        "episode",
        "episodes",
        "trailer",
        "trailers",
        "teaser",
        "teasers",
        "thumbnail",
        "thumbnails",
        "footage",
        "broll",
        "cinematic",
        "infographic",
        "visualizer",
        "visualisation",
        "visualization",
        # delivery / narration
        "narration",
        "narrated",
        "narrator",
        "narrators",
        "voiceover",
        "voiceovers",
        "screenplay",
        "storyboard",
        "storytelling",
        # production direction
        "production",
    }
)

# Simple English inflections, stripped one at a time, so "animations",
# "animated" and "explaining" all reach their base form without a stemmer.
_PRODUCTION_SUFFIXES = ("ing", "ed", "er", "ers", "es", "s")

# Instruction openers used when the description is addressed to a scriptwriter
# rather than stating a fact about the subject.
_SPEECH_ACTS = (
    "an intuitive look at",
    "a deep dive into",
    "a visual guide to",
    "a beginner's guide to",
    "a beginner guide to",
    "a comprehensive guide to",
    "a closer look at",
    "a simple explanation of",
    "a friendly introduction to",
    "an introduction to",
    "an explanation of",
    "an exploration of",
    "a breakdown of",
    "a walkthrough of",
    "a journey into",
    "a tour of",
    "everything about",
    "explore",
    "examines",
    "examine",
    "covering",
    "covers",
    "introduces",
    "introduce",
    "breaks down",
    "dives into",
    "delves into",
)

# Sentences addressed to the production ("this topic covers ...") describe the
# video, not the world, so they are dropped from the factual query.
_META_SENTENCE = re.compile(
    r"\b(this (topic|video|episode|piece|channel)|we('ll| will)? |you('ll| will)? |"
    r"in this (video|episode|topic)|the audience|viewers?)\b",
    re.IGNORECASE,
)

# Generic English function words. Subject-matter words are deliberately absent.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them themselves
    then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself yourselves
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# "the theoretical concept of X" -> "X". A generic English frame: an article,
# up to two modifiers, a head noun, then "of". Purely structural.
_FILLER_HEAD = re.compile(
    r"^(?:the|a|an)\s+(?:[A-Za-z-]+\s+){0,2}[A-Za-z-]+\s+of\s+",
    re.IGNORECASE,
)

_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

# Function words that can be orphaned at either end of a clause once artifact
# vocabulary is removed ("An animation-friendly explainer on X" -> "on X").
# Only ever applied to a clause that actually had production words removed, so a
# normal description is never touched.
_ORPHAN_EDGE = (
    r"the|a|an|on|of|in|at|about|for|with|from|to|and|or|as|by|that|which|whose|when|where|while"
)
_ORPHAN_WORDS = frozenset(_ORPHAN_EDGE.split("|"))
_LEADING_ORPHAN = re.compile(rf"^(?:{_ORPHAN_EDGE})[.,]?\s+", re.IGNORECASE)
_TRAILING_ORPHAN = re.compile(rf"\s+(?:{_ORPHAN_EDGE})[.,]?$", re.IGNORECASE)


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip())


def _is_production_word(lowered: str) -> bool:
    """Is `lowered` (already lowercased) artifact vocabulary, in any inflection?"""
    word = lowered
    # At most a couple of steps: "animations" -> "animation", "explainers" ->
    # "explainer". Guarding the length keeps short words from being over-stripped.
    for _ in range(2):
        if word in _PRODUCTION_VOCABULARY:
            return True
        for suffix in _PRODUCTION_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        else:
            return False
    return word in _PRODUCTION_VOCABULARY


def _is_production_token(token: str) -> bool:
    """Does `token` name the artifact rather than the subject?

    Handles the general case rather than one word: a hyphenated compound whose
    head or modifier is artifact vocabulary ("animation-friendly", "video-driven")
    is itself a production descriptor, so the whole compound goes. A compound with
    no production part ("peer-reviewed", "state-of-the-art") is left untouched, so
    ordinary subject vocabulary survives verbatim.
    """
    token = (token or "").strip().strip(".,;:!?\"'()")
    if not token:
        return False
    if "-" in token:
        return any(_is_production_token(part) for part in token.split("-"))
    return _is_production_word(token.lower())


def _strip_production_vocabulary(sentence: str) -> str:
    """Remove artifact-describing words from a clause, leaving the subject.

    Returns `sentence` byte-identical when it contains none, so descriptions that
    never mention the artifact are unaffected.
    """
    if not any(_is_production_token(match.group(0)) for match in _WORD.finditer(sentence)):
        return sentence

    kept = [m.group(0) for m in _WORD.finditer(sentence) if not _is_production_token(m.group(0))]
    if not kept:
        return ""

    clause = _collapse(" ".join(kept))
    # Removing a word can strand a function word at either end.
    while True:
        stripped = _LEADING_ORPHAN.sub("", clause).strip()
        stripped = _TRAILING_ORPHAN.sub("", stripped).strip()
        if stripped == clause:
            break
        clause = stripped
    # Nothing but function words survived: the clause was entirely about the
    # artifact, so there is no honest query left to search for.
    if all(word.lower() in _ORPHAN_WORDS for word in clause.split()):
        return ""
    return clause


def _strip_filler_head(sentence: str) -> str:
    """Remove a leading abstract-noun frame, once, if it yields real content."""
    stripped = _FILLER_HEAD.sub("", sentence).strip()
    return stripped if len(stripped) >= 12 else sentence


def _title_subject(title: str) -> str:
    """Recover the subject noun phrase from a rhetorical video title.

    "Do Black Holes Actually Shrink?" -> "Black Holes Shrink"
    "How Does AI Actually Think Next Words?" -> "AI Think Next Words"

    Only leading interrogative frames and intensifiers are removed; the original
    word order is otherwise preserved, because rewriting word order is where a
    deterministic planner starts guessing.
    """
    text = _collapse(title).rstrip("?.! ")
    if not text:
        return ""
    tokens = [token for token in text.split(" ") if token]
    while tokens and tokens[0].lower().strip(",") in _QUESTION_LEADERS:
        tokens.pop(0)
    while tokens and tokens[0].lower().strip(",") in _INTENSIFIERS:
        tokens.pop(0)
    kept = [token for token in tokens if token.lower().strip(",") not in _INTENSIFIERS]
    subject = _collapse(" ".join(kept or tokens))
    # "Why Is the Sky Blue?" leaves a dangling article once the frame is gone.
    return _LEADING_ARTICLE.sub("", subject).strip() or subject


def _factual_sentences(description: str) -> list[str]:
    """Sentences of the description that state something about the subject.

    Drops meta sentences about the video itself, instruction openers, and clauses
    that are entirely about the artifact rather than the subject.
    """
    sentences: list[str] = []
    for raw in _SENTENCE_SPLIT.split(description or ""):
        sentence = _collapse(raw)
        if not sentence:
            continue
        if _META_SENTENCE.search(sentence):
            continue
        lowered = sentence.lower()
        for opener in _SPEECH_ACTS:
            if lowered.startswith(opener + " "):
                sentence = _collapse(sentence[len(opener) :])
                break
        if sentence:
            sentence = _strip_production_vocabulary(sentence.rstrip(". ")).rstrip(". ")
        if sentence:
            sentences.append(_strip_filler_head(sentence))
    return sentences


def _keywords(texts: Sequence[str], limit: int = 12) -> str:
    """Distinct content words from `texts`, in first-seen order.

    Order is preserved rather than sorted because earlier words carry the
    subject; only stopwords, single characters, and artifact vocabulary are
    dropped.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        for match in _WORD.finditer(text or ""):
            word = match.group(0)
            lowered = word.lower()
            if len(lowered) < 2 or lowered in _STOPWORDS or lowered in _INTENSIFIERS:
                continue
            if _is_production_token(word):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            kept.append(word)
            if len(kept) >= limit:
                return _collapse(" ".join(kept))
    return _collapse(" ".join(kept))


@dataclass(frozen=True)
class ResearchQuery:
    """One planned query plus the rule that produced it."""

    text: str
    rule: str


@dataclass(frozen=True)
class ResearchQueryPlan:
    """A ranked, deterministic set of factual queries for one topic."""

    topic_id: str
    queries: tuple[ResearchQuery, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(query.text for query in self.queries)

    def as_trace(self) -> str:
        """One auditable line: which rule produced which query."""
        if not self.queries:
            return f"{self.topic_id}: no research query could be derived"
        return f"{self.topic_id}: " + " | ".join(f"{q.rule}={q.text!r}" for q in self.queries)


def plan_research_queries(
    topic: TopicDescriptor,
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> ResearchQueryPlan:
    """Plan factual search queries for `topic`, highest precision first.

    Falls back to the raw title if the topic carries no usable description, and
    returns an empty plan only if the topic has no title either — in which case
    the caller must not invent a query.
    """
    title = _collapse(topic.title)
    sentences = _factual_sentences(topic.description)

    candidates: list[ResearchQuery] = []
    if sentences:
        # The leading factual sentence usually names the mechanism.
        candidates.append(ResearchQuery(text=sentences[0], rule="description_factual_clause"))
        if len(sentences) > 1:
            candidates.append(
                ResearchQuery(text=_collapse(". ".join(sentences[:2])), rule="description_clauses")
            )

    subject = _title_subject(title)
    if subject:
        candidates.append(ResearchQuery(text=subject, rule="title_subject"))

    combined = _keywords([title, *sentences])
    if combined:
        candidates.append(ResearchQuery(text=combined, rule="title_description_keywords"))

    if title:
        # Last resort for recall: the title exactly as Stage 1 wrote it.
        candidates.append(ResearchQuery(text=title, rule="title_verbatim"))

    ordered: list[ResearchQuery] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = _collapse(candidate.text)
        if not text:
            continue
        fingerprint = text.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        ordered.append(candidate)
        if len(ordered) >= max(1, max_queries):
            break

    return ResearchQueryPlan(topic_id=topic.topic_id, queries=tuple(ordered))


__all__ = (
    "DEFAULT_MAX_QUERIES",
    "ResearchQuery",
    "ResearchQueryPlan",
    "plan_research_queries",
)
