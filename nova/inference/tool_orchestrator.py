"""NOVA tool orchestration — web search integration and knowledge augmentation.

Detects knowledge gaps in prompts and augments them with external context
before passing to bracket inference. Placeholder search backends can be
swapped for real APIs (Brave, Serper, SerpAPI, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Knowledge gap detection patterns ─────────────────────────


KNOWLEDGE_PATTERNS = re.compile(
    r"\b(?:"
    r"who\s+is|who\s+was|who\s+are|"
    r"what\s+year|what\s+date|when\s+did|when\s+was|"
    r"current|latest|recent|today|this\s+year|"
    r"capital\s+of|president\s+of|population\s+of|"
    r"how\s+tall|how\s+old|how\s+far|how\s+long|"
    r"where\s+is|where\s+was|where\s+did|"
    r"founded\s+in|invented\s+by|discovered\s+by|"
    r"what\s+is\s+the\s+name|what\s+is\s+the\s+capital|"
    r"stock\s+price|exchange\s+rate|weather|"
    r"released\s+in|published\s+in|born\s+in|died\s+in"
    r")\b",
    re.IGNORECASE,
)

CODE_SEARCH_PATTERNS = re.compile(
    r"\b(?:"
    r"how\s+to\s+implement|example\s+of|"
    r"best\s+practice|design\s+pattern|"
    r"library\s+for|package\s+for|"
    r"api\s+for|sdk\s+for|"
    r"stack\s*overflow|github"
    r")\b",
    re.IGNORECASE,
)

# Confidence threshold below which we trigger search even without keyword match
LOW_CONFIDENCE_THRESHOLD = 0.3


# ── Search result container ──────────────────────────────────


@dataclass
class SearchResult:
    query: str
    snippets: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    success: bool = False


# ── ToolOrchestrator ─────────────────────────────────────────


class ToolOrchestrator:
    """Detects when prompts need external knowledge and augments them.

    Three capabilities:
      - should_search: keyword + confidence heuristic for knowledge gaps
      - search_and_augment: web search -> extract -> augment prompt
      - code_search: search for code patterns/examples -> augment prompt

    The actual search backends are placeholders. Swap web_search() and
    _code_search_backend() with real API calls for production use.
    """

    def __init__(
        self,
        search_backend: str = "placeholder",
        max_snippets: int = 3,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ):
        self.search_backend = search_backend
        self.max_snippets = max_snippets
        self.confidence_threshold = confidence_threshold

    def should_search(
        self, prompt: str, model_confidence: float | None = None
    ) -> bool:
        """Detect whether this prompt likely needs external knowledge.

        Uses keyword matching first. If no keywords match but model
        confidence is below threshold, still returns True.
        """
        if KNOWLEDGE_PATTERNS.search(prompt):
            return True

        if model_confidence is not None and model_confidence < self.confidence_threshold:
            return True

        return False

    def should_code_search(self, prompt: str) -> bool:
        """Detect whether this prompt would benefit from code example search."""
        return bool(CODE_SEARCH_PATTERNS.search(prompt))

    def search_and_augment(self, prompt: str) -> str:
        """Web search for factual info, then augment the prompt with context.

        Returns the augmented prompt with search context prepended.
        If search fails or returns nothing, returns the original prompt.
        """
        query = self._extract_search_query(prompt)
        result = self.web_search(query)

        if not result.success or not result.snippets:
            return prompt

        context = self._format_context(result)
        return (
            f"Use the following reference information to help answer:\n"
            f"{context}\n\n"
            f"Question: {prompt}"
        )

    def code_search(self, prompt: str) -> str:
        """Search for code patterns/examples, augment prompt with approaches.

        Returns augmented prompt with relevant code approaches prepended.
        """
        query = self._extract_search_query(prompt)
        result = self._code_search_backend(query)

        if not result.success or not result.snippets:
            return prompt

        approaches = "\n".join(
            f"- {snippet}" for snippet in result.snippets[:self.max_snippets]
        )
        return (
            f"Relevant approaches from documentation:\n"
            f"{approaches}\n\n"
            f"Task: {prompt}"
        )

    def web_search(self, query: str) -> SearchResult:
        """Execute a web search query.

        Placeholder implementation. Replace with real search API:
          - Brave Search API
          - Serper.dev (Google)
          - SerpAPI
          - Tavily
        """
        # Placeholder: return empty result
        # Real implementation would call an API and parse results
        return SearchResult(
            query=query,
            snippets=[],
            sources=[],
            success=False,
        )

    def _code_search_backend(self, query: str) -> SearchResult:
        """Search GitHub/StackOverflow for code patterns.

        Placeholder implementation. Replace with real API:
          - GitHub Search API
          - StackOverflow API
          - StackExchange API
        """
        return SearchResult(
            query=query,
            snippets=[],
            sources=[],
            success=False,
        )

    def _extract_search_query(self, prompt: str) -> str:
        """Extract a concise search query from the full prompt.

        Takes the first sentence or the first 100 chars, whichever is shorter.
        Strips common instruction prefixes.
        """
        # Remove instruction-style prefixes
        text = re.sub(
            r"^(?:solve|answer|explain|find|calculate|compute|write)\s*:?\s*",
            "", prompt, flags=re.IGNORECASE,
        )
        # Take first sentence
        first_sentence = re.split(r"[.!?\n]", text)[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100]
        return first_sentence

    def _format_context(self, result: SearchResult) -> str:
        """Format search results into a context block for prompt augmentation."""
        parts = []
        for i, snippet in enumerate(result.snippets[:self.max_snippets]):
            source = result.sources[i] if i < len(result.sources) else "web"
            parts.append(f"[{source}] {snippet}")
        return "\n".join(parts)
