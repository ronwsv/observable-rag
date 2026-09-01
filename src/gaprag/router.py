"""Query router — the agent/tool-use layer.

Before retrieving, an LLM decides between two "tools": search the knowledge base
(``retrieve``) or answer directly (``direct``). This skips retrieval for small talk
and pure-reasoning requests, saving latency and cost, and avoids polluting simple
answers with irrelevant context. The model must reply with strict JSON; if parsing
fails for any reason, the router degrades gracefully to a safe heuristic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .providers import LLMProvider

ACTIONS = {"retrieve", "direct"}

_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|bye|good (morning|afternoon|evening))\b", re.I
)
_JSON_RE = re.compile(r"\{.*\}", re.S)

ROUTER_SYSTEM = """You are a router for a question-answering system. Decide which tool to use.

Tools:
- "retrieve": search the knowledge base, then answer. Use for any factual or \
informational question that a documentation corpus could answer.
- "direct": answer without searching. Use only for greetings, small talk, thanks, \
or requests to reformat/rephrase the previous answer.

Respond with a single JSON object and nothing else:
{"action": "retrieve" | "direct", "reason": "<short reason>"}"""


@dataclass
class Route:
    action: str
    reason: str


def _heuristic(query: str) -> Route:
    if _SMALLTALK.match(query):
        return Route("direct", "matched small-talk pattern")
    return Route("retrieve", "default to retrieval for informational queries")


class Router:
    """LLM-based router with a deterministic heuristic fallback."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def route(self, query: str) -> Route:
        try:
            result = self.llm.complete(ROUTER_SYSTEM, query, temperature=0.0)
            match = _JSON_RE.search(result.text)
            if not match:
                return _heuristic(query)
            data = json.loads(match.group(0))
            action = str(data.get("action", "")).lower().strip()
            if action not in ACTIONS:
                return _heuristic(query)
            reason = str(data.get("reason", "")).strip() or "llm decision"
            return Route(action, reason)
        except Exception:
            return _heuristic(query)
