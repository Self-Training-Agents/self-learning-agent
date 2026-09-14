"""
memory_extractor.py

Extracts long-term, user-relevant memories from a raw conversation
using Gemini. This module has ONE job: turn conversation text into a
clean list of memory dicts. It does NOT know about user_id, does NOT
talk to MemoryStore, and does NOT decide novelty vs. duplicates --
that logic belongs to memory_manager.py / memory_scorer.py.

Output shape (per memory) matches what MemoryStore.add_memory expects,
minus user_id:
    {
        "memory": str,
        "type": str,          # one of ALLOWED_TYPES
        "importance": float,  # 0.0 - 1.0
        "confidence": float   # 0.0 - 1.0
    }
"""

import os
import json
import re
import logging
from typing import List, Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_EXTRACTOR_MODEL", "gemini-1.5-flash")
_model = genai.GenerativeModel(MODEL_NAME)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ALLOWED_TYPES = {"preference", "interest", "goal", "skill", "fact"}

_EXTRACTION_PROMPT = """You are a memory extraction system.

From the conversation below, extract ONLY long-term user information
worth remembering across future sessions.

Save things like:
- preferences (e.g. "User prefers Python for programming")
- interests (e.g. "User enjoys AI projects")
- goals (e.g. "User wants to become a backend developer")
- skills (e.g. "User knows FastAPI")
- durable personal facts (e.g. "User works at a fintech startup")

DO NOT save:
- greetings, small talk, or filler
- temporary/one-off events ("User is tired today")
- questions the user asked without stating a fact about themselves
- anything already implied to be short-term or session-only

Return ONLY a valid JSON array (even if there is just one memory, or none).
Do not include any text, explanation, or markdown fences outside the JSON.

Each item in the array must look like:
{{
  "memory": "User prefers Python for programming",
  "type": "preference",
  "importance": 0.9,
  "confidence": 0.95
}}

Allowed values for "type": preference, interest, goal, skill, fact.
"importance" and "confidence" must be numbers between 0 and 1.

If there is nothing worth remembering, return: []

Conversation:
{conversation}
"""


def _strip_code_fences(text: str) -> str:
    """Remove ```json / ``` fences if the model wraps its output in them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json_payload(text: str) -> str:
    """
    Best-effort extraction of a JSON array/object from arbitrary model
    output. Falls back to grabbing the first [...] or {...} block if the
    model added stray prose around the JSON.
    """
    text = _strip_code_fences(text)

    # Fast path: the whole thing already parses.
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find the first balanced [...] block.
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    # Fallback: find the first balanced {...} block (single object case).
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        return obj_match.group(0)

    return text


def _coerce_float(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _validate_and_clean(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate each extracted item and coerce it into a clean, safe shape."""
    cleaned: List[Dict[str, Any]] = []

    for item in raw_items:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict memory item: %r", item)
            continue

        memory_text = str(item.get("memory", "")).strip()
        if not memory_text:
            logger.warning("Skipping memory item with empty 'memory' field: %r", item)
            continue

        memory_type = str(item.get("type", "fact")).strip().lower()
        if memory_type not in ALLOWED_TYPES:
            logger.info(
                "Unrecognized memory type '%s', defaulting to 'fact'", memory_type
            )
            memory_type = "fact"

        cleaned.append({
            "memory": memory_text,
            "type": memory_type,
            "importance": _coerce_float(item.get("importance"), default=0.5),
            "confidence": _coerce_float(item.get("confidence"), default=0.5),
        })

    return cleaned


def extract_memories(conversation: str) -> List[Dict[str, Any]]:
    """
    Extract long-term memories from a conversation string.

    Args:
        conversation: raw conversation text (e.g. "User: ...\\nAssistant: ...")

    Returns:
        A list of memory dicts (possibly empty). Each dict has:
        "memory", "type", "importance", "confidence".
        Never raises on malformed model output -- returns [] instead,
        so a bad LLM response doesn't crash the calling pipeline.
    """
    if not conversation or not conversation.strip():
        logger.info("Empty conversation passed to extract_memories; skipping.")
        return []

    prompt = _EXTRACTION_PROMPT.format(conversation=conversation)

    try:
        response = _model.generate_content(prompt)
        raw_text = (response.text or "").strip()
    except Exception:
        logger.exception("Gemini call failed during memory extraction.")
        return []

    if not raw_text:
        logger.warning("Empty response from model during memory extraction.")
        return []

    payload = _extract_json_payload(raw_text)

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from model output: %r", raw_text)
        return []

    # Model might return a single object instead of a list -- normalize it.
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        logger.error("Unexpected JSON shape from model (not list/dict): %r", parsed)
        return []

    return _validate_and_clean(parsed)


# -------------------------
# Manual test
# -------------------------
if __name__ == "__main__":
    sample_conversation = """
User: I prefer Python for programming.
User: I enjoy AI projects.
User: hey what's up
"""

    memories = extract_memories(sample_conversation)

    print("Extracted Memories:")
    print(json.dumps(memories, indent=4))