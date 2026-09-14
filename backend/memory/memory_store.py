import json
import os
from typing import List, Dict, Optional

# Path to the memory database
MEMORY_FILE = os.path.join("data", "memories.json")


class MemoryStore:
    def __init__(self):
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the memory file if it doesn't exist."""
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)

    def _load_memories(self) -> List[Dict]:
        """Load all memories from JSON."""
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_memories(self, memories: List[Dict]):
        """Save all memories to JSON."""
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=4)

    # -----------------------------
    # CREATE
    # -----------------------------
    def add_memory(
        self,
        user_id: str,
        memory: str,
        memory_type: str,
        importance: float,
        confidence: float
    ):
        """Add a new memory for a user."""

        memories = self._load_memories()

        memories.append({
            "user_id": user_id,
            "memory": memory,
            "type": memory_type,
            "importance": importance,
            "confidence": confidence
        })

        self._save_memories(memories)

    # -----------------------------
    # READ
    # -----------------------------
    def get_user_memories(self, user_id: str) -> List[Dict]:
        """Return all memories for one user."""

        memories = self._load_memories()

        return [
            m for m in memories
            if m["user_id"] == user_id
        ]

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search_memory(
        self,
        user_id: str,
        keyword: str
    ) -> List[Dict]:
        """Search memories containing a keyword."""

        keyword = keyword.lower()

        return [
            m
            for m in self.get_user_memories(user_id)
            if keyword in m["memory"].lower()
        ]

    # -----------------------------
    # UPDATE
    # -----------------------------
    def update_memory(
        self,
        user_id: str,
        old_memory: str,
        new_memory: str
    ) -> bool:
        """Update an existing memory."""

        memories = self._load_memories()

        updated = False

        for m in memories:
            if (
                m["user_id"] == user_id
                and m["memory"] == old_memory
            ):
                m["memory"] = new_memory
                updated = True

        self._save_memories(memories)

        return updated

    # -----------------------------
    # DELETE
    # -----------------------------
    def delete_memory(
        self,
        user_id: str,
        memory_text: str
    ) -> bool:
        """Delete one memory."""

        memories = self._load_memories()

        original_count = len(memories)

        memories = [
            m
            for m in memories
            if not (
                m["user_id"] == user_id
                and m["memory"] == memory_text
            )
        ]

        self._save_memories(memories)

        return len(memories) != original_count

    # -----------------------------
    # TOP MEMORIES
    # -----------------------------
    def get_top_memories(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Return the highest-importance memories."""

        memories = self.get_user_memories(user_id)

        memories.sort(
            key=lambda x: x["importance"],
            reverse=True
        )

        return memories[:limit]


# ------------------------------------
# Example usage (for testing)
# ------------------------------------
if __name__ == "__main__":

    store = MemoryStore()

    store.add_memory(
        user_id="user123",
        memory="User prefers Python",
        memory_type="preference",
        importance=0.9,
        confidence=0.95
    )

    store.add_memory(
        user_id="user123",
        memory="User likes AI",
        memory_type="interest",
        importance=0.8,
        confidence=0.9
    )

    print("\nAll memories:")
    print(store.get_user_memories("user123"))

    print("\nSearch 'Python':")
    print(store.search_memory("user123", "Python"))

    store.update_memory(
        "user123",
        "User prefers Python",
        "User prefers Java"
    )

    print("\nAfter update:")
    print(store.get_user_memories("user123"))

    print("\nTop memories:")
    print(store.get_top_memories("user123"))