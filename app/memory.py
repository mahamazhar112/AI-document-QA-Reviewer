class ConversationMemory:
    def __init__(self):
        self.sessions: dict[str, list[dict]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def get_last_review_id(self, session_id: str) -> int | None:
        history = self.sessions.get(session_id, [])
        for message in reversed(history):
            if message.get("review_id"):
                return message["review_id"]
        return None

    def set_last_review_id(self, session_id: str, review_id: int):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": "system", "content": "review_saved", "review_id": review_id})

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)


memory = ConversationMemory()