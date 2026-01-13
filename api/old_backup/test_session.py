# Test in Python console or create test_session.py
from session_manager import SessionManager

sm = SessionManager()
session_id = sm.create_session("test_user", "What is this?")
print(f"Created session: {session_id}")

sm.add_conversation_turn(
    "test_user",
    session_id,
    "What is this?",
    "This is a test response"
)

history = sm.get_conversation_history("test_user", session_id)
print(f"History: {history}")