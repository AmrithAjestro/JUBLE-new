import hashlib
import json
import secrets
from pathlib import Path

import streamlit as st

USER_DB_FILE = Path(__file__).resolve().parent.parent / "user_db.json"
PBKDF2_ITERATIONS = 200_000


# ---------- Utility Functions ----------
def load_user_db():
    """Load user credentials from JSON file."""
    if USER_DB_FILE.exists():
        with open(USER_DB_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}


def save_user_db(user_db):
    """Save user credentials to JSON file."""
    with open(USER_DB_FILE, "w", encoding="utf-8") as file:
        json.dump(user_db, file, indent=4)


def hash_password(password: str, salt: str | None = None) -> str:
    """Salted PBKDF2-HMAC-SHA256 password hashing. Stored as 'salt$hash'."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> tuple[bool, str | None]:
    """
    Verify a password against a stored hash.
    Returns (is_valid, upgraded_hash). upgraded_hash is set when a legacy
    unsalted SHA-256 hash matched, so the caller can migrate it in place.
    """
    if "$" in stored:
        salt, _ = stored.split("$", 1)
        return secrets.compare_digest(hash_password(password, salt), stored), None

    # Legacy unsalted SHA-256 hash — verify, then signal an upgrade.
    legacy_hash = hashlib.sha256(password.encode()).hexdigest()
    if secrets.compare_digest(legacy_hash, stored):
        return True, hash_password(password)
    return False, None


# ---------- Login/Register UI ----------
def login_ui():
    st.title("🔐 Jublie Login Portal")

    user_db = load_user_db()

    option = st.radio("Select an option", ("Login", "Register"))

    if option == "Login":
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            valid, upgraded_hash = (
                verify_password(password, user_db[username]) if username in user_db else (False, None)
            )
            if valid:
                if upgraded_hash:
                    user_db[username] = upgraded_hash
                    save_user_db(user_db)
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.current_page = "law"
                st.success(f"Welcome back, {username} 👋")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif option == "Register":
        new_username = st.text_input("New Username").strip()
        new_password = st.text_input("New Password", type="password")
        register_btn = st.button("Register")

        if register_btn:
            if new_username in user_db:
                st.error("Username already exists. Please choose another.")
            elif not new_username or not new_password:
                st.error("Both fields are required.")
            else:
                user_db[new_username] = hash_password(new_password)
                save_user_db(user_db)
                st.success("Registration successful! You can now log in.")

