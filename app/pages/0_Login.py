"""
Page 0 - Login / Register
--------------------------
First page shown if user is not authenticated.
Tabs: Login | Register
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

from app.ui import configure_page
from infrastructure import database

configure_page(
    title="Pronunciation Checker - Login",
    icon="🗣️",
    layout="wide",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 12%, rgba(255, 94, 98, 0.20), transparent 28%),
        radial-gradient(circle at 88% 20%, rgba(45, 212, 191, 0.16), transparent 24%),
        linear-gradient(180deg, #08111f 0%, #0b1220 100%);
}
.block-container {
    max-width: 1180px !important;
}
.hero-wrap {
    padding: 2rem 0 1rem 0;
}
.hero-kicker {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.28);
    color: #fda4af;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-title {
    margin: 1rem 0 0.85rem 0;
    color: #f8fafc;
    font-size: clamp(2.7rem, 5vw, 4.8rem);
    line-height: 0.95;
    font-weight: 900;
    max-width: 8ch;
}
.hero-copy {
    max-width: 32rem;
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.8;
    margin-bottom: 1.5rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.25rem;
}
.feature-card {
    padding: 1rem;
    border-radius: 1rem;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.14);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
.feature-card strong {
    display: block;
    color: #f8fafc;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.feature-card span {
    color: #94a3b8;
    font-size: 0.87rem;
    line-height: 1.5;
}
.auth-intro {
    color: #e2e8f0;
    margin-bottom: 0.4rem;
    font-size: 1.7rem;
    font-weight: 800;
}
.auth-note {
    color: #94a3b8;
    margin-bottom: 0.75rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(9, 15, 29, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 1.35rem;
    box-shadow: 0 24px 70px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(14px);
}
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 700;
    color: #94a3b8;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f8fafc;
}
div[data-testid="stTextInputRootElement"] input {
    border-radius: 0.9rem;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.16);
}
div[data-testid="stTextInputRootElement"] input:focus {
    border-color: rgba(248, 113, 113, 0.55);
    box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.2);
}
div.stButton > button {
    min-height: 3rem;
    border-radius: 0.95rem;
    font-weight: 700;
}
@media (max-width: 980px) {
    .hero-wrap {
        padding-top: 0.25rem;
    }
    .feature-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

database.init_db()

if st.session_state.get("user_id"):
    st.switch_page("streamlit_app.py")

hero_col, form_col = st.columns([1.15, 0.85], gap="large")

with hero_col:
    st.markdown(
        """
<div class="hero-wrap">
    <div class="hero-kicker">Pronunciation Lab</div>
    <h1 class="hero-title">Speak better. Fix the exact sound.</h1>
    <p class="hero-copy">
        Record an English sentence, confirm the transcript, and get precise
        phoneme-level feedback with progress saved to your profile.
    </p>
    <div class="feature-grid">
        <div class="feature-card">
            <strong>Phoneme-Level Review</strong>
            <span>See expected and detected sounds word by word.</span>
        </div>
        <div class="feature-card">
            <strong>Transcript Cleanup</strong>
            <span>Review what the app heard before analysis starts.</span>
        </div>
        <div class="feature-card">
            <strong>Progress History</strong>
            <span>Track your scores and common trouble sounds over time.</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with form_col:
    with st.container(border=True):
        st.markdown('<div class="auth-intro">Welcome back</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-note">Sign in to continue your pronunciation practice, or create a new account to get started.</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = database.login_user(username.strip(), password)
                    if user:
                        st.session_state["user_id"] = user["id"]
                        st.session_state["user_name"] = user["name"]
                        st.session_state["username"] = user["username"]
                        st.session_state["user_email"] = user.get("email", "")
                        st.success(f"Welcome back, {user['name']}!")
                        st.switch_page("streamlit_app.py")
                    else:
                        st.error("Invalid username or password.")

        with tab_register:
            r_name = st.text_input("Full Name", key="reg_name")
            r_username = st.text_input("Username", key="reg_username")
            r_email = st.text_input("Email ID", key="reg_email")
            r_password = st.text_input("Password", type="password", key="reg_pass")
            r_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

            if st.button("Create Account", type="primary", use_container_width=True, key="btn_register"):
                if not all([r_name, r_username, r_email, r_password, r_confirm]):
                    st.error("Please fill in all fields.")
                elif r_password != r_confirm:
                    st.error("Passwords do not match.")
                elif len(r_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif "@" not in r_email:
                    st.error("Please enter a valid email address.")
                else:
                    user_id = database.register_user(
                        name=r_name.strip(),
                        username=r_username.strip(),
                        email=r_email.strip(),
                        password=r_password,
                    )
                    if user_id:
                        st.session_state["user_id"] = user_id
                        st.session_state["user_name"] = r_name.strip()
                        st.session_state["username"] = r_username.strip()
                        st.session_state["user_email"] = r_email.strip()
                        st.success(f"Account created. Welcome, {r_name}!")
                        st.switch_page("streamlit_app.py")
                    else:
                        st.error("Username or email already registered. Please login instead.")
