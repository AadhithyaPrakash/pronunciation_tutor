"""
Shared UI helpers for Streamlit pages.
"""

from __future__ import annotations

import streamlit as st


_BASE_CHROME_CSS = """
<style>
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebarNav"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2.5rem !important;
}
</style>
"""


def configure_page(*, title: str, icon: str, layout: str = "wide") -> None:
    """Apply a consistent page config and hide Streamlit's built-in page chrome."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="collapsed",
    )
    st.markdown(_BASE_CHROME_CSS, unsafe_allow_html=True)
