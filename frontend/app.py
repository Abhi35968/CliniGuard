import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components
import html as html_lib
import base64
from src.graph import run_agent_turn
try:
    from frontend.streaming import stream_consultation_events
except ImportError:
    from streaming import stream_consultation_events
from src.policy.loader import PolicyLoader
from src.config import SOPS_FILE_PATH
from src.tools.weather import get_weather_description
import importlib
import src.auth
try:
    importlib.reload(src.auth)
except Exception:
    pass
from src.auth import (
    is_auth_enabled,
    authenticate_session,
    deauthenticate_session,
    check_rate_limit,
    record_query,
    get_usage_stats,
)
from src.db import (
    list_all_sessions,
    get_session_messages,
    delete_session,
    create_or_update_session,
    clear_all_sessions,
)
from src.observability import setup_logging, get_logger

# Initialize central logging for Streamlit frontend execution
setup_logging()
logger = get_logger("frontend")


# Page configuration
st.set_page_config(
    page_title="CliniGuard | Field Environmental Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# FIELD INTELLIGENCE DESIGN SYSTEM
# Editorial Typography • Scientific Telemetry • Warm Mineral Palette
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        /* Primary Brand Palette */
        --mineral-bg: #F3F1E8;
        --forest-ink: #183B32;
        --canopy-green: #376B50;
        --signal-lime: #D3EF68;
        --mist-surface: #E4E8DF;
        --white-surface: #FFFFFF;
        --text-primary: #183B32;
        --text-secondary: #53645B;
        --text-muted: #7A8B80;
        --border-warm: #D5DBD0;
        --border-subtle: #E2E7DE;
        --border-focus: #183B32;

        /* Semantic Palette */
        --semantic-verified: #2E7D5B;
        --semantic-verified-subtle: #E8F5EF;
        --semantic-caution: #C97A1E;
        --semantic-caution-subtle: #FDF5E8;
        --semantic-critical: #C6533D;
        --semantic-critical-subtle: #FDF0EE;
        --semantic-info: #2B6CB0;
        --semantic-info-subtle: #EBF3FA;
        --semantic-neutral: #829288;
        --semantic-neutral-subtle: #EEF2EC;

        /* Elevation & Geometry */
        --shadow-subtle: 0 1px 3px rgba(24, 59, 50, 0.05);
        --shadow-card: 0 2px 8px rgba(24, 59, 50, 0.06);
        --shadow-hover: 0 4px 14px rgba(24, 59, 50, 0.10);
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
    }

    /* Core Application Typography & Surfaces */
    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--mineral-bg) !important;
        background-image: radial-gradient(var(--border-warm) 0.75px, transparent 0.75px);
        background-size: 24px 24px;
    }

    /* Streamlit structural header and toolbar cleanup */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Navigation Rail */
    section[data-testid="stSidebar"] {
        background-color: var(--mist-surface) !important;
        border-right: 1px solid var(--border-warm) !important;
        box-shadow: 2px 0 12px rgba(24, 59, 50, 0.03);
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--border-warm) !important;
        margin: 1rem 0 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: "Inter", sans-serif;
        color: var(--forest-ink);
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1rem;
        margin-bottom: 0.45rem;
    }

    /* Field Intelligence Brand Mark & Header */
    .fi-shell-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.95rem 1.4rem;
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-card);
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }

    .fi-shell-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--forest-ink) 0%, var(--canopy-green) 50%, var(--signal-lime) 100%);
    }

    .fi-brand-lockup {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .fi-logo-symbol {
        width: 44px;
        height: 44px;
        border-radius: var(--radius-md);
        background-color: var(--forest-ink);
        color: var(--signal-lime);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(24, 59, 50, 0.2);
        position: relative;
    }

    .fi-logo-symbol::after {
        content: "";
        position: absolute;
        bottom: -2px;
        right: -2px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--signal-lime);
        border: 2px solid var(--white-surface);
    }

    .fi-brand-name {
        font-family: "Newsreader", "Georgia", serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--forest-ink);
        letter-spacing: -0.02em;
        line-height: 1.15;
    }

    .fi-brand-sub {
        font-size: 0.76rem;
        font-weight: 500;
        color: var(--text-secondary);
        letter-spacing: 0.02em;
        margin-top: 0.15rem;
        text-transform: uppercase;
    }

    .fi-header-metadata {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .fi-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        background-color: var(--semantic-verified-subtle);
        border: 1px solid var(--semantic-verified);
        color: var(--semantic-verified);
        letter-spacing: 0.02em;
    }

    .fi-status-degraded {
        background-color: var(--semantic-caution-subtle) !important;
        border-color: var(--semantic-caution) !important;
        color: var(--semantic-caution) !important;
    }

    .fi-session-tag {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.72rem;
        color: var(--forest-ink);
        background-color: var(--mist-surface);
        padding: 0.3rem 0.65rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-warm);
        letter-spacing: -0.02em;
    }

    /* Editorial Headline Banner */
    .fi-editorial-banner {
        padding: 1.1rem 1.4rem;
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-left: 4px solid var(--forest-ink);
        border-radius: var(--radius-md);
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-subtle);
    }

    .fi-editorial-headline {
        font-family: "Newsreader", "Georgia", serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--forest-ink);
        letter-spacing: -0.015em;
        line-height: 1.25;
    }

    .fi-editorial-desc {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    /* Scenario Starter Field Profiles */
    .fi-profiles-header {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 0.6rem;
    }

    .fi-scenario-card {
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-top: 3px solid var(--forest-ink);
        border-radius: var(--radius-md);
        padding: 0.85rem 1rem;
        margin-bottom: 0.45rem;
        box-shadow: var(--shadow-subtle);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .fi-scenario-card:hover {
        border-top-color: var(--signal-lime);
        border-color: var(--canopy-green);
    }

    .fi-scenario-num {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--canopy-green);
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .fi-scenario-title {
        font-family: "Newsreader", "Georgia", serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--forest-ink);
        line-height: 1.2;
    }

    .fi-scenario-detail {
        font-size: 0.74rem;
        color: var(--text-secondary);
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    /* Scientific Telemetry Metric Strip */
    .fi-telemetry-header {
        background-color: var(--forest-ink);
        color: var(--white-surface);
        padding: 0.85rem 1rem;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .fi-telemetry-title {
        font-family: "Newsreader", "Georgia", serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--white-surface);
    }

    .fi-telemetry-freshness {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.68rem;
        background-color: rgba(211, 239, 104, 0.15);
        color: var(--signal-lime);
        border: 1px solid rgba(211, 239, 104, 0.35);
        padding: 0.2rem 0.5rem;
        border-radius: var(--radius-sm);
        letter-spacing: 0.03em;
    }

    .fi-telemetry-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.65rem;
        margin-top: 0.65rem;
        margin-bottom: 0.85rem;
    }

    .fi-metric-cell {
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-radius: var(--radius-md);
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow-subtle);
        transition: border-color 0.15s ease;
    }

    .fi-metric-cell:hover {
        border-color: var(--canopy-green);
    }

    .fi-metric-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 0.2rem;
    }

    .fi-metric-number {
        font-family: "JetBrains Mono", monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--forest-ink);
        letter-spacing: -0.03em;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }

    .fi-metric-detail {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* Structured Clinical Advisory Presentation */
    .fi-advisory-container {
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-card);
        margin-bottom: 1.25rem;
        overflow: hidden;
    }

    .fi-advisory-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.1rem;
        background-color: var(--mist-surface);
        border-bottom: 1px solid var(--border-warm);
    }

    .fi-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .fi-pill-verified {
        background-color: var(--semantic-verified-subtle);
        border: 1px solid var(--semantic-verified);
        color: var(--semantic-verified);
    }

    .fi-pill-fallback {
        background-color: var(--semantic-info-subtle);
        border: 1px solid var(--semantic-info);
        color: var(--semantic-info);
    }

    .fi-pill-clarification {
        background-color: var(--semantic-caution-subtle);
        border: 1px solid var(--semantic-caution);
        color: var(--semantic-caution);
    }

    /* Risk Badges */
    .fi-risk-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: var(--radius-sm);
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .fi-risk-critical, .fi-risk-high {
        background-color: var(--semantic-critical-subtle);
        border: 1px solid var(--semantic-critical);
        color: var(--semantic-critical);
    }

    .fi-risk-moderate {
        background-color: var(--semantic-caution-subtle);
        border: 1px solid var(--semantic-caution);
        color: var(--semantic-caution);
    }

    .fi-risk-low, .fi-risk-safe {
        background-color: var(--semantic-verified-subtle);
        border: 1px solid var(--semantic-verified);
        color: var(--semantic-verified);
    }

    .fi-risk-info {
        background-color: var(--semantic-info-subtle);
        border: 1px solid var(--semantic-info);
        color: var(--semantic-info);
    }

    /* Provisional Streaming Draft Banner */
    .fi-provisional-banner {
        background-color: var(--semantic-caution-subtle);
        border-left: 4px solid var(--semantic-caution);
        border-radius: var(--radius-md);
        padding: 0.85rem 1.15rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-subtle);
    }

    .fi-provisional-title {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--semantic-caution);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .fi-provisional-sub {
        font-size: 0.74rem;
        color: var(--text-secondary);
        margin-top: 0.15rem;
    }

    .fi-provisional-body {
        margin-top: 0.55rem;
        font-size: 0.88rem;
        color: var(--forest-ink);
        line-height: 1.5;
    }

    /* Operational User Inquiries */
    .fi-user-inquiry-card {
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-left: 3px solid var(--canopy-green);
        border-radius: var(--radius-md);
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: var(--shadow-subtle);
    }

    .fi-user-inquiry-meta {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--canopy-green);
        margin-bottom: 0.25rem;
    }

    .fi-user-inquiry-text {
        font-size: 0.92rem;
        color: var(--forest-ink);
        font-weight: 500;
        line-height: 1.4;
    }

    /* Native Streamlit Buttons Styling */
    .stButton > button {
        background-color: var(--white-surface) !important;
        color: var(--forest-ink) !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 0.45rem 0.9rem !important;
        box-shadow: var(--shadow-subtle) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--mist-surface) !important;
        border-color: var(--forest-ink) !important;
        color: var(--forest-ink) !important;
        box-shadow: var(--shadow-card) !important;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--forest-ink) !important;
        color: var(--white-surface) !important;
        border: 1px solid var(--forest-ink) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: var(--canopy-green) !important;
        border-color: var(--canopy-green) !important;
        color: var(--white-surface) !important;
    }

    /* Native Inputs, Textarea, Selectbox */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        background-color: var(--white-surface) !important;
        border: 1px solid var(--border-warm) !important;
        color: var(--forest-ink) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.85rem !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--forest-ink) !important;
        box-shadow: 0 0 0 2px rgba(24, 59, 50, 0.12) !important;
    }

    /* Chat Input Bar */
    [data-testid="stChatInput"] {
        background-color: var(--white-surface) !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-card) !important;
        padding: 0.2rem !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--forest-ink) !important;
        box-shadow: 0 0 0 3px rgba(24, 59, 50, 0.14) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--forest-ink) !important;
        font-size: 0.90rem !important;
    }

    /* Tabs Component */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border-warm);
        padding-bottom: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.85rem !important;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--forest-ink) !important;
        border-bottom: 2px solid var(--forest-ink) !important;
        background-color: transparent !important;
    }

    /* Status Container */
    [data-testid="stStatusWidget"] {
        background-color: var(--white-surface) !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: var(--white-surface) !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    /* Markdown Tables (Editorial & Tabular Formatting) */
    .stMarkdown table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 0.85rem 0 !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    .stMarkdown th {
        background-color: var(--mist-surface) !important;
        color: var(--forest-ink) !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 0.55rem 0.85rem !important;
        border-bottom: 1px solid var(--border-warm) !important;
        text-align: left !important;
    }

    .stMarkdown td {
        background-color: var(--white-surface) !important;
        color: var(--forest-ink) !important;
        font-size: 0.82rem !important;
        padding: 0.5rem 0.85rem !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Code & Technical Blocks */
    code {
        font-family: "JetBrains Mono", monospace !important;
        font-size: 0.82rem !important;
        background-color: var(--mist-surface) !important;
        color: var(--forest-ink) !important;
        padding: 0.15rem 0.35rem !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-warm) !important;
    }

    /* Streamlit Chat Messages wrapper */
    div[data-testid="stChatMessage"] {
        background-color: var(--white-surface) !important;
        border: 1px solid var(--border-warm) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: var(--shadow-subtle) !important;
    }

    /* Sticky Right Inspector Panel */
    div[data-testid="column"]:nth-of-type(2) {
        position: sticky !important;
        top: 1rem !important;
        align-self: flex-start !important;
    }

    /* Responsive Queries */
    @media (max-width: 768px) {
        .fi-shell-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 0.75rem !important;
            padding: 0.85rem !important;
        }
        .fi-brand-name {
            font-size: 1.2rem !important;
        }
        .fi-telemetry-grid {
            grid-template-columns: 1fr !important;
        }
    }

    /* 1-Click Copy Advisory Button */
    .fi-copy-advisory-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--forest-ink);
        background-color: var(--mineral-bg);
        border: 1px solid var(--border-warm);
        border-radius: var(--radius-sm);
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        user-select: none;
    }

    .fi-copy-advisory-btn:hover {
        background-color: var(--signal-lime);
        color: var(--forest-ink);
        border-color: var(--canopy-green);
        box-shadow: 0 1px 4px rgba(24, 59, 50, 0.15);
    }

    .fi-copy-advisory-btn:active {
        transform: scale(0.97);
    }

    .fi-copy-advisory-btn.fi-copied {
        background-color: var(--forest-ink) !important;
        color: var(--signal-lime) !important;
        border-color: var(--forest-ink) !important;
    }

    /* Semantic SOP Cards & Retrieval Panel */
    .fi-sop-card {
        background-color: var(--white-surface);
        border: 1px solid var(--border-warm);
        border-left: 4px solid var(--border-warm);
        border-radius: var(--radius-md);
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: var(--shadow-subtle);
        transition: all 0.2s ease;
    }

    .fi-sop-card:hover {
        border-color: var(--canopy-green);
        box-shadow: 0 3px 10px rgba(24, 59, 50, 0.08);
    }

    .fi-sop-card-governing {
        border-left: 4px solid var(--forest-ink) !important;
        border-color: var(--canopy-green);
        background-color: #F8FBF7 !important;
        box-shadow: 0 2px 8px rgba(24, 59, 50, 0.1);
    }

    .fi-sop-badge-governing {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: var(--forest-ink);
        color: var(--signal-lime);
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .fi-sop-badge-matched {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: var(--semantic-verified-subtle);
        color: var(--semantic-verified);
        border: 1px solid var(--semantic-verified);
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
    }

    .fi-sop-badge-candidate {
        display: inline-flex;
        align-items: center;
        background-color: var(--mist-surface);
        color: var(--text-secondary);
        border: 1px solid var(--border-warm);
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
    }

    .fi-relevance-bar-track {
        width: 100%;
        height: 5px;
        background-color: var(--mist-surface);
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 0.3rem;
    }

    .fi-relevance-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--canopy-green), var(--signal-lime));
        border-radius: 9999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global Clipboard Copy Manager & Clear-Cache Keydown Interceptor
components.html(
    """
    <script>
    (function() {
        try {
            const pWin = window.parent;
            const pDoc = window.parent.document;
            if (!pDoc) return;

            if (!pDoc.getElementById('cliniguard-runtime-injected')) {
                const s = pDoc.createElement('script');
                s.id = 'cliniguard-runtime-injected';
                s.textContent = `
                    // 1. Intercept 'c' / 'C' keydown on window to suppress Streamlit's native 'Clear cache' modal
                    function interceptStreamlitClearCache(e) {
                        if (e.key === 'c' || e.key === 'C' || e.code === 'KeyC' || e.keyCode === 67) {
                            if (e.ctrlKey || e.metaKey || e.altKey) {
                                return; // Allow Ctrl+C copy
                            }
                            const active = document.activeElement;
                            const tag = active ? active.tagName.toLowerCase() : '';
                            const isEditable = active && (tag === 'input' || tag === 'textarea' || active.isContentEditable);
                            if (!isEditable) {
                                e.stopImmediatePropagation();
                                e.stopPropagation();
                                e.preventDefault();
                            }
                        }
                    }
                    window.addEventListener('keydown', interceptStreamlitClearCache, true);
                    document.addEventListener('keydown', interceptStreamlitClearCache, true);

                    // 2. Decode base64 UTF-8 text helper
                    function cliniguardDecodeB64(b64) {
                        const binString = window.atob(b64);
                        const bytes = Uint8Array.from(binString, c => c.charCodeAt(0));
                        return new TextDecoder().decode(bytes);
                    }

                    // 3. Global click listener for 1-click advisory copy buttons
                    document.addEventListener('click', function(e) {
                        const btn = e.target.closest('.fi-copy-advisory-btn');
                        if (!btn) return;

                        e.preventDefault();
                        e.stopPropagation();

                        const b64 = btn.getAttribute('data-copy-b64');
                        let textToCopy = '';
                        if (b64) {
                            try {
                                textToCopy = cliniguardDecodeB64(b64);
                            } catch(err) {
                                textToCopy = btn.getAttribute('data-copy-text') || '';
                            }
                        } else {
                            textToCopy = btn.getAttribute('data-copy-text') || '';
                        }

                        if (!textToCopy) return;

                        function showSuccess() {
                            btn.innerHTML = '✓ Copied to clipboard';
                            btn.classList.add('fi-copied');
                            setTimeout(() => {
                                btn.innerHTML = '📋 Copy Advisory';
                                btn.classList.remove('fi-copied');
                            }, 2000);
                        }

                        function fallbackCopy(text) {
                            const ta = document.createElement('textarea');
                            ta.value = text;
                            ta.style.position = 'fixed';
                            ta.style.left = '-9999px';
                            ta.style.top = '0';
                            document.body.appendChild(ta);
                            ta.focus();
                            ta.select();
                            try {
                                const successful = document.execCommand('copy');
                                if (successful) {
                                    showSuccess();
                                } else {
                                    btn.innerHTML = '⚠️ Press Ctrl+C to copy';
                                    setTimeout(() => { btn.innerHTML = '📋 Copy Advisory'; }, 2500);
                                }
                            } catch(err) {
                                btn.innerHTML = '⚠️ Press Ctrl+C to copy';
                                setTimeout(() => { btn.innerHTML = '📋 Copy Advisory'; }, 2500);
                            }
                            document.body.removeChild(ta);
                        }

                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(textToCopy)
                                .then(showSuccess)
                                .catch(() => fallbackCopy(textToCopy));
                        } else {
                            fallbackCopy(textToCopy);
                        }
                    }, true);
                `;
                pDoc.head.appendChild(s);
            }
        } catch(err) {
            console.warn('CliniGuard runtime injection notice:', err);
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ==============================================================================
# STATE & SYSTEM HELPERS
# ==============================================================================
def check_service_health():
    is_db_ok = False
    try:
        list_all_sessions()
        is_db_ok = True
    except Exception:
        is_db_ok = False

    is_api_mode = bool(st.session_state.get("custom_api_url") or os.getenv("CLINIGUARD_API_URL"))

    if is_db_ok:
        return {
            "status": "healthy",
            "badge_class": "fi-status-pill",
            "dot": "●",
            "label": "Engine Active" if not is_api_mode else "FastAPI Linked",
        }
    else:
        return {
            "status": "degraded",
            "badge_class": "fi-status-pill fi-status-degraded",
            "dot": "▲",
            "label": "Storage Degraded",
        }


def load_session_state(sid: str):
    msgs = get_session_messages(sid)
    st.session_state.chat_history = msgs
    last_assistant = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last_assistant:
        st.session_state.last_state = {
            "weather_data": last_assistant.get("weather_data"),
            "sop_citation": last_assistant.get("sop_citation"),
            "matched_sops": last_assistant.get("matched_sops", []),
            "active_sop": last_assistant.get("active_sop"),
            "retrieved_sops": last_assistant.get("retrieved_sops", []),
            "retrieval_error": last_assistant.get("retrieval_error"),
            "route": last_assistant.get("route"),
            "execution_trace": last_assistant.get("execution_trace", []),
            "grounding_valid": True,
        }
    else:
        st.session_state.last_state = None


def render_assistant_advisory(msg):
    route = msg.get("route")
    citation = msg.get("sop_citation")
    active_sop = msg.get("active_sop")
    matched = msg.get("matched_sops", [])

    # Determine Status Ribbon
    if route == "deterministic_fallback":
        status_pill = "<span class='fi-pill fi-pill-fallback'>▲ Deterministic Safety Fallback</span>"
    elif route == "clarification":
        status_pill = "<span class='fi-pill fi-pill-clarification'>◆ Clarification Required</span>"
    elif citation and citation != "None":
        status_pill = "<span class='fi-pill fi-pill-verified'>● Verified Clinical Advisory</span>"
    else:
        status_pill = "<span class='fi-pill fi-pill-verified'>● Evaluated Assessment</span>"

    # Determine Severity Badge
    sev = None
    if active_sop and active_sop.get("severity"):
        sev = active_sop.get("severity")
    elif matched and matched[0].get("severity"):
        sev = matched[0].get("severity")
    elif citation and citation == "[NO_SOP_APPLICABLE]":
        sev = "INFORMATIONAL"

    sev_badge = ""
    if sev:
        sev_badge = f"<span class='fi-risk-badge fi-risk-{sev.lower()}'>{sev} RISK</span>"

    sop_badge = ""
    if citation and citation != "None":
        sop_badge = f"<span class='fi-session-tag'>{citation}</span>"

    # Encode advisory text using base64 to ensure 100% safe HTML attribute passing without broken newlines
    raw_content = msg.get("content", "")
    b64_content = base64.b64encode(raw_content.encode("utf-8")).decode("ascii")
    copy_button_html = f"<button class='fi-copy-advisory-btn' data-copy-b64=\"{b64_content}\" title='Copy clinical advisory to clipboard'>📋 Copy Advisory</button>"

    header_html = (
        f"<div style='display: flex; justify-content: space-between; align-items: center; "
        f"padding-bottom: 0.65rem; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-subtle);'>"
        f"<div style='display: flex; align-items: center; gap: 0.5rem;'>{status_pill} {sev_badge}</div>"
        f"<div style='display: flex; align-items: center; gap: 0.55rem;'>{sop_badge} {copy_button_html}</div>"
        f"</div>"
    )

    footer_html = (
        f"<div style='margin-top: 0.85rem; padding-top: 0.55rem; border-top: 1px dashed var(--border-subtle); "
        f"display: flex; justify-content: space-between; align-items: center; font-size: 0.70rem; color: var(--text-muted);'>"
        f"<div>Telemetry: Open-Meteo WMO Surface Model • Protocol: Deterministically Grounded</div>"
        f"<div>{copy_button_html}</div>"
        f"</div>"
    )

    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown(raw_content)
        st.markdown(footer_html, unsafe_allow_html=True)


# Initialize Session State
saved_sessions = list_all_sessions()
if "session_id" not in st.session_state:
    if saved_sessions:
        st.session_state.session_id = saved_sessions[0]["session_id"]
    else:
        st.session_state.session_id = f"session-{int(time.time())}"
    load_session_state(st.session_state.session_id)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

policy_loader = PolicyLoader()

# ==============================================================================
# SIDEBAR: REFINED NAVIGATION & POLICY GOVERNANCE
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.9rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-warm);'>
            <div style='width: 32px; height: 32px; background: var(--forest-ink); color: var(--signal-lime); border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; font-weight: 700;'>
                🛡️
            </div>
            <div>
                <div style='font-family: "Newsreader", serif; font-size: 1.05rem; font-weight: 700; color: var(--forest-ink); line-height: 1.1;'>CliniGuard</div>
                <div style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary);'>Field Intelligence Rail</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Access Authentication Gate (if enabled)
    if is_auth_enabled():
        st.markdown("### Access Authentication")
        if not st.session_state.authenticated:
            key_input = st.text_input(
                "Access Key",
                type="password",
                placeholder="Enter authorized access key…",
                key="auth_key_input",
            )
            if st.button("Unlock Session", use_container_width=True, type="primary"):
                if authenticate_session(st.session_state.session_id, key_input):
                    st.session_state.authenticated = True
                    st.session_state.access_key = key_input
                    st.success("Session unlocked.")
                    st.rerun()
                else:
                    st.error("Invalid access credentials.")
            st.stop()
        else:
            col_auth1, col_auth2 = st.columns([3, 1])
            col_auth1.markdown("<span style='font-size: 0.80rem; color: var(--semantic-verified); font-weight: 600;'>🔒 Authenticated Session</span>", unsafe_allow_html=True)
            if col_auth2.button("Logout", use_container_width=True):
                deauthenticate_session(st.session_state.session_id)
                st.session_state.authenticated = False
                st.session_state.pop("access_key", None)
                st.rerun()

            usage = get_usage_stats(st.session_state.session_id)
            st.progress(
                value=min(1.0, usage["queries_in_window"] / max(usage["limit"], 1)),
                text=f"{usage['queries_in_window']}/{usage['limit']} queries/min",
            )
        st.divider()
    else:
        usage = get_usage_stats(st.session_state.session_id)
        if usage["queries_in_window"] > 0:
            st.progress(
                value=min(1.0, usage["queries_in_window"] / max(usage["limit"], 1)),
                text=f"{usage['queries_in_window']}/{usage['limit']} queries/min",
            )

    # 2. Consultation Threads
    st.markdown("### Consultations")
    if st.button("➕ New Consultation", use_container_width=True, type="primary"):
        new_sid = f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.last_state = None
        st.rerun()

    saved_sessions = list_all_sessions()
    session_options = {s["session_id"]: s["title"] for s in saved_sessions}
    current_sid = st.session_state.session_id
    if current_sid not in session_options:
        session_options[current_sid] = "Active Consultation"

    options_keys = list(session_options.keys())
    current_index = options_keys.index(current_sid) if current_sid in options_keys else 0

    selected_sid = st.selectbox(
        "Consultation Thread",
        options=options_keys,
        format_func=lambda sid: session_options.get(sid, sid),
        index=current_index,
        label_visibility="collapsed",
    )

    if selected_sid != st.session_state.session_id:
        st.session_state.session_id = selected_sid
        load_session_state(selected_sid)
        st.rerun()

    col_del1, col_del2 = st.columns(2)
    if col_del1.button("Delete Thread", use_container_width=True):
        delete_session(st.session_state.session_id)
        remaining = list_all_sessions()
        new_sid = remaining[0]["session_id"] if remaining else f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        load_session_state(new_sid)
        st.rerun()
        
    if col_del2.button("Clear All", use_container_width=True):
        clear_all_sessions()
        new_sid = f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.last_state = None
        st.rerun()

    st.divider()

    # 3. Policy Governance (SOP Management)
    st.markdown("### Policy Governance")
    sops = policy_loader.load_policies(force_reload=True)
    col_sop1, col_sop2 = st.columns([2.5, 1.5])
    col_sop1.markdown(f"<span style='font-size: 0.82rem; color: var(--forest-ink); font-weight: 600; line-height: 2;'><b>{len(sops)}</b> Active SOPs</span>", unsafe_allow_html=True)
    if col_sop2.button("Reload", use_container_width=True, help="Hot-reload sops.json from disk without engine restart"):
        sops = policy_loader.load_policies(force_reload=True)
        st.toast(f"Policy catalog reloaded ({len(sops)} active policies)", icon="🔄")

    with st.expander("Inspect Policy Catalog", expanded=False):
        for s in sops:
            sev = s.get("severity", "LOW")
            st.markdown(
                f"<span class='fi-risk-badge fi-risk-{sev.lower()}'>{sev}</span> <code>{s['id']}</code><br>"
                f"<div style='font-size: 0.82rem; color: var(--forest-ink); font-weight: 600; margin-top: 0.2rem;'>{s['title']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Activities: {', '.join(s.get('target_activities', []))}")
            st.markdown(f"<div style='font-size: 0.74rem; color: var(--text-secondary); line-height: 1.35;'>{s.get('guidance', '')[:85]}…</div>", unsafe_allow_html=True)
            st.markdown("<hr style='border: none; border-top: 1px solid var(--border-warm); margin: 0.45rem 0;'>", unsafe_allow_html=True)

    with st.expander("Author Custom Policy", expanded=False):
        new_id = st.text_input("Policy ID", value=f"SOP-CUSTOM-{len(sops)+1:03d}")
        new_title = st.text_input("Policy Title", value="Extreme Heat Sports & Athletics Policy")
        new_category = st.selectbox("Category", ["outdoor_exercise", "travel_commute", "vulnerable_groups", "outdoor_leisure"])
        new_activity = st.text_input("Target Activities", value="tennis, badminton, pickleball, soccer")
        new_severity = st.selectbox("Severity Level", ["CRITICAL", "HIGH", "MODERATE", "LOW"], index=1)
        new_temp = st.number_input("Threshold Temperature (°C)", value=36.0, step=1.0)
        new_guidance = st.text_area("Authorized Guidance", value="High intensity continuous movement under elevated thermal load creates acute hyperthermia risk.")
        new_precautions = st.text_area("Mandatory Precautions (1 per line)", value="Mandatory 10-minute hydration intervals\nWear UV-rated apparel and polarized eye protection\nAvoid exposed hard-court surfaces during peak solar hours")
        
        if st.button("Commit Policy to Catalog", use_container_width=True, type="primary"):
            new_sop_obj = {
                "id": new_id,
                "title": new_title,
                "category": new_category,
                "target_activities": [a.strip() for a in new_activity.split(",") if a.strip()],
                "target_demographics": ["all"],
                "severity": new_severity,
                "conditions": {
                    "type": "numeric",
                    "rules": [{"field": "temperature_2m", "op": ">=", "value": new_temp}]
                },
                "guidance": new_guidance,
                "precautions": [p.strip() for p in new_precautions.split("\n") if p.strip()]
            }
            with open(SOPS_FILE_PATH, "r", encoding="utf-8") as f:
                all_sops = json.load(f)
            all_sops.append(new_sop_obj)
            with open(SOPS_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(all_sops, f, indent=2)
            st.success(f"Committed {new_id} to catalog.")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # 4. System & Connectivity
    st.markdown("### System & Connectivity")
    default_api = os.getenv("CLINIGUARD_API_URL", "")
    custom_url = st.text_input(
        "FastAPI Service URL",
        value=st.session_state.get("custom_api_url", default_api),
        placeholder="e.g. http://localhost:8000",
        help="Leave blank to use direct in-process LangGraph engine.",
        key="custom_api_url_input",
    )
    st.session_state.custom_api_url = custom_url.strip()
    if st.session_state.custom_api_url:
        st.markdown(f"<div style='font-size: 0.76rem; color: var(--forest-ink);'>Mode: <b style='color: var(--canopy-green);'>FastAPI SSE</b> (<code>{st.session_state.custom_api_url}</code>)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.76rem; color: var(--text-secondary);'>Mode: <b>Direct In-Process LangGraph</b></div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN VIEW: SHELL HEADER
# ==============================================================================
health_info = check_service_health()
st.markdown(
    f"""
    <div class='fi-shell-header'>
        <div class='fi-brand-lockup'>
            <div class='fi-logo-symbol'>🛡️</div>
            <div>
                <div class='fi-brand-name'>CliniGuard</div>
                <div class='fi-brand-sub'>Field Environmental Safety Intelligence // Clinical SOP Governance</div>
            </div>
        </div>
        <div class='fi-header-metadata'>
            <div class='{health_info["badge_class"]}'>{health_info["dot"]} {health_info["label"]}</div>
            <div class='fi-session-tag'>{st.session_state.session_id}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Responsive two-column workspace
col_stream, col_inspector = st.columns([12, 9], gap="large")

# ==============================================================================
# RIGHT PANEL: SCIENTIFIC TELEMETRY & DECISION CENTER
# ==============================================================================
with col_inspector:
    tab_telemetry, tab_policy, tab_retrieval, tab_guardrails, tab_trace = st.tabs(
        ["Telemetry", "Policy Engine", "Semantic SOPs", "Guardrails", "Agent Trace"]
    )
    
    last_state = st.session_state.last_state
    weather = last_state.get("weather_data") if last_state else None
    
    with tab_telemetry:
        if weather:
            loc = weather.get("location_name") or last_state.get("location") or "Target Coordinate"
            temp = weather.get("temperature_2m", "N/A")
            app_temp = weather.get("apparent_temperature", temp)
            wind = weather.get("wind_speed_10m", "N/A")
            gusts = weather.get("wind_gusts_10m", wind)
            rain_prob = weather.get("precipitation_probability", "0")
            rain = weather.get("precipitation", "0.0")
            uv = weather.get("uv_index", "0.0")
            humidity = weather.get("relative_humidity_2m", "N/A")
            cond = weather.get("weather_description", "Verified")

            # Calibrate categorization labels without inventing values
            try:
                uv_f = float(uv)
                if uv_f < 3:
                    uv_risk_label = "Low Solar Radiation"
                elif uv_f < 6:
                    uv_risk_label = "Moderate UV Exposure"
                elif uv_f < 8:
                    uv_risk_label = "High UV Exposure"
                elif uv_f < 11:
                    uv_risk_label = "Very High UV Hazard"
                else:
                    uv_risk_label = "Extreme UV Hazard"
            except Exception:
                uv_risk_label = "Verified Solar Index"

            st.markdown(
                f"""
                <div class='fi-telemetry-header'>
                    <div class='fi-telemetry-title'>📍 {loc}</div>
                    <div class='fi-telemetry-freshness'>● Live Telemetry</div>
                </div>
                <div style='background-color: var(--white-surface); border: 1px solid var(--border-warm); border-top: none; padding: 0.75rem 1rem; margin-bottom: 0.65rem; border-radius: 0 0 var(--radius-md) var(--radius-md);'>
                    <div style='font-size: 0.78rem; color: var(--text-secondary);'>Synoptic Assessment: <b style='color: var(--forest-ink);'>{cond}</b></div>
                </div>
                
                <div class='fi-telemetry-grid'>
                    <div class='fi-metric-cell'>
                        <div class='fi-metric-label'>Ambient Temperature</div>
                        <div class='fi-metric-number'>{temp}°C</div>
                        <div class='fi-metric-detail'>Apparent: {app_temp}°C</div>
                    </div>
                    <div class='fi-metric-cell'>
                        <div class='fi-metric-label'>Precipitation Risk</div>
                        <div class='fi-metric-number'>{rain_prob}%</div>
                        <div class='fi-metric-detail'>Accumulation: {rain} mm</div>
                    </div>
                    <div class='fi-metric-cell'>
                        <div class='fi-metric-label'>Wind Velocity</div>
                        <div class='fi-metric-number'>{wind} <span style='font-size: 0.85rem; font-weight: 500;'>km/h</span></div>
                        <div class='fi-metric-detail'>Peak Gusts: {gusts} km/h</div>
                    </div>
                    <div class='fi-metric-cell'>
                        <div class='fi-metric-label'>Solar UV Index</div>
                        <div class='fi-metric-number'>{uv}</div>
                        <div class='fi-metric-detail'>{uv_risk_label} • RH: {humidity}%</div>
                    </div>
                </div>
                <div style='font-size: 0.70rem; color: var(--text-muted); text-align: right;'>
                    Data Source: Open-Meteo High-Precision Weather API • Calibrated Diurnal Cycle
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='background-color: var(--white-surface); border: 1px dashed var(--border-warm); border-radius: var(--radius-lg); padding: 2rem 1.5rem; text-align: center; box-shadow: var(--shadow-subtle);'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.5rem;'>📡</div>
                    <div style='font-family: "Newsreader", serif; font-size: 1.1rem; font-weight: 600; color: var(--forest-ink);'>Telemetry Standby</div>
                    <div style='font-size: 0.80rem; color: var(--text-secondary); margin-top: 0.35rem; line-height: 1.45;'>
                        Meteorological measurements will calibrate automatically when a target location and activity are submitted.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_policy:
        if last_state:
            citation = last_state.get("sop_citation", "None")
            matched = last_state.get("matched_sops", [])
            primary = last_state.get("active_sop") or (matched[0] if matched else None)
            
            if primary:
                sev = primary.get("severity", "LOW")
                precautions_html = ""
                if primary.get("precautions"):
                    items = "".join([f"<li style='margin-bottom: 0.25rem;'>{p}</li>" for p in primary.get("precautions")])
                    precautions_html = f"""
                    <div style='margin-top: 0.75rem;'>
                        <div style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.3rem;'>Mandatory Precautions:</div>
                        <ul style='font-size: 0.80rem; color: var(--forest-ink); margin: 0; padding-left: 1.1rem; line-height: 1.4;'>{items}</ul>
                    </div>
                    """

                st.markdown(
                    f"""
                    <div style='background: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-lg); padding: 1rem 1.15rem; margin-bottom: 0.85rem; box-shadow: var(--shadow-subtle);'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                            <span class='fi-risk-badge fi-risk-{sev.lower()}'>{sev} SEVERITY</span>
                            <code>{primary['id']}</code>
                        </div>
                        <div style='font-family: "Newsreader", serif; font-size: 1.1rem; font-weight: 600; color: var(--forest-ink);'>{primary['title']}</div>
                        <div style='font-size: 0.76rem; color: var(--text-secondary); margin-top: 0.25rem;'>Category: <b>{primary.get('category', 'N/A')}</b> • Rule Priority: <b>{primary.get('priority', 0)}</b></div>
                        <div style='margin-top: 0.75rem; padding: 0.75rem 0.9rem; background: var(--mist-surface); border-radius: var(--radius-md); font-size: 0.82rem; color: var(--forest-ink); border-left: 3px solid var(--canopy-green); line-height: 1.45;'>
                            {primary.get('guidance', 'N/A')}
                        </div>
                        {precautions_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                if len(matched) > 1:
                    st.markdown(f"<div style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.4rem;'>Compound Risks ({len(matched)-1} Secondary SOPs):</div>", unsafe_allow_html=True)
                    for s in matched[1:]:
                        st.markdown(
                            f"""
                            <div style='font-size: 0.80rem; padding: 0.45rem 0.65rem; background: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-sm); margin-bottom: 0.35rem;'>
                                <code>{s['id']}</code> <b>{s['title']}</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    f"""
                    <div style='background: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-lg); padding: 1.15rem; box-shadow: var(--shadow-subtle);'>
                        <div style='font-family: "Newsreader", serif; font-size: 1.05rem; font-weight: 600; color: var(--semantic-info);'>Governing Policy: <code>{citation}</code></div>
                        <div style='font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.4rem; line-height: 1.45;'>
                            No formal clinical SOP triggered a restriction under these environmental readings. Standard civic precautions apply.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Policy compliance evaluation will display here upon consultation execution.")

    with tab_retrieval:
        if last_state is None:
            st.markdown(
                """
                <div style='background-color: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-md); padding: 1.25rem 1.15rem; text-align: center; box-shadow: var(--shadow-subtle);'>
                    <div style='font-size: 1.6rem; margin-bottom: 0.35rem;'>🧭</div>
                    <div style='font-family: "Newsreader", serif; font-size: 1.05rem; font-weight: 700; color: var(--forest-ink);'>
                        Semantic Vector Retrieval Standby
                    </div>
                    <div style='font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; line-height: 1.45;'>
                        Submit an operational assessment query to initiate semantic embedding similarity search against the clinical Standard Operating Procedures catalog.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif last_state.get("retrieval_error"):
            err_msg = last_state.get("retrieval_error")
            st.markdown(
                f"""
                <div style='background-color: var(--semantic-caution-subtle); border-left: 4px solid var(--semantic-caution); border-radius: var(--radius-md); padding: 1rem 1.15rem; margin-bottom: 0.85rem;'>
                    <div style='font-family: "Newsreader", serif; font-size: 1rem; font-weight: 700; color: var(--semantic-caution);'>
                        ⚠️ Semantic Retrieval Service Unavailable
                    </div>
                    <div style='font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.25rem;'>
                        The vector similarity index encountered an issue: <code>{html_lib.escape(str(err_msg))}</code>
                    </div>
                    <div style='font-size: 0.72rem; color: var(--text-muted); margin-top: 0.35rem;'>
                        The consultation pipeline safely engaged deterministic policy rules.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            sops_retrieved = last_state.get("retrieved_sops") or []
            if not sops_retrieved:
                st.markdown(
                    """
                    <div style='background-color: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-md); padding: 1.1rem 1.15rem; box-shadow: var(--shadow-subtle);'>
                        <div style='font-family: "Newsreader", serif; font-size: 1rem; font-weight: 700; color: var(--forest-ink);'>
                            No Matching Clinical SOPs Found
                        </div>
                        <div style='font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.25rem;'>
                            Vector similarity search did not identify any Standard Operating Procedures covering the specific activity or criteria in this query.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                active_sop = last_state.get("active_sop") or {}
                active_sop_id = active_sop.get("id") or last_state.get("sop_citation")
                matched_ids = [
                    m.get("id") for m in (last_state.get("matched_sops") or [])
                    if isinstance(m, dict) and m.get("id")
                ]

                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.65rem;'>
                        <div style='font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--forest-ink);'>
                            Retrieved Candidates ({len(sops_retrieved)})
                        </div>
                        <div style='font-size: 0.70rem; color: var(--text-secondary);'>
                            ChromaDB Cosine Embeddings
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for s in sops_retrieved:
                    sid = s.get("id", "UNKNOWN")
                    title = s.get("title", "Untitled Policy")
                    cat = s.get("category", "General")
                    sev = s.get("severity", "LOW")
                    guidance = s.get("guidance", "")
                    activities = ", ".join(s.get("target_activities", [])) or "all"
                    raw_score = s.get("similarity_score")

                    is_governing = bool(active_sop_id and (sid == active_sop_id))
                    is_matched = bool(sid in matched_ids)

                    # Card style
                    card_cls = "fi-sop-card fi-sop-card-governing" if is_governing else "fi-sop-card"

                    # Status badge
                    if is_governing:
                        badge_html = "<span class='fi-sop-badge-governing'>● GOVERNING POLICY</span>"
                    elif is_matched:
                        badge_html = "<span class='fi-sop-badge-matched'>✓ CONDITION MATCHED</span>"
                    else:
                        badge_html = "<span class='fi-sop-badge-candidate'>CANDIDATE</span>"

                    # Severity badge
                    sev_badge_html = f"<span class='fi-risk-badge fi-risk-{sev.lower()}'>{sev}</span>"

                    # Score calculation
                    if raw_score is not None and isinstance(raw_score, (int, float)):
                        score_pct = raw_score * 100 if raw_score <= 1.0 else raw_score
                        score_str = f"{score_pct:.1f}%"
                        fill_w = max(5, min(100, int(score_pct)))
                    else:
                        score_str = "N/A"
                        fill_w = 0

                    guidance_snippet = guidance[:160] + ("..." if len(guidance) > 160 else "")

                    st.markdown(
                        f"""
                        <div class='{card_cls}'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;'>
                                <div style='display: flex; align-items: center; gap: 0.45rem;'>
                                    <code>{sid}</code>
                                    {badge_html}
                                </div>
                                <div>{sev_badge_html}</div>
                            </div>
                            <div style='font-size: 0.86rem; font-weight: 600; color: var(--forest-ink); margin-bottom: 0.25rem;'>
                                {title}
                            </div>
                            <div style='display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: var(--text-secondary); margin-top: 0.35rem;'>
                                <span>Vector Relevance Metric</span>
                                <span style='font-family: "JetBrains Mono", monospace; font-weight: 700; color: var(--forest-ink);'>{score_str}</span>
                            </div>
                            <div class='fi-relevance-bar-track'>
                                <div class='fi-relevance-bar-fill' style='width: {fill_w}%;'></div>
                            </div>
                            <div style='font-size: 0.74rem; color: var(--text-secondary); margin-top: 0.55rem; line-height: 1.4; border-top: 1px dotted var(--border-subtle); padding-top: 0.4rem;'>
                                {guidance_snippet}
                            </div>
                            <div style='font-size: 0.68rem; color: var(--text-muted); margin-top: 0.35rem;'>
                                Category: <b>{cat}</b> | Target Activities: <b>{activities}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_guardrails:
        if last_state:
            g_valid = last_state.get("grounding_valid", True)
            inp_valid = last_state.get("input_guardrail_passed", True)
            g_results = last_state.get("guardrail_results", [])

            inp_color = "var(--semantic-verified)" if inp_valid else "var(--semantic-critical)"
            g_color = "var(--semantic-verified)" if g_valid else "var(--semantic-caution)"

            st.markdown(
                f"""
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin-bottom: 0.85rem;'>
                    <div style='background: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-md); padding: 0.75rem; box-shadow: var(--shadow-subtle);'>
                        <div style='font-size: 0.68rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;'>Input Security</div>
                        <div style='font-size: 1.1rem; font-weight: 700; color: {inp_color}; margin-top: 0.15rem;'>{'PASSED' if inp_valid else 'BLOCKED'}</div>
                    </div>
                    <div style='background: var(--white-surface); border: 1px solid var(--border-warm); border-radius: var(--radius-md); padding: 0.75rem; box-shadow: var(--shadow-subtle);'>
                        <div style='font-size: 0.68rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;'>Output Grounding</div>
                        <div style='font-size: 1.1rem; font-weight: 700; color: {g_color}; margin-top: 0.15rem;'>{'GROUNDED' if g_valid else 'FALLBACK'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            if g_results:
                st.markdown("<div style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.35rem;'>Guardrail Verification Log:</div>", unsafe_allow_html=True)
                for gr in g_results:
                    st.code(f"[{gr.get('guardrail_type', 'guardrail').upper()}] Passed={gr.get('passed')} -> {gr.get('details')}", language="text")
        else:
            st.caption("Input security & output grounding audit logs will render here.")

    with tab_trace:
        if last_state:
            route = last_state.get("route", "N/A")
            st.markdown(
                f"""
                <div style='margin-bottom: 0.65rem;'>
                    <span style='font-size: 0.78rem; color: var(--text-secondary);'>Routing Branch:</span>
                    <code>{route}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            trace_items = last_state.get("execution_trace", [])
            if trace_items:
                st.markdown("<div style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.35rem;'>State Machine Transitions:</div>", unsafe_allow_html=True)
                for step in trace_items:
                    st.code(step, language="text")
        else:
            st.caption("LangGraph execution trace and state machine transitions will log here.")

# ==============================================================================
# LEFT PANEL: ADVISORY CONSULTATION CONSOLE
# ==============================================================================
with col_stream:
    st.markdown(
        """
        <div class='fi-editorial-banner'>
            <div class='fi-editorial-headline'>Environmental Safety Consultation Workspace</div>
            <div class='fi-editorial-desc'>
                Evidence-grounded meteorological advisory strictly calibrated against verified clinical Standard Operating Procedures.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Render Conversation / Assessment History
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class='fi-user-inquiry-card'>
                    <div class='fi-user-inquiry-meta'>Operational Assessment Request</div>
                    <div class='fi-user-inquiry-text'>{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_assistant_advisory(msg)

    # Empty State: Scenario Starters
    if len(st.session_state.chat_history) == 0:
        st.markdown(
            """
            <div class='fi-profiles-header'>Select an Operational Assessment Scenario</div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class='fi-scenario-card'>
                    <div class='fi-scenario-num'>01 // COMMUTER TRANSIT</div>
                    <div class='fi-scenario-title'>Bhopal Cycling Safety</div>
                    <div class='fi-scenario-detail'>Rain accumulation & high wind velocity limits for morning bicycle commutes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Evaluate Bhopal Cycle Query", key="btn_scen_1", use_container_width=True):
                st.session_state.sample_input = "Is it safe to cycle to work in Bhopal today?"
                st.rerun()

        with c2:
            st.markdown(
                """
                <div class='fi-scenario-card'>
                    <div class='fi-scenario-num'>02 // OUTDOOR LEISURE</div>
                    <div class='fi-scenario-title'>Kolkata Picnic Comfort</div>
                    <div class='fi-scenario-detail'>Thermal index, apparent temperature, and solar UV radiation thresholds.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Evaluate Kolkata Picnic Query", key="btn_scen_2", use_container_width=True):
                st.session_state.sample_input = "Can we plan an outdoor picnic in Kolkata this afternoon?"
                st.rerun()

        with c3:
            st.markdown(
                """
                <div class='fi-scenario-card'>
                    <div class='fi-scenario-num'>03 // PEDIATRIC PLAYGROUND</div>
                    <div class='fi-scenario-title'>Jaipur Toddler Outing</div>
                    <div class='fi-scenario-detail'>Surface playground heat and extreme solar radiation vulnerability checks.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Evaluate Jaipur Toddler Query", key="btn_scen_3", use_container_width=True):
                st.session_state.sample_input = "Is it safe to take my 3-year-old toddler to the playground in Jaipur today?"
                st.rerun()

    default_text = st.session_state.pop("sample_input", None)
    prompt = st.chat_input("Enter activity and location (e.g., 'Is it safe to cycle in Bhopal today?')") or default_text

    # Location Disambiguation Panel
    last_candidates = (
        st.session_state.last_state.get("location_candidates", [])
        if st.session_state.last_state
        else []
    )
    if len(last_candidates) > 1 and st.session_state.get("show_disambiguation"):
        st.markdown(
            f"""
            <div style='background-color: var(--semantic-caution-subtle); border-left: 4px solid var(--semantic-caution); border-radius: var(--radius-md); padding: 0.95rem 1.15rem; margin: 0.85rem 0;'>
                <div style='font-family: "Newsreader", serif; font-size: 1.05rem; font-weight: 700; color: var(--semantic-caution);'>Geographic Jurisdiction Disambiguation Required</div>
                <div style='font-size: 0.80rem; color: var(--text-secondary); margin-top: 0.2rem;'>
                    Multiple municipal entities match <b>{last_candidates[0]['name']}</b>. Select the target jurisdiction to ensure accurate telemetry:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        candidate_labels = [
            f"{i+1}. {c['display_name']} (pop. {c.get('population') or 'N/A'})"
            for i, c in enumerate(last_candidates)
        ]
        chosen_label = st.radio(
            "Select Target Jurisdiction:",
            candidate_labels,
            key="location_disambig_radio",
        )
        chosen_idx = candidate_labels.index(chosen_label)
        if st.button("Confirm Jurisdiction & Re-run", type="primary"):
            chosen = last_candidates[chosen_idx]
            last_query = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
            clarified_query = f"{last_query} (city: {chosen['display_name']})"
            st.session_state.show_disambiguation = False
            st.session_state.pending_disambiguation_query = clarified_query
            st.rerun()

    if st.session_state.get("pending_disambiguation_query"):
        prompt = st.session_state.pop("pending_disambiguation_query")

    if prompt:
        try:
            allowed, wait_secs = check_rate_limit(st.session_state.session_id, record=True)
        except TypeError:
            allowed, wait_secs = check_rate_limit(st.session_state.session_id)
            if allowed:
                record_query(st.session_state.session_id)

        if not allowed:
            st.warning(
                f"Rate limit reached ({get_usage_stats(st.session_state.session_id)['limit']} queries/min). "
                f"Please wait {wait_secs}s before submitting another query."
            )
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.markdown(
                f"""
                <div class='fi-user-inquiry-card'>
                    <div class='fi-user-inquiry-meta'>Operational Assessment Request</div>
                    <div class='fi-user-inquiry-text'>{prompt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.chat_message("assistant", avatar="🛡️"):
                status_container = st.status(
                    "Calibrating environmental safety consultation pipeline...",
                    expanded=True,
                )
                provisional_placeholder = st.empty()
                advisory_placeholder = st.empty()

                final_result = None
                has_error = False

                api_url = st.session_state.get("custom_api_url") or os.getenv("CLINIGUARD_API_URL")
                api_key = st.session_state.get("access_key") or os.getenv("ACCESS_KEY")

                for event in stream_consultation_events(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    api_url=api_url,
                    api_key=api_key,
                ):
                    event_type = event.get("event")
                    event_data = event.get("data", {})

                    if event_type == "status":
                        msg = event_data.get("message", "Processing stage...")
                        status_container.update(label=msg, state="running")
                        with status_container:
                            st.write(msg)

                    elif event_type == "provisional":
                        prov_text = event_data.get("text", "")
                        if prov_text:
                            provisional_placeholder.markdown(
                                f"""<div class='fi-provisional-banner'>
                                <div class='fi-provisional-title'>⚠️ Provisional Assessment Draft (Unverified)</div>
                                <div class='fi-provisional-sub'>Clinical safety evaluation & output grounding validation in progress...</div>
                                <div class='fi-provisional-body'>{prov_text}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                    elif event_type == "result":
                        final_result = event_data
                        provisional_placeholder.empty()

                        final_text = (
                            event_data.get("final_response")
                            or event_data.get("advisory")
                            or "No advisory generated."
                        )

                        if event_data.get("requires_clarification"):
                            status_container.update(
                                label="Clarification required to ensure clinical safety",
                                state="complete",
                                expanded=False,
                            )
                        elif event_data.get("route") == "deterministic_fallback":
                            status_container.update(
                                label="Safety fallback triggered — deterministic clinical guidance active",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            status_container.update(
                                label="Consultation verified & grounded against clinical SOPs",
                                state="complete",
                                expanded=False,
                            )

                        advisory_placeholder.markdown(final_text)

                    elif event_type == "error":
                        has_error = True
                        provisional_placeholder.empty()
                        err_text = event_data.get("error", "An unexpected pipeline error occurred.")
                        status_container.update(
                            label="Consultation pipeline interrupted",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"CliniGuard Safety Notice: {err_text}")

                if final_result and not has_error:
                    candidates = final_result.get("location_candidates", [])
                    st.session_state.show_disambiguation = bool(len(candidates) > 1)

                    st.session_state.last_state = final_result
                    load_session_state(st.session_state.session_id)

                    # Ensure last_state maintains retrieved_sops and retrieval_error from final_result
                    if not st.session_state.last_state:
                        st.session_state.last_state = final_result
                    else:
                        if "retrieved_sops" in final_result:
                            st.session_state.last_state["retrieved_sops"] = final_result.get("retrieved_sops") or []
                        if "retrieval_error" in final_result:
                            st.session_state.last_state["retrieval_error"] = final_result.get("retrieval_error")
                        if "active_sop" in final_result:
                            st.session_state.last_state["active_sop"] = final_result.get("active_sop")
                        if "matched_sops" in final_result:
                            st.session_state.last_state["matched_sops"] = final_result.get("matched_sops") or []

                    if not st.session_state.chat_history or st.session_state.chat_history[-1].get("role") != "assistant":
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": final_result.get("final_response") or final_result.get("advisory", ""),
                            "weather_data": final_result.get("weather") or final_result.get("weather_data"),
                            "sop_citation": final_result.get("sop_citation"),
                            "matched_sops": final_result.get("matched_sops", []),
                            "active_sop": final_result.get("active_sop"),
                            "retrieved_sops": final_result.get("retrieved_sops", []),
                            "retrieval_error": final_result.get("retrieval_error"),
                            "route": final_result.get("route"),
                            "execution_trace": final_result.get("execution_trace", []),
                        })

                    time.sleep(0.3)
                    st.rerun()
