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
from src.graph import run_agent_turn
from src.sop_engine import SOPEngine, SOPS_FILE_PATH
from src.weather import get_weather_description
from src.db import (
    list_all_sessions,
    get_session_messages,
    delete_session,
    create_or_update_session,
    clear_all_sessions,
)

# Page configuration
st.set_page_config(
    page_title="CliniGuard | Clinical Weather-Advisory System",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise Clinical CSS Design System
st.markdown(
    """
    <style>
    /* Font & Base Styles */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header styling */
    .clinical-header-container {
        padding: 0.8rem 1.2rem;
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border-radius: 8px;
        border-left: 5px solid #0EA5E9;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .clinical-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .clinical-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    .system-status-badge {
        background: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Telemetry KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        transition: border-color 0.2s;
    }
    .kpi-card:hover {
        border-color: #0EA5E9;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: #64748B;
    }
    
    /* SOP Severity Badges */
    .badge-critical {
        background-color: #DC2626;
        color: white;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-high {
        background-color: #EA580C;
        color: white;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-moderate {
        background-color: #D97706;
        color: white;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-low {
        background-color: #059669;
        color: white;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-info {
        background-color: #2563EB;
        color: white;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    
    /* Structured Markdown styling inside chat */
    .stMarkdown table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 0.8rem 0 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }
    .stMarkdown th {
        background-color: #0F172A !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.85rem !important;
        border-bottom: 2px solid #334155 !important;
    }
    .stMarkdown td {
        padding: 0.45rem 0.75rem !important;
        font-size: 0.85rem !important;
        border-bottom: 1px solid #1E293B !important;
    }
    
    /* Session card in sidebar */
    .session-item-active {
        background: #0F172A;
        border-left: 3px solid #0EA5E9;
        padding: 0.5rem 0.7rem;
        border-radius: 4px;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper function to synchronize state from persistent database
def load_session_state(sid: str):
    msgs = get_session_messages(sid)
    st.session_state.chat_history = msgs
    last_assistant = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last_assistant and last_assistant.get("weather_data"):
        st.session_state.last_state = {
            "weather_data": last_assistant.get("weather_data"),
            "sop_citation": last_assistant.get("sop_citation"),
            "matched_sops": last_assistant.get("matched_sops", []),
            "active_sop": last_assistant.get("active_sop"),
            "route": last_assistant.get("route"),
            "execution_trace": last_assistant.get("execution_trace", []),
            "grounding_valid": True,
        }
    else:
        st.session_state.last_state = None


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

sop_engine = SOPEngine()

# ==========================================
# SIDEBAR: PERSISTENCE & POLICY MANAGEMENT
# ==========================================
with st.sidebar:
    st.markdown("### 🗄️ Consultation Sessions")
    st.caption("Persistent SQLite consultation history")

    # Action: Create New Session
    if st.button("➕ New Consultation", use_container_width=True, type="primary"):
        new_sid = f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.last_state = None
        st.rerun()

    # Re-fetch latest sessions from SQLite to ensure titles are fresh
    saved_sessions = list_all_sessions()
    session_options = {s["session_id"]: s["title"] for s in saved_sessions}
    
    current_sid = st.session_state.session_id
    if current_sid not in session_options:
        session_options[current_sid] = "New Consultation"

    options_keys = list(session_options.keys())
    current_index = options_keys.index(current_sid) if current_sid in options_keys else 0

    selected_sid = st.selectbox(
        "Select Consultation",
        options=options_keys,
        format_func=lambda sid: session_options.get(sid, sid),
        index=current_index,
        label_visibility="collapsed",
    )

    if selected_sid != st.session_state.session_id:
        st.session_state.session_id = selected_sid
        load_session_state(selected_sid)
        st.rerun()

    # Delete & Clear Actions
    col_del1, col_del2 = st.columns(2)
    if col_del1.button("🗑️ Delete Session", use_container_width=True):
        delete_session(st.session_state.session_id)
        remaining = list_all_sessions()
        if remaining:
            new_sid = remaining[0]["session_id"]
        else:
            new_sid = f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        load_session_state(new_sid)
        st.rerun()
        
    if col_del2.button("🧹 Clear All", use_container_width=True):
        clear_all_sessions()
        new_sid = f"session-{int(time.time())}"
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.last_state = None
        st.rerun()

    st.divider()

    # Active Policy Catalog
    st.markdown("### 📚 Clinical SOP Catalog")
    sops = sop_engine.load_sops(force_reload=True)
    st.caption(f"**{len(sops)} active Standard Operating Procedures** loaded")

    with st.expander("🔍 View All Clinical Policies", expanded=False):
        for s in sops:
            sev = s.get("severity", "LOW")
            badge_class = f"badge-{sev.lower()}"
            st.markdown(
                f"<span class='{badge_class}'>{sev}</span> **{s['id']}**<br><small>{s['title']}</small>",
                unsafe_allow_html=True,
            )
            st.caption(f"Activities: `{', '.join(s.get('target_activities', []))}`")
            st.text(f"Guidance: {s.get('guidance')[:85]}...")
            st.divider()

    # Dynamic 11th SOP Creator
    st.markdown("### ➕ Zero-Code Policy Creator")
    st.caption("Add an authorized SOP in real-time without Python code changes.")
    
    with st.expander("📝 Author New Policy"):
        new_id = st.text_input("Policy ID", value=f"SOP-CUSTOM-{len(sops)+1:03d}")
        new_title = st.text_input("Title", value="Extreme Heat Tennis & Racket Sports Policy")
        new_category = st.selectbox("Category", ["outdoor_exercise", "travel_commute", "vulnerable_groups", "outdoor_leisure"])
        new_activity = st.text_input("Target Activities (comma-separated)", value="tennis, badminton, pickleball")
        new_severity = st.selectbox("Severity Level", ["CRITICAL", "HIGH", "MODERATE", "LOW"], index=1)
        new_temp = st.number_input("Min Temperature Trigger (°C)", value=36.0, step=1.0)
        new_guidance = st.text_area("Authorized Guidance", value="High intensity continuous court movement under extreme thermal load carries severe risk of exertional heat cramps.")
        new_precautions = st.text_area("Mandatory Precautions (1 per line)", value="Mandatory 10-min hydration break every set\nWear UV visor and polarized sunglasses\nAvoid court surfaces between 12 PM - 3 PM")
        
        if st.button("💾 Append SOP to sops.json", use_container_width=True, type="primary"):
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
            st.success(f"Added {new_id}! Reloading policy engine...")
            time.sleep(1)
            st.rerun()

# ==========================================
# MAIN VIEW: CLINICAL HEADER
# ==========================================
st.markdown(
    f"""
    <div class='clinical-header-container'>
        <div>
            <div class='clinical-title'>🛡️ CliniGuard | Clinical Weather-Advisory System</div>
            <div class='clinical-subtitle'>Safety-critical outdoor activity guidance strictly grounded in authorized SOPs & live Open-Meteo meteorological telemetry.</div>
        </div>
        <div style='text-align: right;'>
            <div class='system-status-badge'>🟢 SYSTEM ONLINE & PERSISTENT</div>
            <div style='font-size: 0.72rem; color: #64748B; margin-top: 0.3rem;'>Session: <code>{st.session_state.session_id}</code></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout: 62% Chat & Advisory Stream | 38% Telemetry & Decision Inspector
col_stream, col_inspector = st.columns([13, 8])

# ==========================================
# RIGHT PANEL: TELEMETRY & DECISION CENTER
# ==========================================
with col_inspector:
    tab_telemetry, tab_policy, tab_trace = st.tabs(["📡 Live Telemetry", "📋 Governing Policy", "🧠 Graph Trace"])
    
    last_state = st.session_state.last_state
    weather = last_state.get("weather_data") if last_state else None
    
    with tab_telemetry:
        if weather:
            loc = weather.get("location_name") or last_state.get("location") or "Resolved Target"
            temp = weather.get("temperature_2m", "N/A")
            app_temp = weather.get("apparent_temperature", temp)
            wind = weather.get("wind_speed_10m", "N/A")
            gusts = weather.get("wind_gusts_10m", wind)
            rain_prob = weather.get("precipitation_probability", "0")
            rain = weather.get("precipitation", "0.0")
            uv = weather.get("uv_index", "0.0")
            humidity = weather.get("relative_humidity_2m", "N/A")
            cond = weather.get("weather_description", "Unknown")

            st.markdown(
                f"""
                <div style='background: #0F172A; border-radius: 8px; padding: 0.85rem; border: 1px solid #1E293B; margin-bottom: 0.8rem;'>
                    <div style='font-size: 1.05rem; font-weight: 700; color: #38BDF8;'>📍 {loc}</div>
                    <div style='font-size: 0.85rem; color: #94A3B8; margin-top: 0.1rem;'>Current Condition: <b style='color: #F8FAFC;'>{cond}</b></div>
                </div>
                
                <div class='kpi-grid'>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Temperature</div>
                        <div class='kpi-value'>{temp}°C</div>
                        <div class='kpi-sub'>Apparent: {app_temp}°C</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Rainfall Probability</div>
                        <div class='kpi-value'>{rain_prob}%</div>
                        <div class='kpi-sub'>Accumulation: {rain} mm</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Wind Velocity</div>
                        <div class='kpi-value'>{wind} <span style='font-size: 0.8rem;'>km/h</span></div>
                        <div class='kpi-sub'>Max Gusts: {gusts} km/h</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-label'>UV Radiation</div>
                        <div class='kpi-value'>{uv}</div>
                        <div class='kpi-sub'>Humidity: {humidity}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Telemetry metrics will render once a location is evaluated.")

    with tab_policy:
        if last_state:
            citation = last_state.get("sop_citation", "None")
            matched = last_state.get("matched_sops", [])
            primary = last_state.get("active_sop") or (matched[0] if matched else None)
            
            if primary:
                sev = primary.get("severity", "LOW")
                badge_class = f"badge-{sev.lower()}"
                st.markdown(
                    f"<span class='{badge_class}'>{sev}</span> **{primary['id']}** — *{primary['title']}*",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Category:** `{primary.get('category', 'N/A')}` | **Priority:** `{primary.get('priority', 0)}`")
                st.markdown(f"**Policy Guidance:**\n> {primary.get('guidance', 'N/A')}")
                
                if len(matched) > 1:
                    st.divider()
                    st.markdown(f"**Compound Risks ({len(matched)-1} Secondary SOPs):**")
                    for s in matched[1:]:
                        st.markdown(f"- `{s['id']}` (*{s['title']}*): {s.get('guidance', '')[:80]}...")
            else:
                st.markdown(f"**Active Policy Status:** `{citation}`")
                if citation == "NO_SOP_APPLICABLE":
                    st.caption("No authorized clinical standard operating procedure matches these parameters.")
        else:
            st.caption("Policy compliance evaluation will display here.")

    with tab_trace:
        if last_state:
            route = last_state.get("route", "N/A")
            st.markdown(f"**Graph Routing Path:** `{route}`")
            st.markdown(f"**Guardrail Grounding Status:** `{'PASSED' if last_state.get('grounding_valid', True) else 'FAILED'}`")
            
            trace_items = last_state.get("execution_trace", [])
            if trace_items:
                st.markdown("**Execution Step Trace:**")
                for step in trace_items:
                    st.code(step, language="text")
        else:
            st.caption("LangGraph execution trace and state machine transitions will be logged here.")

# ==========================================
# LEFT PANEL: ADVISORY STREAM & CHAT
# ==========================================
with col_stream:
    st.markdown("### 💬 Clinical Advisory Stream")
    
    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])

    # Example Starter Chips
    if len(st.session_state.chat_history) == 0:
        st.caption("Select a sample consultation scenario to begin:")
        c1, c2, c3 = st.columns(3)
        if c1.button("🚴 Cycling in Bhopal", use_container_width=True):
            st.session_state.sample_input = "Is it safe to cycle to work in Bhopal today?"
        if c2.button("🧺 Picnic in Kolkata", use_container_width=True):
            st.session_state.sample_input = "Can we plan an outdoor picnic in Kolkata this afternoon?"
        if c3.button("🧒 Toddler Park in Jaipur", use_container_width=True):
            st.session_state.sample_input = "Is it safe to take my 3-year-old toddler to the playground in Jaipur today?"

    # Chat Input Box
    default_text = st.session_state.pop("sample_input", None)
    prompt = st.chat_input("Enter your activity & location (e.g., 'Is it safe to cycle in Bhopal today?')") or default_text

    if prompt:
        # Display User Input
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # Process through LangGraph Workflow
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Fetching live meteorological telemetry & evaluating clinical SOPs..."):
                response_dict = run_agent_turn(
                    user_query=prompt,
                    session_id=st.session_state.session_id,
                )
                final_text = response_dict.get("final_response", "No advisory generated.")
                st.markdown(final_text)
                
                # Synchronize full state from SQLite persistence
                st.session_state.last_state = response_dict
                load_session_state(st.session_state.session_id)
                st.rerun()
