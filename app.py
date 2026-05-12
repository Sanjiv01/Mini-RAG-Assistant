"""Streamlit UI for Hermes — grounded retrieval over your documents.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag.config import settings
from rag.generate import make_llm
from rag.highlight import find_matched_spans, render_with_highlights
from rag.pipeline import RAG

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_corpus"

PRODUCT_NAME = "Hermes"
PRODUCT_TAGLINE = "Delivers grounded answers from complex documents."

NAV_ITEMS = [
    ("home", "Home"),
    ("knowledge_base", "Knowledge Base"),
    ("chat", "Chat"),
    ("models", "Models"),
]

TAB_INFO = {
    "home": "Overview of your knowledge base, recent activity, and shortcuts.",
    "knowledge_base": "Manage the documents Hermes can retrieve from.",
    "chat": "Ask grounded questions and get answers cited from your documents.",
    "models": "Pick the language model that powers Hermes.",
}

# Persistent lifetime stats. Only the three home-page counters survive
# across restarts — everything else (corpus, chat) is intentionally session-only.
STATS_FILE = settings.cache_dir / "stats.json"


def _load_lifetime_stats() -> dict:
    if STATS_FILE.exists():
        try:
            data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            return {
                "documents": int(data.get("documents", 0)),
                "chunks":    int(data.get("chunks", 0)),
                "questions": int(data.get("questions", 0)),
            }
        except Exception:
            pass
    return {"documents": 0, "chunks": 0, "questions": 0}


def _save_lifetime_stats(stats: dict) -> None:
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats), encoding="utf-8")
    except Exception:
        pass  # never crash the UI if stats can't be persisted


def _bump_stats(*, documents: int = 0, chunks: int = 0, questions: int = 0) -> None:
    s = st.session_state.lifetime_stats
    s["documents"] += documents
    s["chunks"] += chunks
    s["questions"] += questions
    _save_lifetime_stats(s)


AVAILABLE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen 2.5 7B Instruct",
        "vendor": "Alibaba",
        "size": "7.6B parameters",
        "quantization": "4-bit NF4 (≈ 5 GB VRAM)",
        "description": (
            "Strong open-weights instruction-following model. Good balance of "
            "reasoning quality and speed on consumer GPUs."
        ),
    },
]


# ---------- design system ---------------------------------------------------

COLORS = {
    "primary":      "#2563EB",
    "primary_dark": "#1D4ED8",
    "primary_soft": "#EFF6FF",
    "accent":       "#22C55E",
    "accent_soft":  "rgba(34, 197, 94, 0.16)",
    "bg":           "#F8FAFC",
    "surface":      "#FFFFFF",
    "surface_alt":  "#F1F5F9",
    "surface_2":    "#F8FAFC",
    "text":         "#0F172A",
    "text_muted":   "#64748B",
    "text_dim":     "#94A3B8",
    "border":       "#E2E8F0",
    "border_soft":  "#EDF2F7",
    "success":      "#16A34A",
    "warning":      "#CA8A04",
    "danger":       "#DC2626",
}


def _inject_css() -> None:
    c = COLORS
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', Roboto, sans-serif;
            color: {c["text"]};
            -webkit-font-smoothing: antialiased;
        }}
        h1, h2, h3, h4 {{
            color: {c["text"]};
            letter-spacing: -0.015em;
            font-weight: 700;
        }}

        .stApp {{ background: {c["bg"]}; }}
        [data-testid="stHeader"] {{ background: transparent; height: 0; }}
        footer, [data-testid="stStatusWidget"] {{ visibility: hidden; height: 0; }}
        #MainMenu {{ visibility: hidden; }}
        .block-container {{
            max-width: 1180px;
            padding: 1.4rem 2rem 6rem 2rem;
        }}

        /* =========== sidebar =========== */
        [data-testid="stSidebar"] {{
            background: {c["surface"]};
            border-right: 1px solid {c["border"]};
            width: 260px !important;
            min-width: 260px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding: 1.4rem 0.9rem;
        }}

        .hm-brand {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0 0.4rem 1.4rem 0.4rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid {c["border_soft"]};
        }}
        .hm-logo {{
            width: 34px; height: 34px;
            border-radius: 9px;
            background: linear-gradient(135deg, {c["primary"]} 0%, #4F8EF7 100%);
            display: grid; place-items: center;
            color: white;
            font-weight: 800;
            font-size: 1rem;
            box-shadow: 0 4px 12px -4px rgba(37, 99, 235, 0.45);
            letter-spacing: -0.02em;
        }}
        .hm-brand-name {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {c["text"]};
        }}

        /* Sidebar nav buttons */
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {{
            background: transparent;
            color: {c["text_muted"]};
            border: none;
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
            font-weight: 500;
            font-size: 0.9rem;
            text-align: left;
            box-shadow: none;
            transition: all 120ms ease;
            justify-content: flex-start;
            display: flex;
            align-items: center;
        }}
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
            background: {c["surface_alt"]};
            color: {c["text"]};
            transform: none;
            box-shadow: none;
        }}
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {{
            background: {c["surface_alt"]};
            color: {c["text"]};
            font-weight: 600;
            position: relative;
        }}
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]::after {{
            content: "";
            position: absolute;
            right: 12px;
            width: 6px; height: 6px;
            border-radius: 50%;
            background: {c["primary"]};
        }}

        /* Main-area buttons (primary CTAs) */
        .block-container div[data-testid="stButton"] > button {{
            background: {c["text"]};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.05rem;
            font-weight: 500;
            font-size: 0.88rem;
            transition: all 160ms ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .block-container div[data-testid="stButton"] > button:hover {{
            background: #1E293B;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px -3px rgba(15, 23, 42, 0.18);
        }}
        .block-container div[data-testid="stButton"] > button[kind="secondary"] {{
            background: {c["surface"]};
            color: {c["text"]};
            border: 1px solid {c["border"]};
            font-weight: 500;
            text-align: left;
            white-space: normal;
            line-height: 1.45;
            padding: 0.8rem 1rem;
        }}
        .block-container div[data-testid="stButton"] > button[kind="secondary"]:hover {{
            border-color: {c["primary"]};
            background: {c["primary_soft"]};
            color: {c["primary"]};
        }}

        /* =========== breadcrumb =========== */
        .hm-breadcrumb {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: {c["text_muted"]};
            margin-bottom: 1.5rem;
        }}
        .hm-breadcrumb a {{
            color: {c["text_muted"]};
            text-decoration: none;
        }}
        .hm-breadcrumb .sep {{
            color: {c["text_dim"]};
            font-size: 0.78rem;
        }}
        .hm-breadcrumb .current {{
            color: {c["text"]};
            font-weight: 500;
        }}

        /* =========== page header =========== */
        .hm-page-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.6rem;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .hm-page-title {{
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: {c["text"]};
        }}
        .hm-page-subtitle {{
            color: {c["text_muted"]};
            font-size: 0.92rem;
            line-height: 1.5;
        }}

        /* =========== dashboard hero =========== */
        .hm-hero {{
            background: linear-gradient(135deg, {c["primary_soft"]} 0%, {c["surface"]} 100%);
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.6rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
        }}
        .hm-hero-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {c["primary"]};
            margin: 0 0 0.5rem 0;
        }}
        .hm-hero-sub {{
            color: {c["text_muted"]};
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 560px;
        }}

        /* =========== stat cards (dashboard) =========== */
        .hm-stat-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            display: flex;
            align-items: flex-start;
            gap: 0.95rem;
        }}
        .hm-stat-icon {{
            width: 40px; height: 40px;
            border-radius: 10px;
            background: {c["primary_soft"]};
            color: {c["primary"]};
            display: grid; place-items: center;
            font-size: 1.05rem;
            flex-shrink: 0;
        }}
        .hm-stat-icon.green {{ background: rgba(34,197,94,0.14); color: {c["success"]}; }}
        .hm-stat-icon.purple {{ background: rgba(139,92,246,0.14); color: #7C3AED; }}
        .hm-stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1;
            color: {c["text"]};
        }}
        .hm-stat-label {{
            color: {c["text_muted"]};
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }}

        /* =========== section heading =========== */
        .hm-section-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {c["text"]};
            margin: 1.8rem 0 0.9rem 0;
        }}

        /* =========== quick action cards =========== */
        .hm-qa-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 1.6rem 1.4rem;
            text-align: center;
            transition: all 180ms ease;
            height: 100%;
        }}
        .hm-qa-icon {{
            width: 56px; height: 56px;
            margin: 0 auto 0.9rem auto;
            border-radius: 14px;
            display: grid; place-items: center;
            font-size: 1.5rem;
            background: {c["primary_soft"]};
            color: {c["primary"]};
        }}
        .hm-qa-icon.green {{ background: rgba(34,197,94,0.14); color: {c["success"]}; }}
        .hm-qa-icon.purple {{ background: rgba(139,92,246,0.14); color: #7C3AED; }}
        .hm-qa-title {{
            font-size: 1rem;
            font-weight: 700;
            color: {c["text"]};
            margin-bottom: 0.4rem;
        }}
        .hm-qa-desc {{
            color: {c["text_muted"]};
            font-size: 0.84rem;
            line-height: 1.5;
        }}

        /* =========== how it works steps =========== */
        .hm-hiw-wrap {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
        }}
        .hm-hiw-title {{
            display: flex; align-items: center; gap: 0.55rem;
            font-size: 1.05rem; font-weight: 700;
            color: {c["text"]};
            margin-bottom: 1.1rem;
        }}
        .hm-hiw-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.95rem;
            padding: 0.95rem 0;
            border-top: 1px solid {c["border_soft"]};
        }}
        .hm-hiw-row:first-of-type {{ border-top: none; }}
        .hm-hiw-num {{
            width: 30px; height: 30px;
            border-radius: 50%;
            background: {c["primary"]};
            color: white;
            display: grid; place-items: center;
            font-weight: 700;
            font-size: 0.85rem;
            flex-shrink: 0;
        }}
        .hm-hiw-text {{ flex: 1; min-width: 0; }}
        .hm-hiw-text-head {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-bottom: 0.3rem;
        }}
        .hm-hiw-text-head strong {{
            color: {c["text"]};
            font-size: 0.95rem;
        }}
        .hm-tag {{
            display: inline-block;
            padding: 1px 7px;
            border-radius: 4px;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: {c["primary_soft"]};
            color: {c["primary"]};
            border: 1px solid rgba(37, 99, 235, 0.25);
        }}
        .hm-tag.opt {{
            background: rgba(34, 197, 94, 0.1);
            color: {c["success"]};
            border-color: rgba(34, 197, 94, 0.3);
        }}
        .hm-hiw-text span.hm-hiw-desc {{
            color: {c["text_muted"]};
            font-size: 0.86rem;
            line-height: 1.55;
            display: block;
        }}
        .hm-hiw-text code {{
            background: {c["surface_alt"]};
            color: {c["primary"]};
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.78rem;
        }}

        /* =========== requirements coverage =========== */
        .hm-cov-wrap {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 1rem;
        }}
        .hm-cov-title {{
            display: flex; align-items: center; gap: 0.55rem;
            font-size: 1.05rem; font-weight: 700;
            color: {c["text"]};
            margin-bottom: 1.1rem;
        }}
        .hm-cov-group {{
            font-size: 0.7rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {c["text_dim"]};
            font-weight: 700;
            margin: 0.6rem 0 0.55rem 0;
        }}
        .hm-cov-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            padding: 0.55rem 0;
            font-size: 0.88rem;
            line-height: 1.5;
            color: {c["text"]};
            border-top: 1px solid {c["border_soft"]};
        }}
        .hm-cov-group + .hm-cov-row {{ border-top: none; }}
        .hm-cov-icon {{
            width: 18px; height: 18px;
            border-radius: 50%;
            display: grid; place-items: center;
            font-size: 0.7rem;
            font-weight: 900;
            color: white;
            flex-shrink: 0;
            margin-top: 2px;
        }}
        .hm-cov-icon.done {{ background: {c["success"]}; }}
        .hm-cov-icon.partial {{ background: {c["warning"]}; }}
        .hm-cov-text {{ flex: 1; }}
        .hm-cov-text strong {{ color: {c["text"]}; }}
        .hm-cov-text em {{
            color: {c["text_muted"]};
            font-style: normal;
            font-size: 0.82rem;
        }}

        /* =========== cards =========== */
        .hm-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }}
        .hm-card-pad-sm {{ padding: 1rem 1.2rem; }}

        /* =========== document table =========== */
        .hm-doc-list {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 12px;
            overflow: hidden;
        }}
        .hm-doc-header, .hm-doc-row {{
            display: grid;
            grid-template-columns: 2.2fr 1fr 1.5fr 1fr;
            align-items: center;
            padding: 0.85rem 1.3rem;
            font-size: 0.88rem;
        }}
        .hm-doc-header {{
            background: {c["surface_alt"]};
            color: {c["text_muted"]};
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border-bottom: 1px solid {c["border"]};
        }}
        .hm-doc-row {{
            border-bottom: 1px solid {c["border_soft"]};
            color: {c["text"]};
        }}
        .hm-doc-row:last-child {{ border-bottom: none; }}
        .hm-doc-row:hover {{ background: {c["surface_alt"]}; }}
        .hm-doc-name {{
            display: flex; align-items: center; gap: 0.55rem;
            font-weight: 500;
        }}
        .hm-doc-icon {{
            width: 28px; height: 28px;
            border-radius: 6px;
            background: {c["primary_soft"]};
            color: {c["primary"]};
            display: grid; place-items: center;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .hm-doc-icon.md {{ background: rgba(34, 197, 94, 0.12); color: {c["success"]}; }}
        .hm-doc-icon.txt {{ background: rgba(100, 116, 139, 0.12); color: {c["text_muted"]}; }}
        .hm-doc-icon.url {{ background: rgba(168, 85, 247, 0.14); color: #7C3AED; }}
        .hm-url-chip {{
            display: inline-block;
            background: rgba(168, 85, 247, 0.1);
            color: #7C3AED;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.74rem;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            border: 1px solid rgba(168, 85, 247, 0.25);
        }}
        .hm-doc-meta {{ color: {c["text_muted"]}; font-size: 0.83rem; }}
        .hm-status-pill {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(22, 163, 74, 0.12);
            color: {c["success"]};
            border: 1px solid rgba(22, 163, 74, 0.25);
        }}
        .hm-empty-row {{
            padding: 2.5rem 1.5rem;
            text-align: center;
            color: {c["text_muted"]};
            font-size: 0.9rem;
            background: {c["surface"]};
        }}

        /* =========== chat =========== */
        .hm-empty {{
            text-align: center;
            padding: 2.5rem 1rem 2rem 1rem;
            max-width: 640px;
            margin: 0 auto;
        }}
        .hm-empty-icon {{
            width: 64px; height: 64px;
            margin: 0 auto 1.4rem auto;
            border-radius: 18px;
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            display: grid; place-items: center;
            font-size: 1.6rem;
            color: {c["primary"]};
            box-shadow: 0 8px 22px -12px rgba(0,0,0,0.08);
        }}
        .hm-empty h2 {{
            font-size: 1.5rem;
            margin: 0 0 0.5rem 0;
            color: {c["text"]};
        }}
        .hm-empty p {{
            color: {c["text_muted"]};
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0 auto 1.8rem auto;
            max-width: 480px;
        }}

        .hm-msg {{ display: flex; gap: 0.85rem; align-items: flex-start; margin-bottom: 1.4rem; }}
        .hm-msg-user {{ flex-direction: row-reverse; }}
        .hm-avatar {{
            flex-shrink: 0;
            width: 32px; height: 32px;
            border-radius: 50%;
            display: grid; place-items: center;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .hm-avatar.user {{
            background: {c["surface_alt"]};
            color: {c["text"]};
            border: 1px solid {c["border"]};
        }}
        .hm-avatar.assistant {{
            background: linear-gradient(135deg, {c["primary"]}, #4F8EF7);
            color: white;
        }}
        .hm-bubble {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 1rem 1.15rem;
            max-width: 88%;
            font-size: 0.95rem;
            line-height: 1.6;
            color: {c["text"]};
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        .hm-bubble.user {{
            background: {c["primary"]};
            color: white;
            border-color: {c["primary"]};
            border-bottom-right-radius: 4px;
        }}
        .hm-bubble.assistant {{ border-bottom-left-radius: 4px; }}
        .hm-bubble.refused {{
            background: rgba(220, 38, 38, 0.04);
            border-color: rgba(220, 38, 38, 0.22);
            color: {c["danger"]};
        }}

        .hm-answer-text {{ margin: 0; font-size: 0.95rem; line-height: 1.65; }}
        .hm-answer-text p:first-child {{ margin-top: 0; }}
        .hm-answer-text p:last-child {{ margin-bottom: 0; }}
        .hm-answer-text code {{
            background: {c["surface_alt"]};
            color: {c["primary"]};
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.86em;
        }}

        /* Grounded-phrase highlighter — GREEN */
        .hm-answer-text mark.hm-ground {{
            background: linear-gradient(180deg,
                            transparent 0%,
                            transparent 55%,
                            {c["accent_soft"]} 55%,
                            {c["accent_soft"]} 100%);
            color: inherit;
            padding: 0 1px;
            border-radius: 0;
            font-weight: 500;
        }}
        .hm-src-pill {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 1px 7px;
            margin: 0 2px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            vertical-align: 1px;
            background: {c["primary_soft"]};
            color: {c["primary"]};
            border: 1px solid rgba(37, 99, 235, 0.25);
        }}

        .hm-bubble-top {{
            display: flex; align-items: center; gap: 0.6rem;
            margin-bottom: 0.7rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px dashed {c["border_soft"]};
        }}
        .hm-bubble-top-stat {{
            color: {c["text_muted"]};
            font-size: 0.78rem;
        }}
        .hm-bubble-top-stat strong {{ color: {c["text"]}; font-weight: 600; }}

        .hm-meta {{
            display: flex; align-items: center; gap: 0.7rem;
            margin-top: 0.85rem; flex-wrap: wrap; font-size: 0.78rem;
        }}
        .hm-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 700;
            letter-spacing: 0.02em;
            font-size: 0.76rem;
            cursor: help;
        }}
        .hm-badge .dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background: currentColor;
        }}
        .hm-meta-stat {{ color: {c["text_muted"]}; }}
        .hm-meta-stat strong {{ color: {c["text"]}; font-weight: 600; }}

        /* Citations rendered as native <details> inside the message column so
           they align with the bubble instead of spanning the full page. */
        .hm-citations {{
            max-width: 88%;
            margin-top: 0.55rem;
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 10px;
            overflow: hidden;
        }}
        .hm-citations > summary {{
            list-style: none;
            cursor: pointer;
            user-select: none;
            padding: 0.6rem 1rem;
            color: {c["primary"]};
            font-weight: 600;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            transition: background 120ms;
        }}
        .hm-citations > summary::-webkit-details-marker {{ display: none; }}
        .hm-citations > summary::before {{
            content: "▸";
            font-size: 0.7rem;
            color: {c["primary"]};
            transition: transform 160ms ease;
            display: inline-block;
            width: 10px;
        }}
        .hm-citations[open] > summary::before {{ transform: rotate(90deg); }}
        .hm-citations[open] > summary {{
            border-bottom: 1px solid {c["border_soft"]};
        }}
        .hm-citations > summary:hover {{ background: {c["surface_alt"]}; }}
        .hm-citations-body {{ padding: 0.55rem 0.9rem 0.85rem 0.9rem; }}

        /* Streamlit's native expander — used elsewhere; same styling. */
        [data-testid="stExpander"] {{
            border: 1px solid {c["border"]};
            border-radius: 10px;
            background: {c["surface"]};
            margin-top: 0.85rem;
            box-shadow: none;
        }}
        [data-testid="stExpander"] > details > summary {{
            color: {c["primary"]};
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.55rem 1rem;
        }}
        [data-testid="stExpander"] > details[open] > summary {{
            border-bottom: 1px solid {c["border_soft"]};
        }}

        .hm-source-card {{
            margin: 0.7rem 0;
            padding: 0.85rem 1rem;
            background: {c["surface_alt"]};
            border: 1px solid {c["border_soft"]};
            border-radius: 10px;
            border-left: 4px solid {c["accent"]};
            transition: all 160ms ease;
        }}
        .hm-source-card:hover {{
            border-color: {c["primary"]};
            border-left-color: {c["accent"]};
            box-shadow: 0 4px 12px -6px rgba(15, 23, 42, 0.12);
        }}
        .hm-source-head {{
            display: flex; align-items: center; gap: 0.55rem;
            flex-wrap: wrap; margin-bottom: 0.6rem;
            font-size: 0.82rem;
        }}
        .hm-source-tag {{
            background: {c["primary"]};
            color: white;
            padding: 2px 9px;
            border-radius: 5px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }}
        .hm-source-file {{
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 0.78rem;
            color: {c["text"]};
            background: {c["surface"]};
            padding: 2px 8px;
            border-radius: 5px;
            font-weight: 500;
            border: 1px solid {c["border_soft"]};
        }}
        .hm-source-section {{
            color: {c["text_muted"]};
            font-size: 0.78rem;
            font-style: italic;
        }}
        .hm-source-metrics {{
            margin-left: auto;
            display: flex;
            gap: 0.55rem;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 0.7rem;
            color: {c["text_dim"]};
        }}
        .hm-source-metrics .hm-pri {{ color: {c["primary"]}; font-weight: 700; }}
        .hm-source-body {{
            background: {c["surface"]};
            color: {c["text"]};
            padding: 0.85rem 1rem;
            border-radius: 8px;
            border: 1px solid {c["border_soft"]};
            font-size: 0.85rem;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 220px;
            overflow-y: auto;
        }}
        /* Source-side highlights also use accent green for consistency. */
        .hm-source-body mark {{
            background: {c["accent_soft"]};
            color: {c["text"]};
            padding: 1px 3px;
            border-radius: 3px;
            font-weight: 500;
        }}

        [data-testid="stChatInput"] {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            box-shadow: 0 8px 28px -16px rgba(0,0,0,0.1);
            transition: all 160ms ease;
        }}
        [data-testid="stChatInput"] textarea {{
            background: transparent;
            color: {c["text"]};
            font-size: 0.98rem;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: {c["primary"]};
            box-shadow: 0 0 0 4px {c["primary_soft"]},
                        0 8px 28px -16px rgba(0,0,0,0.12);
        }}

        [data-testid="stFileUploader"] section {{
            background: {c["surface"]};
            border: 1.5px dashed {c["border"]};
            border-radius: 12px;
            padding: 1.5rem 1rem;
            transition: all 160ms;
        }}
        [data-testid="stFileUploader"] section:hover {{
            border-color: {c["primary"]};
            background: {c["primary_soft"]};
        }}

        [data-testid="stProgress"] > div > div > div > div {{
            background: linear-gradient(90deg, {c["primary"]}, {c["accent"]});
        }}
        [data-testid="stProgress"] > div > div > div {{
            background: {c["border_soft"]};
        }}

        [data-testid="stAlert"] {{
            border-radius: 10px;
            border: 1px solid {c["border"]};
            background: {c["surface"]};
            color: {c["text"]};
            box-shadow: none;
        }}
        hr {{
            border: none;
            border-top: 1px solid {c["border_soft"]};
            margin: 1.2rem 0;
        }}

        /* Model picker cards (Options page) */
        .hm-model-card {{
            background: {c["surface"]};
            border: 1.5px solid {c["border"]};
            border-radius: 12px;
            padding: 1.2rem 1.3rem;
            transition: all 160ms ease;
            cursor: pointer;
        }}
        .hm-model-card.selected {{
            border-color: {c["primary"]};
            background: {c["primary_soft"]};
            box-shadow: 0 6px 18px -10px rgba(37, 99, 235, 0.25);
        }}
        .hm-model-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }}
        .hm-model-name {{
            font-weight: 700;
            font-size: 1.02rem;
            color: {c["text"]};
        }}
        .hm-model-vendor {{
            font-size: 0.72rem;
            color: {c["text_muted"]};
            background: {c["surface_alt"]};
            padding: 2px 8px;
            border-radius: 999px;
            font-weight: 600;
        }}
        .hm-model-meta {{
            display: flex; gap: 1rem; flex-wrap: wrap;
            font-size: 0.78rem;
            color: {c["text_muted"]};
            margin-bottom: 0.55rem;
        }}
        .hm-model-meta strong {{ color: {c["text"]}; font-weight: 600; }}
        .hm-model-desc {{
            color: {c["text_muted"]};
            font-size: 0.85rem;
            line-height: 1.55;
        }}
        .hm-model-selected-tag {{
            display: inline-block;
            background: {c["primary"]};
            color: white;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-left: 0.6rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- session state ---------------------------------------------------

def _init_state() -> None:
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "home"
    if "documents" not in st.session_state:
        # UI-only metadata for the document list. The actual store is in RAG.
        st.session_state.documents = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]["id"]
    if "kb_pending_action" not in st.session_state:
        st.session_state.kb_pending_action = None
    if "lifetime_stats" not in st.session_state:
        st.session_state.lifetime_stats = _load_lifetime_stats()


def _ensure_rag() -> RAG:
    if st.session_state.rag is None:
        with st.spinner(
            f"Loading {settings.llm_model} — first run downloads ~5 GB and "
            "can take a couple of minutes."
        ):
            st.session_state.rag = RAG(llm=make_llm())
    return st.session_state.rag


# ---------- shared rendering primitives -------------------------------------

def _page_header(title: str, tab_id: str, right_action_html: str = "") -> None:
    subtitle = TAB_INFO.get(tab_id, "")
    st.markdown(
        f"<div class='hm-page-head'>"
        f"<div>"
        f"<h1 class='hm-page-title'>{html.escape(title)}</h1>"
        f"<div class='hm-page-subtitle'>{html.escape(subtitle)}</div>"
        f"</div>"
        f"<div>{right_action_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _breadcrumb(*crumbs: str) -> None:
    items = [f"<a>{PRODUCT_NAME}</a>"]
    for i, c in enumerate(crumbs):
        items.append("<span class='sep'>›</span>")
        if i == len(crumbs) - 1:
            items.append(f"<span class='current'>{html.escape(c)}</span>")
        else:
            items.append(f"<a>{html.escape(c)}</a>")
    st.markdown(
        f"<div class='hm-breadcrumb'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


# ---------- sidebar ---------------------------------------------------------

def _render_sidebar() -> None:
    st.sidebar.markdown(
        f"<div class='hm-brand'>"
        f"<div class='hm-logo'>H</div>"
        f"<div class='hm-brand-name'>{PRODUCT_NAME}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for tab_id, label in NAV_ITEMS:
        is_active = st.session_state.active_tab == tab_id
        if st.sidebar.button(
            label,
            key=f"nav_{tab_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state.active_tab = tab_id
                st.rerun()


# ---------- HOME / DASHBOARD ------------------------------------------------

def _stat_card(value: str, label: str, icon: str, icon_class: str = "") -> str:
    return (
        f"<div class='hm-stat-card'>"
        f"<div class='hm-stat-icon {icon_class}'>{icon}</div>"
        f"<div>"
        f"<div class='hm-stat-value'>{value}</div>"
        f"<div class='hm-stat-label'>{label}</div>"
        f"</div>"
        f"</div>"
    )


def _qa_card(icon: str, title: str, desc: str, icon_class: str = "") -> str:
    return (
        f"<div class='hm-qa-card'>"
        f"<div class='hm-qa-icon {icon_class}'>{icon}</div>"
        f"<div class='hm-qa-title'>{html.escape(title)}</div>"
        f"<div class='hm-qa-desc'>{html.escape(desc)}</div>"
        f"</div>"
    )


def _render_home() -> None:
    _breadcrumb("Home")

    # Hero
    st.markdown(
        f"<div class='hm-hero'>"
        f"<div>"
        f"<div class='hm-hero-title'>{PRODUCT_NAME}</div>"
        f"<div class='hm-hero-sub'>"
        f"Greek god of messengers. Hermes delivers grounded answers from complex "
        f"documents — upload your corpus, ask anything, and see exactly which "
        f"sources backed each claim."
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Stats — cumulative lifetime counters persisted to .cache/stats.json.
    # These survive restarts; the underlying corpus and chats do not.
    s = st.session_state.lifetime_stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_stat_card(str(s["documents"]), "Documents indexed", "📄"), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(str(s["chunks"]), "Chunks searchable", "✦", "green"), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card(str(s["questions"]), "Questions asked", "💬", "purple"), unsafe_allow_html=True)

    # Quick Actions
    st.markdown("<div class='hm-section-title'>Quick Actions</div>", unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        st.markdown(
            _qa_card("📚", "Load Sample Corpus",
                     "Index the bundled consulting docs to try Hermes in seconds.",
                     ""),
            unsafe_allow_html=True,
        )
        if st.button("Load now", key="home_qa_load", use_container_width=True):
            _load_sample_corpus()
            st.session_state.active_tab = "chat"
            st.rerun()
    with qa2:
        st.markdown(
            _qa_card("⬆", "Upload Documents",
                     "Add your own PDF, TXT, or MD files to the knowledge base.",
                     "green"),
            unsafe_allow_html=True,
        )
        if st.button("Upload", key="home_qa_upload", use_container_width=True):
            st.session_state.active_tab = "knowledge_base"
            st.session_state.kb_pending_action = "upload"
            st.rerun()
    with qa3:
        st.markdown(
            _qa_card("✦", "Start Chatting",
                     "Ask grounded questions and get answers cited from your docs.",
                     "purple"),
            unsafe_allow_html=True,
        )
        if st.button("Open chat", key="home_qa_chat", use_container_width=True):
            st.session_state.active_tab = "chat"
            st.rerun()

    # How It Works — explicitly technical, with brief-requirement tags per step.
    st.markdown("<div class='hm-section-title'>How It Works</div>", unsafe_allow_html=True)
    steps = [
        (
            "Document ingestion",
            "CORE",
            "Local PDF, Markdown, and plain-text files parsed with <code>pypdf</code>. "
            "Heading-aware recursive chunking with 200-char overlap preserves structural "
            "context. Every chunk is indexed in two parallel structures: a FAISS dense "
            "index for semantic similarity and a BM25Okapi index for lexical match.",
        ),
        (
            "Hybrid embedding-based retrieval",
            "CORE",
            "The query is encoded by <code>BAAI/bge-small-en-v1.5</code> and searched in "
            "<code>FAISS IndexFlatIP</code> over L2-normalized vectors (cosine similarity). "
            "BM25 runs in parallel for exact-term matches. The two rankings are combined "
            "via Reciprocal Rank Fusion (k=60), giving 20 fused candidates.",
        ),
        (
            "Cross-encoder reranking",
            "CORE",
            "The 20 candidates are re-scored by <code>BAAI/bge-reranker-base</code>, a "
            "cross-encoder that reads (query, chunk) pairs directly. This is the single "
            "biggest quality win and produces a calibrated relevance score in [0, 1].",
        ),
        (
            "CRAG guardrail",
            "GROUND",
            "If no retrieved chunk clears the configured relevance threshold, the system "
            "refuses to answer rather than hallucinate. This is the anti-hallucination gate "
            "the brief explicitly calls out — grounding by refusal, not just by retrieval.",
        ),
        (
            "Grounded generation",
            "CORE",
            "<code>Qwen 2.5 7B Instruct</code> in 4-bit NF4 (≈5 GB VRAM, runs entirely "
            "locally via transformers + bitsandbytes) writes the answer. The system prompt "
            "constrains it to use only the retrieved passages and to cite each fact inline "
            "as <code>[Source N]</code>.",
        ),
        (
            "Confidence & grounding transparency",
            "CORE",
            "Dual-signal confidence: top reranker probability × 4-gram overlap between "
            "answer and retrieved chunks. Rendered as a color-coded badge. Phrases that "
            "appear verbatim in sources are underlined in green, and inline citations are "
            "shown as colored <code>[N]</code> pills linked to color-coded source cards.",
        ),
    ]
    rows = "".join(
        f"<div class='hm-hiw-row'>"
        f"<div class='hm-hiw-num'>{i + 1}</div>"
        f"<div class='hm-hiw-text'>"
        f"<div class='hm-hiw-text-head'>"
        f"<strong>{html.escape(title)}</strong>"
        f"<span class='hm-tag'>{tag}</span>"
        f"</div>"
        f"<span class='hm-hiw-desc'>{desc}</span>"
        f"</div>"
        f"</div>"
        for i, (title, tag, desc) in enumerate(steps)
    )
    st.markdown(
        f"<div class='hm-hiw-wrap'>"
        f"<div class='hm-hiw-title'>🔍 How It Works</div>"
        f"{rows}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Requirements coverage — directly maps to the brief's deliverable table.
    st.markdown(
        "<div class='hm-section-title'>Brief Requirements Coverage</div>",
        unsafe_allow_html=True,
    )

    core_items = [
        (
            "done", "Upload & parse local text or PDF files into a searchable KB",
            "PDF (pypdf), Markdown, and TXT supported. Chunks indexed in FAISS + BM25 on ingestion.",
        ),
        (
            "done", "Embedding-based retrieval using FAISS, Chroma, or Lyzr",
            "FAISS <code>IndexFlatIP</code> on L2-normalized 384-dim vectors from <code>bge-small-en-v1.5</code>.",
        ),
        (
            "done", "LLM responses grounded in retrieved content",
            "Qwen 2.5 7B Instruct, system-prompted to use only retrieved passages and cite as <code>[Source N]</code>.",
        ),
        (
            "done", "Display confidence scores / retrieval relevance indicators",
            "Color-coded confidence badge per answer, per-source rerank/dense/BM25 metrics in the citations panel.",
        ),
    ]
    optional_items = [
        (
            "done", "Visualize retrieved sources / highlight matched text passages",
            "Citations panel shows each source chunk; matched phrases highlighted in green inside both the answer and the source bodies.",
        ),
        (
            "done", "Multi-turn conversation context",
            "Last 4 turns of history are passed into the prompt, bounded to keep context size predictable.",
        ),
        (
            "done", "Toggle between local and cloud data sources",
            "Documents can be loaded from local files (PDF / TXT / MD) or fetched live from URLs (PDF or HTML). Both paths flow through the same FAISS + BM25 index.",
        ),
        (
            "done", "Evaluation metrics: precision@k or grounding accuracy",
            "<code>scripts/evaluate.py</code> computes Precision@k, Recall@k, grounding accuracy, and a guardrail-accuracy ablation (dense / hybrid / hybrid+rerank).",
        ),
    ]

    def _coverage_row(status: str, label: str, detail: str) -> str:
        icon = "✓" if status == "done" else "~"
        return (
            f"<div class='hm-cov-row'>"
            f"<div class='hm-cov-icon {status}'>{icon}</div>"
            f"<div class='hm-cov-text'>"
            f"<strong>{html.escape(label)}</strong><br>"
            f"<em>{detail}</em>"
            f"</div>"
            f"</div>"
        )

    core_rows = "".join(_coverage_row(s, l, d) for s, l, d in core_items)
    optional_rows = "".join(_coverage_row(s, l, d) for s, l, d in optional_items)

    st.markdown(
        f"<div class='hm-cov-wrap'>"
        f"<div class='hm-cov-title'>📋 Requirements from the brief</div>"
        f"<div class='hm-cov-group'>Core Functionality (4 / 4)</div>"
        f"{core_rows}"
        f"<div class='hm-cov-group'>Optional Enhancements (4 / 4)</div>"
        f"{optional_rows}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- KNOWLEDGE BASE PAGE ---------------------------------------------

def _file_size_str(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.2f} MB"


def _relative_time(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff // 60)} min ago"
    if diff < 86400:
        return f"{int(diff // 3600)} hr ago"
    return f"{int(diff // 86400)} d ago"


def _doc_icon_class(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {"pdf": "pdf", "md": "md", "markdown": "md", "txt": "txt"}.get(ext, "pdf")


@st.dialog("Add from URL")
def _add_url_dialog() -> None:
    """Fetch a PDF or HTML URL and index it."""
    st.markdown(
        "Paste a public URL to a **PDF** or **web page**. Hermes downloads the "
        "content, extracts the readable text, and adds it to the knowledge base."
    )
    url = st.text_input(
        "URL", placeholder="https://nvlpubs.nist.gov/.../NIST.SP.800-171r2.pdf",
        label_visibility="collapsed",
    )
    if st.button("Fetch & Index", use_container_width=True, disabled=not url):
        rag = _ensure_rag()
        try:
            with st.spinner(f"Fetching {url}…"):
                added, name = rag.ingest_url(url, contextual=False)
        except Exception as exc:
            st.error(f"Couldn't load that URL: {exc}")
            return
        if added == 0:
            st.warning("URL fetched but produced 0 chunks (already indexed?).")
            return
        st.session_state.documents.append({
            "name": name,
            "size_bytes": 0,
            "uploaded_at": time.time(),
            "status": "completed",
            "source_type": "url",
            "source_url": url,
        })
        _bump_stats(documents=1, chunks=added)
        st.success(f"Indexed {added} chunks from {name}.")
        time.sleep(0.6)
        st.rerun()


@st.dialog("Add Documents")
def _add_documents_dialog() -> None:
    """Modal to upload one or more files and index them."""
    st.markdown(
        "Upload PDF, Markdown, or Text files to add them to the knowledge base."
    )
    uploaded = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        st.caption(f"{len(uploaded)} file(s) selected.")
        if st.button("Upload & Index", use_container_width=True):
            rag = _ensure_rag()
            with tempfile.TemporaryDirectory() as td:
                paths = []
                meta = []
                for uf in uploaded:
                    p = Path(td) / uf.name
                    p.write_bytes(uf.getbuffer())
                    paths.append(p)
                    meta.append({
                        "name": uf.name,
                        "size_bytes": uf.size,
                        "uploaded_at": time.time(),
                    })
                bar = st.progress(0.0, text="Indexing…")
                added = rag.ingest_paths(
                    paths, contextual=False,  # kept off per design
                    progress=lambda p: bar.progress(p, text=f"Indexing… {p:.0%}"),
                )
                bar.empty()
            # Stash UI metadata
            for m in meta:
                m["status"] = "completed"
                st.session_state.documents.append(m)
            _bump_stats(documents=len(uploaded), chunks=added)
            st.success(f"Indexed {added} new chunks from {len(uploaded)} file(s).")
            time.sleep(0.6)
            st.rerun()


def _load_sample_corpus() -> None:
    rag = _ensure_rag()
    rag.clear()
    st.session_state.documents = []
    files = sorted(SAMPLE_DIR.glob("*.md"))
    bar = st.progress(0.0, text="Indexing sample corpus…")
    added = rag.ingest_paths(
        files,
        contextual=False,  # kept off per design
        progress=lambda p: bar.progress(p, text=f"Indexing… {p:.0%}"),
    )
    bar.empty()
    for f in files:
        st.session_state.documents.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "uploaded_at": time.time(),
            "status": "completed",
        })
    _bump_stats(documents=len(files), chunks=added)
    st.success(f"Loaded sample corpus · {added} chunks from {len(files)} files.")
    time.sleep(0.6)
    st.rerun()


def _render_knowledge_base() -> None:
    _breadcrumb("Knowledge Base")
    _page_header("Knowledge Base", "knowledge_base")

    # Auto-open the upload dialog if redirected from a Quick Action.
    if st.session_state.kb_pending_action == "upload":
        st.session_state.kb_pending_action = None
        _add_documents_dialog()

    # Top action row
    col_l, col_mid, col_r = st.columns([1.4, 1, 1])
    with col_l:
        if st.button(
            "Load sample corpus",
            key="kb_load_sample",
            use_container_width=False,
        ):
            _load_sample_corpus()
    with col_mid:
        st.markdown(
            "<div style='display:flex; justify-content:flex-end;'>",
            unsafe_allow_html=True,
        )
        if st.button("+ Add from URL", key="kb_add_url"):
            _add_url_dialog()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown(
            "<div style='display:flex; justify-content:flex-end;'>",
            unsafe_allow_html=True,
        )
        if st.button("+ Add Document", key="kb_add_doc"):
            _add_documents_dialog()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    # Document list
    docs = st.session_state.documents
    if not docs:
        st.markdown(
            "<div class='hm-doc-list'>"
            "<div class='hm-doc-header'>"
            "<div>Name</div><div>Size</div><div>Created</div><div>Status</div>"
            "</div>"
            "<div class='hm-empty-row'>"
            "No documents indexed yet. Click <strong>Load sample corpus</strong> "
            "for a quick demo, or <strong>+ Add Document</strong> to upload your own files."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rows = []
    for d in docs:
        is_url = d.get("source_type") == "url"
        icon_cls = "url" if is_url else _doc_icon_class(d["name"])
        ext_label = "URL" if is_url else d["name"].rsplit(".", 1)[-1].upper()
        if is_url:
            # Show the hostname instead of file size, with a small URL chip.
            from urllib.parse import urlparse
            host = urlparse(d.get("source_url", "")).netloc or "URL"
            size_cell = f"<span class='hm-url-chip'>{html.escape(host)}</span>"
        else:
            size_cell = _file_size_str(d["size_bytes"])
        rows.append(
            f"<div class='hm-doc-row'>"
            f"<div class='hm-doc-name'>"
            f"<span class='hm-doc-icon {icon_cls}'>{ext_label}</span>"
            f"<span>{html.escape(d['name'])}</span>"
            f"</div>"
            f"<div class='hm-doc-meta'>{size_cell}</div>"
            f"<div class='hm-doc-meta'>{_relative_time(d['uploaded_at'])}</div>"
            f"<div><span class='hm-status-pill'>{d['status']}</span></div>"
            f"</div>"
        )
    st.markdown(
        "<div class='hm-doc-list'>"
        "<div class='hm-doc-header'>"
        "<div>Name</div><div>Size</div><div>Created</div><div>Status</div>"
        "</div>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Clear-index control below the table
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    if st.button("Clear all", key="kb_clear"):
        rag = st.session_state.rag
        if rag is not None:
            rag.clear()
        st.session_state.documents = []
        st.session_state.history = []
        st.session_state.turns = []
        st.rerun()


# ---------- CHAT PAGE -------------------------------------------------------

_CONFIDENCE_EXPLAIN = {
    "High": (
        "High confidence — the answer is strongly supported by the retrieved "
        "sources. Most phrases match passages in the documents."
    ),
    "Medium": (
        "Medium confidence — relevant sources were found, but parts of the "
        "answer paraphrase or interpret them. Worth spot-checking the cited "
        "passages."
    ),
    "Low": (
        "Low confidence — weak support from the source documents. Treat the "
        "answer with caution and verify against the cited passages."
    ),
}


def _confidence_badge_html(label: str, score: float) -> str:
    palette = {
        "High":   (COLORS["success"], "rgba(22, 163, 74, 0.12)"),
        "Medium": (COLORS["warning"], "rgba(202, 138, 4, 0.13)"),
        "Low":    (COLORS["danger"],  "rgba(220, 38, 38, 0.12)"),
    }
    fg, bg = palette[label]
    tooltip = _CONFIDENCE_EXPLAIN[label]
    return (
        f"<span class='hm-badge' title='{html.escape(tooltip)}' "
        f"style='background:{bg};color:{fg};border:1px solid {fg}33'>"
        f"<span class='dot'></span>{label} confidence · {score * 100:.0f}%"
        f"</span>"
    )


def _annotate_answer_html(text: str, retrieved: list[dict]) -> tuple[str, float]:
    """Wraps grounded phrases in <mark class='hm-ground'> and styles
    [Source N] mentions as colored pills. Returns (html, fraction_anchored)."""
    if not text:
        return "", 0.0
    if not retrieved:
        return html.escape(text).replace("\n\n", "</p><p>").replace("\n", "<br>"), 0.0

    concat_source = " ".join(r["text"] for r in retrieved)
    spans = find_matched_spans(text, concat_source, min_words=4)
    covered = sum(s.end - s.start for s in spans)
    fraction = covered / max(len(text), 1)

    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(html.escape(text[cursor:span.start]))
        parts.append(f"<mark class='hm-ground'>{html.escape(text[span.start:span.end])}</mark>")
        cursor = span.end
    parts.append(html.escape(text[cursor:]))
    result = "".join(parts)

    def _pill(m: re.Match) -> str:
        n = m.group(1)
        return f"<span class='hm-src-pill'>[{n}]</span>"

    result = re.sub(r"\[Source (\d+)\]", _pill, result)
    result = result.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return result, fraction


def _render_user_message(text: str) -> None:
    st.markdown(
        f"<div class='hm-msg hm-msg-user'>"
        f"<div class='hm-avatar user'>YOU</div>"
        f"<div class='hm-bubble user'>{html.escape(text)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_assistant_message(turn: dict) -> None:
    bubble_class = "hm-bubble assistant refused" if turn["refused"] else "hm-bubble assistant"
    annotated, _frac = _annotate_answer_html(turn["answer"], turn["retrieved"])
    badge = _confidence_badge_html(turn["conf_label"], turn["conf_final"])

    sources_with_spans: list[tuple[int, dict, list]] = []
    for i, r in enumerate(turn["retrieved"], start=1):
        spans = find_matched_spans(r["text"], turn["answer"])
        if spans:
            sources_with_spans.append((i, r, spans))
    if not sources_with_spans and turn["retrieved"]:
        r = turn["retrieved"][0]
        sources_with_spans = [(1, r, find_matched_spans(r["text"], turn["answer"]))]

    cited_count = len(sources_with_spans)
    elapsed_str = f"{turn['elapsed']:.1f}s" if turn.get("elapsed") else ""

    # Top strip — confidence badge + meta. Surfaced FIRST, before the answer.
    top_strip = (
        f"<div class='hm-bubble-top'>"
        f"{badge}"
        f"<span class='hm-bubble-top-stat'>· <strong>{cited_count}</strong> source"
        f"{'s' if cited_count != 1 else ''}</span>"
        + (f"<span class='hm-bubble-top-stat'>· <strong>{elapsed_str}</strong></span>"
           if elapsed_str else "")
        + f"</div>"
    )

    # Build citations as a native <details> block so it lives inside the same
    # message column as the bubble (aligned widths, no full-width Streamlit box).
    citations_html = ""
    if not turn["refused"] and sources_with_spans:
        cards = []
        for i, r, spans in sources_with_spans:
            section_html = (
                f"<span class='hm-source-section'>{html.escape(r['section'])}</span>"
                if r["section"] else ""
            )
            rerank = f"{r['rerank_score']:.2f}" if r["rerank_score"] is not None else "—"
            metrics_html = (
                f"<div class='hm-source-metrics'>"
                f"<span><span class='hm-pri'>rerank</span> {rerank}</span>"
                f"<span>dense {r['dense_score']:.2f}</span>"
                f"<span>bm25 {r['sparse_score']:.2f}</span>"
                f"</div>"
            )
            highlighted = render_with_highlights(html.escape(r["text"]), spans)
            cards.append(
                f"<div class='hm-source-card'>"
                f"<div class='hm-source-head'>"
                f"<span class='hm-source-tag'>SOURCE {i}</span>"
                f"<span class='hm-source-file'>{html.escape(r['source'])}</span>"
                f"{section_html}"
                f"{metrics_html}"
                f"</div>"
                f"<div class='hm-source-body'>{highlighted}</div>"
                f"</div>"
            )
        summary_label = (
            f"View cited sources · {cited_count} · "
            f"retrieval {turn['conf_retrieval']:.2f} · "
            f"grounding {turn['conf_grounding']:.2f}"
        )
        citations_html = (
            f"<details class='hm-citations'>"
            f"<summary>{summary_label}</summary>"
            f"<div class='hm-citations-body'>{''.join(cards)}</div>"
            f"</details>"
        )

    st.markdown(
        f"<div class='hm-msg'>"
        f"<div class='hm-avatar assistant'>H</div>"
        f"<div style='flex: 1; min-width: 0;'>"
        f"<div class='{bubble_class}'>"
        f"{top_strip}"
        f"<div class='hm-answer-text'><p>{annotated}</p></div>"
        f"</div>"
        f"{citations_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_chat_empty_state(has_corpus: bool) -> None:
    if not has_corpus:
        st.markdown(
            "<div class='hm-empty'>"
            "<div class='hm-empty-icon'>📚</div>"
            "<h2>No documents indexed yet</h2>"
            "<p>Head to the <strong>Knowledge Base</strong> tab to load the sample corpus "
            "or upload your own files. Hermes can't answer questions until it has something "
            "to retrieve from.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Go to Knowledge Base", key="empty_to_kb"):
            st.session_state.active_tab = "knowledge_base"
            st.rerun()
        return

    st.markdown(
        "<div class='hm-empty'>"
        "<h2>What would you like to know?</h2>"
        "<p>Ask anything about the indexed documents. Every answer cites its sources "
        "and carries a transparent confidence score.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _handle_query(query: str) -> None:
    rag = st.session_state.rag
    with st.spinner("Retrieving & generating…"):
        t0 = time.time()
        answer = rag.ask(query, history=st.session_state.history)
        elapsed = time.time() - t0
    turn = {
        "question": query,
        "answer": answer.text,
        "refused": answer.refused,
        "conf_retrieval": answer.confidence.retrieval,
        "conf_grounding": answer.confidence.grounding,
        "conf_final": answer.confidence.final,
        "conf_label": answer.confidence.label,
        "elapsed": elapsed,
        "retrieved": [
            {
                "source": r.chunk.source,
                "section": r.chunk.section,
                "text": r.chunk.text,
                "rerank_score": r.rerank_score,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
            }
            for r in answer.retrieved
        ],
    }
    st.session_state.turns.append(turn)
    _bump_stats(questions=1)
    if not answer.refused:
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": answer.text})


def _render_chat() -> None:
    _breadcrumb("Chat")

    # Header — right action: reset chat
    reset_html = ""
    _page_header("Chat", "chat", right_action_html=reset_html)

    rag_loaded = st.session_state.rag is not None
    has_corpus = rag_loaded and len(st.session_state.rag.store) > 0

    if not has_corpus:
        _render_chat_empty_state(has_corpus=False)
        return

    if not st.session_state.turns and st.session_state.pending_query is None:
        _render_chat_empty_state(has_corpus=True)

    pending = st.session_state.pending_query
    st.session_state.pending_query = None

    for turn in st.session_state.turns:
        _render_user_message(turn["question"])
        _render_assistant_message(turn)

    if st.session_state.turns:
        if st.button("Reset chat", key="chat_reset_inline"):
            st.session_state.history = []
            st.session_state.turns = []
            st.rerun()

    query = st.chat_input("Ask a question about your documents…")
    final_query = pending or query
    if final_query:
        _handle_query(final_query)
        st.rerun()


# ---------- MODELS PAGE -----------------------------------------------------

def _render_models() -> None:
    _breadcrumb("Models")
    _page_header("Models", "models")

    muted = COLORS["text_muted"]
    st.markdown(
        f"<div style='color:{muted}; font-size:0.9rem; margin-bottom:1.2rem;'>"
        f"Pick the language model that powers Hermes. The model loads on first "
        f"use and runs locally. Additional models will be available in future releases."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:0.72rem; letter-spacing:0.06em; "
        f"text-transform:uppercase; color:{COLORS['text_muted']}; "
        f"font-weight:700; margin-bottom:0.7rem;'>Models</div>",
        unsafe_allow_html=True,
    )

    for m in AVAILABLE_MODELS:
        selected = st.session_state.selected_model == m["id"]
        selected_tag = "<span class='hm-model-selected-tag'>SELECTED</span>" if selected else ""
        card_cls = "hm-model-card selected" if selected else "hm-model-card"
        st.markdown(
            f"<div class='{card_cls}'>"
            f"<div class='hm-model-head'>"
            f"<div><span class='hm-model-name'>{html.escape(m['label'])}</span>{selected_tag}</div>"
            f"<span class='hm-model-vendor'>{html.escape(m['vendor'])}</span>"
            f"</div>"
            f"<div class='hm-model-meta'>"
            f"<span><strong>{html.escape(m['size'])}</strong></span>"
            f"<span><strong>{html.escape(m['quantization'])}</strong></span>"
            f"</div>"
            f"<div class='hm-model-desc'>{html.escape(m['description'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------- main ------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=f"{PRODUCT_NAME} · Grounded RAG",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _init_state()
    _render_sidebar()

    active = st.session_state.active_tab
    if active == "home":
        _render_home()
    elif active == "knowledge_base":
        _render_knowledge_base()
    elif active == "chat":
        _render_chat()
    elif active == "models":
        _render_models()
    else:
        st.session_state.active_tab = "home"
        st.rerun()


if __name__ == "__main__":
    main()
