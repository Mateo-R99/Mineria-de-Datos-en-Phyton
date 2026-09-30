"""
Componentes visuales reutilizables para la app de despliegue.

Este módulo NO contiene lógica de negocio (no toca el modelo ni los datos):
solo funciones de presentación (HTML/CSS embebido y gráficos) que
`despliegue.py` usa para armar la interfaz. Mantenerlas separadas facilita
ajustar el diseño sin tocar la lógica de predicción.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets"


def cargar_css() -> None:
    """Inyecta el CSS personalizado (app/assets/style.css) en la página."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero(titulo: str, subtitulo: str, eyebrow: str, badges: list[str]) -> None:
    """Encabezado tipo 'hero section' con degradado, título y chips informativos."""
    badges_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-eyebrow">{eyebrow}</span>
            <h1 class="hero-title">{titulo}</h1>
            <p class="hero-subtitle">{subtitulo}</p>
            <div class="hero-badges">{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(icon: str, titulo: str, subtitulo: str | None = None) -> None:
    """Título de sección con un pequeño ícono en una 'chip' redondeada."""
    st.markdown(
        f'<div class="section-title"><span class="icon-chip">{icon}</span>{titulo}</div>',
        unsafe_allow_html=True,
    )
    if subtitulo:
        st.markdown(f'<div class="section-subtitle">{subtitulo}</div>', unsafe_allow_html=True)


def render_stat_row(stats: list[tuple[str, str]]) -> None:
    """Fila de 'stat chips' (valor grande + etiqueta). `stats` es una lista de (valor, etiqueta)."""
    chips = "".join(
        f'<div class="stat-chip"><div class="stat-value">{valor}</div>'
        f'<div class="stat-label">{etiqueta}</div></div>'
        for valor, etiqueta in stats
    )
    st.markdown(f'<div class="stat-row">{chips}</div>', unsafe_allow_html=True)


def render_result_card(sobrevive: bool, probabilidad: float) -> None:
    """Tarjeta grande de resultado (verde si sobrevive, roja si no)."""
    if sobrevive:
        icono, titulo, desc, clase = (
            "🟢",
            "Predicción: SOBREVIVE",
            f"El modelo estima una probabilidad de {probabilidad:.1%} de que este pasajero sobreviva.",
            "survive",
        )
    else:
        icono, titulo, desc, clase = (
            "🔴",
            "Predicción: NO SOBREVIVE",
            f"El modelo estima una probabilidad de {1 - probabilidad:.1%} de que este pasajero no sobreviva.",
            "no-survive",
        )
    st.markdown(
        f"""
        <div class="result-card {clase}">
            <div class="result-icon">{icono}</div>
            <div>
                <p class="result-title">{titulo}</p>
                <p class="result-desc">{desc}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(prob_sobrevive: float, prob_no_sobrevive: float) -> None:
    """Barras de probabilidad personalizadas (reemplazo visual de st.metric)."""
    st.markdown(
        f"""
        <div class="prob-block">
            <div class="prob-label"><span>🟢 Probabilidad de sobrevivir</span><span>{prob_sobrevive:.1%}</span></div>
            <div class="prob-track"><div class="prob-fill survive" style="width:{prob_sobrevive*100:.1f}%"></div></div>
        </div>
        <div class="prob-block">
            <div class="prob-label"><span>🔴 Probabilidad de no sobrevivir</span><span>{prob_no_sobrevive:.1%}</span></div>
            <div class="prob-track"><div class="prob-fill no-survive" style="width:{prob_no_sobrevive*100:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance_chart(importancias: pd.Series) -> go.Figure:
    """Gráfico de importancia de variables con Plotly (interactivo, con degradado)."""
    importancias = importancias.sort_values(ascending=True)
    valores = [float(v) for v in importancias.values]
    fig = go.Figure(
        go.Bar(
            x=valores,
            y=list(importancias.index),
            orientation="h",
            marker=dict(
                color=valores,
                colorscale=[[0, "#0e7490"], [1, "#06b6d4"]],
                cmin=min(valores),
                cmax=max(valores),
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=10, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=11),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def render_sidebar_card_start(titulo: str, icon: str) -> None:
    st.markdown(
        f'<div class="sidebar-card"><div class="sidebar-title">{icon} {titulo}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar_badges(items: list[str]) -> None:
    html = "".join(f'<span class="sb-badge">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)
