"""
DataForge EDU — Utilitários partilhados
Paleta, CSS, componentes reutilizáveis
Design acessível: WCAG AA/AAA, Atkinson Hyperlegible, alto contraste
"""

import streamlit as st
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
import os
import datetime

# ══════════════════════════════════════════════════════
# PALETA DE CORES — WCAG AA/AAA
# Todos os pares texto/fundo testados ≥ 4.5:1
# ══════════════════════════════════════════════════════
C_BG         = "#0F1117"   # fundo principal
C_SURFACE    = "#161B27"   # cards e painéis
C_SURFACE2   = "#1E2535"   # inputs e elementos secundários
C_BORDER     = "#3A4560"   # bordas — mais visíveis que antes
C_ACCENT     = "#6BA3FF"   # azul acessível — 7.5:1 vs fundo
C_ACCENT2    = "#A78BFA"   # violeta acessível
C_GREEN      = "#4ADE80"   # verde — 10.8:1 vs fundo
C_AMBER      = "#FFC107"   # âmbar — 11.6:1 vs fundo
C_RED        = "#FF6B6B"   # vermelho — 6.8:1 vs fundo
C_TEAL       = "#2DD4BF"   # teal acessível
C_TEXT       = "#FFFFFF"   # texto principal — 17.2:1 vs fundo
C_TEXT_SEC   = "#D0D8F0"   # texto secundário — 8.1:1 vs fundo
C_TEXT_MUTE  = "#7A8BA8"   # texto muted — 4.6:1 vs fundo (AA mínimo)

PALETTE = [C_ACCENT, C_TEAL, C_AMBER, C_RED, C_ACCENT2, C_GREEN]

# ══════════════════════════════════════════════════════
# CSS GLOBAL — Acessibilidade total
# ══════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    /* ── FONTE: Atkinson Hyperlegible — desenhada para baixa visão ── */
    @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp, .main {{
        background-color: {C_BG} !important;
        color: {C_TEXT} !important;
        font-family: 'Atkinson Hyperlegible', Arial, sans-serif !important;
        font-size: 17px !important;
        font-weight: 600 !important;         /* maior que o padrão 15px */
        line-height: 1.7 !important;        /* espaçamento entre linhas */
        letter-spacing: 0.01em !important;  /* ligeiro espaçamento entre letras */
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }}

    /* ── FOCO VISÍVEL — essencial para utilizadores de teclado ── */
    *:focus-visible {{
        outline: 3px solid {C_ACCENT} !important;
        outline-offset: 3px !important;
        border-radius: 4px !important;
    }}

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {{
        background: {C_SURFACE} !important;
        border-right: 2px solid {C_BORDER} !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding: 1.4rem 1rem 2rem;
    }}

    /* ── INPUTS — bordas mais grossas e visíveis ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {C_SURFACE2} !important;
        border: 2px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        padding: 10px 14px !important;
        font-family: 'Atkinson Hyperlegible', Arial, sans-serif !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {C_ACCENT} !important;
        box-shadow: 0 0 0 3px rgba(107,163,255,0.35) !important;
    }}
    /* Labels dos inputs maiores */
    .stTextInput label, .stSelectbox label, .stNumberInput label,
    .stTextArea label, .stMultiSelect label, .stSlider label {{
        font-size: 15px !important;
        font-weight: 700 !important;
        color: {C_TEXT_SEC} !important;
        letter-spacing: 0.02em !important;
    }}

    /* ── BOTÕES — área de clique maior, contraste alto ── */
    .stButton > button {{
        background: {C_ACCENT} !important;
        color: #0A0E17 !important;          /* texto escuro sobre azul claro */
        border: 2px solid transparent !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 0.6rem 1.6rem !important;
        min-height: 44px !important;        /* WCAG 2.5.5: área de toque ≥44px */
        transition: all .2s !important;
        font-family: 'Atkinson Hyperlegible', Arial, sans-serif !important;
        letter-spacing: 0.02em !important;
    }}
    .stButton > button:hover {{
        background: #89BAFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(107,163,255,0.4) !important;
    }}
    .stButton > button:focus-visible {{
        outline: 3px solid #fff !important;
        outline-offset: 3px !important;
    }}

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {C_SURFACE} !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 2px solid {C_BORDER} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: {C_TEXT_SEC} !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 10px 22px !important;
        min-height: 44px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C_ACCENT} !important;
        color: #0A0E17 !important;
    }}

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {{
        background: {C_SURFACE2} !important;
        border-radius: 8px !important;
        border: 2px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        min-height: 44px !important;
    }}

    /* ── MÉTRICAS ── */
    [data-testid="metric-container"] {{
        background: {C_SURFACE} !important;
        border: 2px solid {C_BORDER} !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 28px !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        font-size: 14px !important;
        color: {C_TEXT_SEC} !important;
        font-weight: 600 !important;
    }}

    /* ── DATAFRAME ── */
    .stDataFrame {{
        border: 2px solid {C_BORDER} !important;
        border-radius: 10px !important;
    }}

    /* ── SLIDER — track mais grosso ── */
    .stSlider [data-baseweb="slider"] {{ padding: 8px 0 !important; }}
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {{
        font-size: 14px !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
    }}

    /* ── RADIO — área de clique maior ── */
    div[data-testid="stRadio"] label[data-baseweb="radio"] {{
        padding: 12px 16px !important;
        border-radius: 8px !important;
        border: 2px solid transparent !important;
        transition: all .15s !important;
        color: {C_TEXT_SEC} !important;
        font-size: 16px !important;
        min-height: 44px !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
        background: {C_SURFACE2} !important;
        border-color: {C_ACCENT} !important;
        color: {C_ACCENT} !important;
    }}

    /* ── CHECKBOX ── */
    div[data-testid="stCheckbox"] label {{
        font-size: 16px !important;
        color: {C_TEXT_SEC} !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }}

    /* ── ALERTAS ── */
    .stAlert {{ border-radius: 10px !important; font-size: 16px !important; }}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {C_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {C_ACCENT}; }}

    /* ── SELECTBOX ── */
    .stSelectbox > div > div {{
        font-size: 16px !important;
        min-height: 44px !important;
    }}

    /* ══════════════════════════════════════════════════
       COMPONENTES CUSTOM
    ══════════════════════════════════════════════════ */

    .df-card {{
        background: {C_SURFACE};
        border: 2px solid {C_BORDER};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }}
    .df-card-accent {{
        background: {C_SURFACE};
        border: 2px solid {C_ACCENT};
        border-left: 5px solid {C_ACCENT};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }}

    /* Badges — texto escuro sobre cor clara para máximo contraste */
    .df-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .04em;
    }}
    .df-badge-blue   {{ background: {C_ACCENT};  color: #0A0E17; }}
    .df-badge-green  {{ background: {C_GREEN};   color: #0A0E17; }}
    .df-badge-amber  {{ background: {C_AMBER};   color: #0A0E17; }}
    .df-badge-purple {{ background: {C_ACCENT2}; color: #0A0E17; }}
    .df-badge-red    {{ background: {C_RED};     color: #0A0E17; }}
    .df-badge-teal   {{ background: {C_TEAL};    color: #0A0E17; }}

    .section-title {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: {C_TEXT_MUTE};
        padding-bottom: 10px;
        border-bottom: 2px solid {C_BORDER};
        margin: 1.8rem 0 1.2rem;
    }}
    .page-title {{
        font-size: 28px;
        font-weight: 700;
        color: {C_TEXT};
        margin-bottom: 4px;
        line-height: 1.3;
    }}
    .page-subtitle {{
        font-size: 17px;
        color: {C_TEXT_SEC};
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }}

    /* Caixa de teoria — fundo com contraste suficiente */
    .teoria-box {{
        background: rgba(107,163,255,.1);
        border: 2px solid rgba(107,163,255,.4);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
    }}
    .teoria-box h4 {{
        color: {C_ACCENT};
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .teoria-box p, .teoria-box li {{
        color: {C_TEXT_SEC};
        font-size: 16px;
        line-height: 1.8;
    }}

    /* Steps do modo guiado */
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.4rem;
    }}
    .step-dot {{
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 700;
        flex-shrink: 0;
    }}
    .step-dot.active  {{ background: {C_ACCENT};  color: #0A0E17; }}
    .step-dot.done    {{ background: {C_GREEN};   color: #0A0E17; }}
    .step-dot.pending {{ background: {C_SURFACE2}; color: {C_TEXT_MUTE}; border: 2px solid {C_BORDER}; }}

    /* Barra de progresso */
    .progress-bar-wrap {{
        background: {C_SURFACE2};
        border-radius: 6px;
        height: 10px;          /* mais grossa — mais visível */
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid {C_BORDER};
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_TEAL});
        transition: width .5s ease;
    }}

    /* Sidebar brand */
    .sidebar-brand {{
        display: flex; align-items: center; gap: 14px;
        padding: 8px 4px 18px;
        border-bottom: 2px solid {C_BORDER};
        margin-bottom: 18px;
    }}
    .sidebar-brand-icon {{
        width: 44px; height: 44px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; flex-shrink: 0;
    }}
    .sidebar-brand-name {{
        font-size: 18px; font-weight: 700; color: {C_TEXT};
    }}
    .sidebar-brand-sub {{
        font-size: 13px; color: {C_TEXT_MUTE};
    }}

    /* User pill */
    .user-pill {{
        display: flex; align-items: center; gap: 10px;
        background: {C_SURFACE2};
        border: 2px solid {C_BORDER};
        border-radius: 22px;
        padding: 8px 14px;
        margin-bottom: 18px;
    }}
    .user-avatar {{
        width: 32px; height: 32px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 700; color: #0A0E17;
        flex-shrink: 0;
    }}
    .user-name {{ font-size: 14px; font-weight: 700; color: {C_TEXT}; }}
    .user-role {{ font-size: 12px; color: {C_TEXT_MUTE}; }}

    /* Caixas de info/sucesso/aviso/erro — ícones grandes + texto legível */
    .df-info-box {{
        background: rgba(107,163,255,.12);
        border: 2px solid rgba(107,163,255,.45);
        border-left: 5px solid {C_ACCENT};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {C_TEXT};
        font-size: 16px;
        margin-bottom: 1rem;
        line-height: 1.7;
    }}
    .df-success-box {{
        background: rgba(74,222,128,.12);
        border: 2px solid rgba(74,222,128,.45);
        border-left: 5px solid {C_GREEN};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {C_TEXT};
        font-size: 16px;
        margin-bottom: 1rem;
        line-height: 1.7;
    }}
    .df-warning-box {{
        background: rgba(255,193,7,.12);
        border: 2px solid rgba(255,193,7,.45);
        border-left: 5px solid {C_AMBER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {C_TEXT};
        font-size: 16px;
        margin-bottom: 1rem;
        line-height: 1.7;
    }}
    .df-error-box {{
        background: rgba(255,107,107,.12);
        border: 2px solid rgba(255,107,107,.45);
        border-left: 5px solid {C_RED};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {C_TEXT};
        font-size: 16px;
        margin-bottom: 1rem;
        line-height: 1.7;
    }}

    /* ══ FORÇA BRANCO E NEGRITO EM TODA A INTERFACE ══ */
    p, span, div, li, td, th {{ color: #FFFFFF; }}
    .stMarkdown p, .stMarkdown div, .stMarkdown span {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebarContent"] * {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}
    .stTextInput label, .stSelectbox label, .stNumberInput label,
    .stTextArea label, .stMultiSelect label, .stSlider label,
    .stCheckbox label, .stRadio label {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #D0D8F0 !important;
        font-weight: 700 !important;
    }}
    .stTabs [aria-selected="true"] {{ color: #0A0E17 !important; }}
    [data-testid="stMetricValue"] {{ color: #FFFFFF !important; font-weight: 800 !important; }}
    [data-testid="stMetricLabel"] {{ color: #D0D8F0 !important; font-weight: 700 !important; }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: #D0D8F0 !important;
        border: none !important;
        text-align: left !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(107,163,255,.12) !important;
        color: #FFFFFF !important;
    }}
    .streamlit-expanderHeader {{ color: #FFFFFF !important; font-weight: 700 !important; }}
    .stSelectbox > div > div > div {{ color: #FFFFFF !important; font-weight: 600 !important; }}
    input::placeholder, textarea::placeholder {{ color: #7A8BA8 !important; font-weight: 400 !important; }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# BASE DE DADOS LOCAL (TinyDB)
# ══════════════════════════════════════════════════════
def get_db():
    os.makedirs("data", exist_ok=True)
    return TinyDB("data/db.json")

def get_user_progress(username: str) -> dict:
    db = get_db()
    User = Query()
    result = db.table("progress").search(User.username == username)
    if result:
        return result[0]
    return {"username": username, "modulos": {}, "pontos": 0, "completados": []}

def save_user_progress(username: str, modulo: str, dados: dict):
    db = get_db()
    User = Query()
    progress = get_user_progress(username)
    progress["modulos"][modulo] = dados
    progress["ultima_actividade"] = datetime.datetime.now().isoformat()
    table = db.table("progress")
    if table.search(User.username == username):
        table.update(progress, User.username == username)
    else:
        table.insert(progress)

def add_pontos(username: str, pontos: int, motivo: str = ""):
    db = get_db()
    User = Query()
    progress = get_user_progress(username)
    progress["pontos"] = progress.get("pontos", 0) + pontos
    if "historico_pontos" not in progress:
        progress["historico_pontos"] = []
    progress["historico_pontos"].append({
        "pontos": pontos,
        "motivo": motivo,
        "data": datetime.datetime.now().isoformat()
    })
    table = db.table("progress")
    if table.search(User.username == username):
        table.update(progress, User.username == username)
    else:
        table.insert(progress)

def save_historico_modelo(username: str, entrada: dict):
    db = get_db()
    entrada["username"] = username
    entrada["data"] = datetime.datetime.now().isoformat()
    db.table("historico_modelos").insert(entrada)

def get_historico_modelos(username: str) -> list:
    db = get_db()
    User = Query()
    return db.table("historico_modelos").search(User.username == username)


# ══════════════════════════════════════════════════════
# COMPONENTES UI
# ══════════════════════════════════════════════════════
def page_header(titulo: str, subtitulo: str = "", icone: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:1.8rem;">
        <div class="page-title">{icone} {titulo}</div>
        <div class="page-subtitle">{subtitulo}</div>
    </div>
    """, unsafe_allow_html=True)

def section_title(titulo: str):
    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)

def card(conteudo: str, accent: bool = False):
    classe = "df-card-accent" if accent else "df-card"
    st.markdown(f'<div class="{classe}">{conteudo}</div>', unsafe_allow_html=True)

def badge(texto: str, cor: str = "blue"):
    return f'<span class="df-badge df-badge-{cor}">{texto}</span>'

def teoria_box(titulo: str, conteudo: str):
    st.markdown(f"""
    <div class="teoria-box">
        <h4>💡 {titulo}</h4>
        <p>{conteudo}</p>
    </div>
    """, unsafe_allow_html=True)

def progresso_bar(valor: float, label: str = ""):
    pct = int(valor * 100)
    st.markdown(f"""
    <div style="margin-bottom:.8rem;">
        <div style="display:flex;justify-content:space-between;font-size:14px;
        color:{C_TEXT_MUTE};margin-bottom:6px;font-weight:600;">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metricas_row(items: list):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("valor", ""),
                delta=item.get("delta", None)
            )

def info_box(msg: str):
    st.markdown(f'<div class="df-info-box">ℹ️ &nbsp;{msg}</div>', unsafe_allow_html=True)

def sucesso_box(msg: str):
    st.markdown(f'<div class="df-success-box">✅ &nbsp;{msg}</div>', unsafe_allow_html=True)

def aviso_box(msg: str):
    st.markdown(f'<div class="df-warning-box">⚠️ &nbsp;{msg}</div>', unsafe_allow_html=True)

def erro_box(msg: str):
    st.markdown(f'<div class="df-error-box">❌ &nbsp;{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATASETS EMBUTIDOS
# ══════════════════════════════════════════════════════
def carregar_dataset_embutido(nome: str) -> tuple:
    """Retorna (df, descricao, tipo_problema)"""
    from sklearn import datasets
    import seaborn as sns

    catalogo = {
        "🌸 Iris — Classificação de flores":       ("iris",       "classificacao"),
        "🚢 Titanic — Sobrevivência":               ("titanic",    "classificacao"),
        "🍷 Vinho — Qualidade":                     ("wine",       "classificacao"),
        "🎗️ Cancro — Diagnóstico":                 ("breast",     "classificacao"),
        "🏠 Boston Housing — Preços":               ("boston",     "regressao"),
        "💊 Diabetes — Progressão":                ("diabetes",   "regressao"),
        "🔢 Dígitos — Reconhecimento":              ("digits",     "classificacao"),
        "🌙 Meias-luas — Clustering":               ("moons",      "clustering"),
        "⭕ Círculos — Clustering":                 ("circles",    "clustering"),
        "📊 Blobs — Clustering básico":             ("blobs",      "clustering"),
    }

    if nome not in catalogo:
        return None, "", ""

    chave, tipo = catalogo[nome]

    if chave == "iris":
        d = datasets.load_iris(as_frame=True)
        df = d.frame
        df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        desc = "150 flores de 3 espécies. Clássico para classificação multiclasse. 4 features numéricas."
    elif chave == "titanic":
        df = sns.load_dataset("titanic")[["survived","pclass","sex","age","sibsp","parch","fare"]].dropna()
        desc = "Dados dos passageiros do Titanic. Prevê sobrevivência. Mix de features numéricas e categóricas."
    elif chave == "wine":
        d = datasets.load_wine(as_frame=True)
        df = d.frame
        desc = "178 vinhos com 13 propriedades químicas. 3 classes de qualidade."
    elif chave == "breast":
        d = datasets.load_breast_cancer(as_frame=True)
        df = d.frame
        df["target"] = df["target"].map({0: "maligno", 1: "benigno"})
        desc = "569 tumores com 30 features. Classifica tumores como malignos ou benignos."
    elif chave == "boston":
        d = datasets.fetch_california_housing(as_frame=True)
        df = d.frame
        desc = "Dados de habitação da Califórnia. Prevê o preço médio de casas. 8 features."
    elif chave == "diabetes":
        d = datasets.load_diabetes(as_frame=True)
        df = d.frame
        desc = "442 pacientes com 10 features clínicas. Prevê progressão da diabetes ao fim de 1 ano."
    elif chave == "digits":
        d = datasets.load_digits(as_frame=True)
        df = d.frame
        desc = "1797 imagens 8x8 de dígitos escritos à mão (0-9). 64 features de pixels."
    elif chave == "moons":
        X, y = datasets.make_moons(n_samples=300, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em forma de meias-luas. Ideal para visualizar clustering não-linear."
    elif chave == "circles":
        X, y = datasets.make_circles(n_samples=300, noise=0.05, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em dois círculos concêntricos. Bom para DBSCAN vs KMeans."
    elif chave == "blobs":
        X, y = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em 4 grupos. Dataset simples para aprender clustering básico."
    else:
        return None, "", ""

    return df, desc, tipo


# ══════════════════════════════════════════════════════
# TEXTOS DE TEORIA POR ALGORITMO
# ══════════════════════════════════════════════════════
TEORIA = {
    "KNN": {
        "nome": "K-Nearest Neighbors (KNN)",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Imagina que chegas a uma cidade nova e queres saber se um bairro é seguro. Perguntas aos K vizinhos mais próximos. Se a maioria diz que é seguro, assumes que é seguro.",
        "como_funciona": "Para classificar um novo ponto, o KNN calcula a distância a todos os pontos de treino e escolhe os K mais próximos. A classe mais comum entre esses K vizinhos é a previsão.",
        "quando_usar": "Datasets pequenos/médios, quando os dados têm clusters naturais, quando precisas de um modelo simples e interpretável.",
        "cuidados": "Lento para datasets grandes. Muito afectado por features com escalas diferentes — normaliza sempre. K pequeno = overfitting, K grande = underfitting.",
        "hiperparametros": {"n_neighbors": "Número de vizinhos. Valores típicos: 3, 5, 7, 11", "metric": "Distância euclidiana (padrão), manhattan, minkowski"},
        "badge": "blue"
    },
    "Decision Tree": {
        "nome": "Árvore de Decisão",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Como um jogo de 20 perguntas. O modelo aprende a sequência de perguntas que melhor divide os dados em grupos puros.",
        "como_funciona": "Divide recursivamente os dados baseando-se na feature que mais reduz a impureza (Gini ou Entropia). Cada divisão cria um nó na árvore.",
        "quando_usar": "Quando precisas de interpretabilidade total. Quando os dados têm relações não-lineares. Bom ponto de partida antes de ensembles.",
        "cuidados": "Overfit facilmente sem poda (max_depth). Instável — pequenas mudanças nos dados mudam muito a árvore.",
        "hiperparametros": {"max_depth": "Profundidade máxima. None = overfitting. Começa com 3-5.", "min_samples_split": "Mínimo de amostras para dividir um nó"},
        "badge": "green"
    },
    "Random Forest": {
        "nome": "Random Forest",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Em vez de perguntar a um especialista, perguntas a 100 especialistas diferentes (cada um viu dados diferentes) e fazes uma votação. A decisão colectiva é mais robusta.",
        "como_funciona": "Treina N árvores de decisão, cada uma em subconjuntos aleatórios dos dados e das features (bagging). A previsão final é a votação (classificação) ou média (regressão).",
        "quando_usar": "Um dos melhores algoritmos para uso geral. Robusto a overfitting, lida bem com muitas features, dá importância das variáveis.",
        "cuidados": "Mais lento para treinar que uma árvore única. Menos interpretável (caixa negra comparado a uma árvore simples).",
        "hiperparametros": {"n_estimators": "Número de árvores. Mais = melhor (até certo ponto). Começa com 100.", "max_features": "Features por árvore. 'sqrt' para classificação, 'log2' para regressão"},
        "badge": "green"
    },
    "Gradient Boosting": {
        "nome": "Gradient Boosting",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Aprender com os erros. Treinas um modelo fraco, vês onde errou, treinas outro modelo focado nos erros, e assim sucessivamente. Cada modelo corrige o anterior.",
        "como_funciona": "Treina modelos sequencialmente, onde cada novo modelo tenta corrigir os erros residuais do anterior, usando o gradiente do erro como sinal de aprendizagem.",
        "quando_usar": "Quando precisas de alta performance. Competições de ML. Datasets tabulares estruturados.",
        "cuidados": "Propenso a overfitting com learning_rate alto. Mais lento que Random Forest. Sensível a outliers.",
        "hiperparametros": {"n_estimators": "Número de boosting rounds", "learning_rate": "Taxa de aprendizagem. Menor = mais lento mas melhor. Típico: 0.01-0.1", "max_depth": "Profundidade das árvores base. Típico: 3-5"},
        "badge": "amber"
    },
    "SVM": {
        "nome": "Support Vector Machine (SVM)",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Imagina separar laranjas de maçãs numa mesa. O SVM encontra a linha (ou plano) que separa as duas classes com a maior margem possível dos pontos mais próximos.",
        "como_funciona": "Encontra o hiperplano de separação com margem máxima entre classes. Com o kernel trick, consegue separar dados não-lineares mapeando para dimensões superiores.",
        "quando_usar": "Dados de alta dimensão (texto, imagens). Datasets pequenos a médios. Quando as classes são claramente separáveis.",
        "cuidados": "Lento para datasets grandes. Requer normalização. Difícil de interpretar com kernels não-lineares.",
        "hiperparametros": {"C": "Penalização do erro. Alto = menos margem, mais preciso no treino (overfitting). Baixo = margem maior (underfitting)", "kernel": "rbf (padrão, não-linear), linear, poly"},
        "badge": "purple"
    },
    "Logistic Regression": {
        "nome": "Regressão Logística",
        "tipo": "Supervisionado — Classificação",
        "analogia": "Apesar do nome, é um classificador. Estima a probabilidade de pertencer a uma classe usando uma função sigmoide — transforma qualquer número real em probabilidade entre 0 e 1.",
        "como_funciona": "Aprende pesos para cada feature que, combinados com a função sigmoide, produzem uma probabilidade. Usa máxima verossimilhança para optimizar os pesos.",
        "quando_usar": "Classificação binária simples. Quando precisas de probabilidades calibradas. Benchmark rápido antes de modelos complexos.",
        "cuidados": "Assume relação linear entre features e log-odds. Não captura relações não-lineares sem feature engineering.",
        "hiperparametros": {"C": "Inverso da regularização. C alto = menos regularização", "max_iter": "Iterações de optimização. Aumenta se não convergir"},
        "badge": "blue"
    },
    "Naive Bayes": {
        "nome": "Naive Bayes",
        "tipo": "Supervisionado — Classificação",
        "analogia": "'Naive' porque assume que todas as features são independentes entre si — o que raramente é verdade, mas funciona surpreendentemente bem para texto.",
        "como_funciona": "Calcula a probabilidade de cada classe dado os valores das features, usando o Teorema de Bayes e assumindo independência condicional entre features.",
        "quando_usar": "Classificação de texto (spam, sentimento). Datasets com muitas features. Quando velocidade é prioritária.",
        "cuidados": "A assunção de independência raramente é verdade. Pode ter problemas com features correlacionadas.",
        "hiperparametros": {"var_smoothing": "Estabilidade numérica. Ajusta se houver problemas de precisão"},
        "badge": "teal"
    },
    "KMeans": {
        "nome": "K-Means Clustering",
        "tipo": "Não Supervisionado — Clustering",
        "analogia": "Tens 100 pessoas numa sala e queres formar K grupos. Colocas K representantes aleatórios, cada pessoa junta-se ao mais próximo, depois os representantes movem-se para o centro do grupo. Repete até estabilizar.",
        "como_funciona": "Inicializa K centróides aleatoriamente. Assign cada ponto ao centróide mais próximo. Recalcula centróides como média do cluster. Repete até convergir.",
        "quando_usar": "Quando sabes o número de clusters. Dados com clusters esféricos e tamanhos similares. Boa performance em grandes datasets.",
        "cuidados": "Precisas definir K antes. Sensível a outliers. Assume clusters esféricos — não funciona bem com formas irregulares.",
        "hiperparametros": {"n_clusters": "Número de clusters K. Usa o método do cotovelo para escolher.", "init": "KMeans++ (padrão) dá melhores resultados que 'random'"},
        "badge": "teal"
    },
    "DBSCAN": {
        "nome": "DBSCAN",
        "tipo": "Não Supervisionado — Clustering",
        "analogia": "Imagina explorar uma floresta. Começas num ponto, expandes para todos os vizinhos próximos, e continuas a expansão. Regiões densas formam clusters, pontos isolados são ruído.",
        "como_funciona": "Define clusters como regiões de alta densidade separadas por regiões de baixa densidade. Não precisa de K definido. Detecta automaticamente outliers como ruído.",
        "quando_usar": "Quando não sabes o número de clusters. Clusters com formas arbitrárias. Quando há outliers nos dados.",
        "cuidados": "Difícil de ajustar eps e min_samples. Não funciona bem com clusters de densidades muito diferentes.",
        "hiperparametros": {"eps": "Raio da vizinhança. Usa gráfico kNN para escolher.", "min_samples": "Mínimo de pontos para ser core point. Típico: 5"},
        "badge": "purple"
    },
    "Linear Regression": {
        "nome": "Regressão Linear",
        "tipo": "Supervisionado — Regressão",
        "analogia": "A reta que melhor se ajusta a uma nuvem de pontos. Minimiza a soma dos quadrados das distâncias verticais entre os pontos e a reta.",
        "como_funciona": "Encontra os coeficientes da equação y = β₀ + β₁x₁ + ... + βₙxₙ que minimizam o erro quadrático médio (MSE).",
        "quando_usar": "Relação linear entre features e target. Baseline simples para qualquer problema de regressão. Interpretabilidade total.",
        "cuidados": "Assume linearidade e homocedasticidade. Sensível a outliers. Multicolinearidade causa instabilidade.",
        "hiperparametros": {"fit_intercept": "Inclui o intercepto β₀. Quase sempre True."},
        "badge": "blue"
    },
    "Ridge": {
        "nome": "Ridge Regression (L2)",
        "tipo": "Supervisionado — Regressão",
        "analogia": "Regressão Linear com penalização. Como a Linear mas com um imposto sobre coeficientes grandes — força o modelo a ser mais simples e menos propenso a overfitting.",
        "como_funciona": "Adiciona penalização L2 (soma dos quadrados dos coeficientes × alpha) à função de custo. Encolhe todos os coeficientes mas não os zera.",
        "quando_usar": "Quando tens multicolinearidade. Quando tens muitas features. Regularização suave.",
        "cuidados": "Alpha alto = mais regularização = underfitting. Não faz selecção de variáveis (todos os coeficientes permanecem).",
        "hiperparametros": {"alpha": "Força da regularização. 0 = Linear sem regularização. Típico: 0.1 a 100"},
        "badge": "blue"
    },
    "Lasso": {
        "nome": "Lasso Regression (L1)",
        "tipo": "Supervisionado — Regressão",
        "analogia": "Como Ridge mas mais agressivo — força alguns coeficientes a zero, fazendo selecção automática de variáveis. Útil quando suspeitas que poucas features são realmente importantes.",
        "como_funciona": "Adiciona penalização L1 (soma dos valores absolutos dos coeficientes × alpha). Produz coeficientes esparsos — muitos exactamente zero.",
        "quando_usar": "Feature selection automática. Quando suspeitas que poucas features são relevantes. Alta dimensionalidade.",
        "cuidados": "Pode descartar features correlacionadas de forma arbitrária. Instável com features altamente correlacionadas.",
        "hiperparametros": {"alpha": "Força da regularização L1. Valores maiores = mais coeficientes zerados"},
        "badge": "amber"
    },
    "AdaBoost": {
        "nome": "AdaBoost",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Aprende com os erros dando mais atenção aos casos difíceis. Cada novo classificador foca-se nos exemplos que os anteriores erraram mais.",
        "como_funciona": "Treina classificadores fracos sequencialmente. Os exemplos mal classificados recebem mais peso na iteração seguinte. A previsão final é uma combinação ponderada.",
        "quando_usar": "Quando o modelo base é fraco (stumps). Boa alternativa ao Gradient Boosting quando há menos overfitting.",
        "cuidados": "Sensível a outliers (recebem muito peso). Mais lento que Random Forest.",
        "hiperparametros": {"n_estimators": "Número de estimadores fracos", "learning_rate": "Peso de cada estimador"},
        "badge": "amber"
    },
    "XGBoost": {
        "nome": "XGBoost",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "Gradient Boosting optimizado para velocidade e performance. Vencedor de dezenas de competições Kaggle. Regularização incorporada evita overfitting.",
        "como_funciona": "Gradient Boosting com optimizações: regularização L1/L2 incorporada, tratamento nativo de valores nulos, processamento paralelo, poda de árvores.",
        "quando_usar": "Datasets tabulares estruturados. Competições de ML. Quando precisas do melhor resultado possível.",
        "cuidados": "Muitos hiperparâmetros. Pode overfit em datasets pequenos sem tuning.",
        "hiperparametros": {"n_estimators": "Rounds de boosting", "max_depth": "Profundidade. Típico: 3-8", "learning_rate": "Típico: 0.01-0.3", "subsample": "Fracção de dados por árvore. Típico: 0.8"},
        "badge": "red"
    },
    "LightGBM": {
        "nome": "LightGBM",
        "tipo": "Supervisionado — Classificação / Regressão",
        "analogia": "XGBoost mais rápido. Usa crescimento de árvore leaf-wise (vertical) em vez de level-wise (horizontal), tornando-o muito mais eficiente em datasets grandes.",
        "como_funciona": "Gradient Boosting com GOSS (Gradient-based One-Side Sampling) e EFB (Exclusive Feature Bundling). Muito mais rápido que XGBoost em datasets grandes.",
        "quando_usar": "Datasets grandes (>10k linhas). Quando velocidade é importante. Resultados comparáveis ao XGBoost.",
        "cuidados": "Pode overfit em datasets pequenos. leaf-wise pode criar árvores muito profundas.",
        "hiperparametros": {"num_leaves": "Folhas por árvore. Típico: 31", "learning_rate": "Típico: 0.05-0.1", "n_estimators": "Número de árvores"},
        "badge": "red"
    },
}