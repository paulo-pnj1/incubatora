"""
DataForge EDU — Utilitários partilhados
Paleta, CSS, componentes reutilizáveis
"""

import streamlit as st
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
import os
import datetime

# ══════════════════════════════════════════════════════
# PALETA DE CORES
# ══════════════════════════════════════════════════════
C_BG         = "#0F1117"
C_SURFACE    = "#161B27"
C_SURFACE2   = "#1E2535"
C_BORDER     = "#2A3347"
C_ACCENT     = "#4F8EF7"
C_ACCENT2    = "#7C5CBF"
C_GREEN      = "#2ECC71"
C_AMBER      = "#F39C12"
C_RED        = "#E74C3C"
C_TEAL       = "#1ABC9C"
C_TEXT       = "#E8EBF0"
C_TEXT_SEC   = "#9BA3B2"
C_TEXT_MUTE  = "#5C6478"

PALETTE = [C_ACCENT, C_TEAL, C_AMBER, C_RED, C_ACCENT2, C_GREEN]

# ══════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp, .main {{
        background-color: {C_BG} !important;
        color: {C_TEXT} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }}

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {{
        background: {C_SURFACE} !important;
        border-right: 1px solid {C_BORDER} !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding: 1.2rem 1rem 2rem;
    }}

    /* ── INPUTS ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {C_SURFACE2} !important;
        border: 1px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        border-radius: 8px !important;
    }}

    /* ── BOTÕES ── */
    .stButton > button {{
        background: {C_ACCENT} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.5rem 1.4rem !important;
        transition: opacity .2s !important;
    }}
    .stButton > button:hover {{ opacity: 0.85 !important; }}

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {C_SURFACE} !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid {C_BORDER} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: {C_TEXT_SEC} !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C_ACCENT} !important;
        color: #fff !important;
    }}

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {{
        background: {C_SURFACE2} !important;
        border-radius: 8px !important;
        border: 1px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        font-weight: 500 !important;
    }}

    /* ── MÉTRICAS ── */
    [data-testid="metric-container"] {{
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }}

    /* ── DATAFRAME ── */
    .stDataFrame {{ border: 1px solid {C_BORDER} !important; border-radius: 8px !important; }}

    /* ── SLIDER ── */
    .stSlider [data-baseweb="slider"] {{ padding: 0 !important; }}

    /* ── RADIO ── */
    div[data-testid="stRadio"] label[data-baseweb="radio"] {{
        padding: 10px 14px !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        transition: all .15s !important;
        color: {C_TEXT_SEC} !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
        background: {C_SURFACE2} !important;
        border-color: {C_ACCENT} !important;
        color: {C_ACCENT} !important;
    }}

    /* ── ALERTAS ── */
    .stAlert {{ border-radius: 10px !important; }}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {C_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 3px; }}

    /* ── COMPONENTES CUSTOM ── */
    .df-card {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .df-card-accent {{
        background: {C_SURFACE};
        border: 1px solid {C_ACCENT};
        border-left: 4px solid {C_ACCENT};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .df-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .03em;
    }}
    .df-badge-blue {{ background: rgba(79,142,247,.15); color: {C_ACCENT}; border: 1px solid rgba(79,142,247,.3); }}
    .df-badge-green {{ background: rgba(46,204,113,.15); color: {C_GREEN}; border: 1px solid rgba(46,204,113,.3); }}
    .df-badge-amber {{ background: rgba(243,156,18,.15); color: {C_AMBER}; border: 1px solid rgba(243,156,18,.3); }}
    .df-badge-purple {{ background: rgba(124,92,191,.15); color: {C_ACCENT2}; border: 1px solid rgba(124,92,191,.3); }}
    .df-badge-red {{ background: rgba(231,76,60,.15); color: {C_RED}; border: 1px solid rgba(231,76,60,.3); }}

    .section-title {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: {C_TEXT_MUTE};
        padding-bottom: 8px;
        border-bottom: 1px solid {C_BORDER};
        margin: 1.6rem 0 1rem;
    }}
    .page-title {{
        font-size: 26px;
        font-weight: 700;
        color: {C_TEXT};
        margin-bottom: 4px;
    }}
    .page-subtitle {{
        font-size: 15px;
        color: {C_TEXT_SEC};
        margin-bottom: 1.6rem;
    }}
    .teoria-box {{
        background: linear-gradient(135deg, rgba(79,142,247,.08), rgba(124,92,191,.08));
        border: 1px solid rgba(79,142,247,.25);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }}
    .teoria-box h4 {{
        color: {C_ACCENT};
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .teoria-box p, .teoria-box li {{
        color: {C_TEXT_SEC};
        font-size: 14px;
        line-height: 1.7;
    }}
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.2rem;
    }}
    .step-dot {{
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 700;
        flex-shrink: 0;
    }}
    .step-dot.active {{ background: {C_ACCENT}; color: #fff; }}
    .step-dot.done {{ background: {C_GREEN}; color: #fff; }}
    .step-dot.pending {{ background: {C_SURFACE2}; color: {C_TEXT_MUTE}; border: 1px solid {C_BORDER}; }}
    .progress-bar-wrap {{
        background: {C_SURFACE2};
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
        margin: 6px 0;
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_TEAL});
        transition: width .4s ease;
    }}
    .sidebar-brand {{
        display: flex; align-items: center; gap: 12px;
        padding: 8px 4px 16px;
        border-bottom: 1px solid {C_BORDER};
        margin-bottom: 16px;
    }}
    .sidebar-brand-icon {{
        width: 40px; height: 40px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }}
    .sidebar-brand-name {{
        font-size: 17px; font-weight: 700; color: {C_TEXT};
    }}
    .sidebar-brand-sub {{
        font-size: 11px; color: {C_TEXT_MUTE};
    }}
    .user-pill {{
        display: flex; align-items: center; gap: 8px;
        background: {C_SURFACE2};
        border: 1px solid {C_BORDER};
        border-radius: 20px;
        padding: 6px 12px;
        margin-bottom: 16px;
    }}
    .user-avatar {{
        width: 28px; height: 28px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700; color: #fff;
        flex-shrink: 0;
    }}
    .user-name {{ font-size: 13px; font-weight: 600; color: {C_TEXT}; }}
    .user-role {{ font-size: 11px; color: {C_TEXT_MUTE}; }}
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
    <div style="margin-bottom:1.6rem;">
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
    <div style="margin-bottom:.6rem;">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:{C_TEXT_MUTE};margin-bottom:4px;">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metricas_row(items: list):
    """items = [{"label": str, "valor": str, "delta": str, "cor": str}]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("valor", ""),
                delta=item.get("delta", None)
            )

def info_box(msg: str):
    st.markdown(f"""
    <div style="background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.3);
    border-radius:10px;padding:.8rem 1rem;color:{C_TEXT_SEC};font-size:14px;margin-bottom:.8rem;">
    ℹ️ {msg}</div>""", unsafe_allow_html=True)

def sucesso_box(msg: str):
    st.markdown(f"""
    <div style="background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.3);
    border-radius:10px;padding:.8rem 1rem;color:{C_TEXT_SEC};font-size:14px;margin-bottom:.8rem;">
    ✅ {msg}</div>""", unsafe_allow_html=True)

def aviso_box(msg: str):
    st.markdown(f"""
    <div style="background:rgba(243,156,18,.1);border:1px solid rgba(243,156,18,.3);
    border-radius:10px;padding:.8rem 1rem;color:{C_TEXT_SEC};font-size:14px;margin-bottom:.8rem;">
    ⚠️ {msg}</div>""", unsafe_allow_html=True)

def erro_box(msg: str):
    st.markdown(f"""
    <div style="background:rgba(231,76,60,.1);border:1px solid rgba(231,76,60,.3);
    border-radius:10px;padding:.8rem 1rem;color:{C_TEXT_SEC};font-size:14px;margin-bottom:.8rem;">
    ❌ {msg}</div>""", unsafe_allow_html=True)


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
        desc = "Dados dos passageiros do Titanic. Preve sobrevivência. Mix de features numéricas e categóricas."

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
        "analogia": "Baseado no Teorema de Bayes. 'Naive' porque assume que todas as features são independentes entre si — o que raramente é verdade, mas funciona surpreendentemente bem.",
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
