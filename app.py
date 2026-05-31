"""
================================================================
INCUBADORA ML — Universidade Kimpa Vita
Disciplina: Inteligência Artificial II · Capítulo III
Design: Acessível — WCAG 2.1 AA
Versão 2.0 — Sistema Geral + Visão Computacional
================================================================
Execute com:  streamlit run app.py
Dependências extras:
  pip install tensorflow pillow opencv-python-headless
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import pickle
import io
import os
from PIL import Image
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    silhouette_score, calinski_harabasz_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Incubadora ML — UKV",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# PALETA ACESSÍVEL — contraste mínimo 4.5:1 (WCAG AA)
# ══════════════════════════════════════════════════════════════
C_BG        = "#F5F7FA"
C_SURFACE   = "#FFFFFF"
C_BORDER    = "#C5CDD8"
C_BORDER_ST = "#7A90A8"
C_BLUE      = "#1A5FAD"
C_BLUE_LT   = "#E3EDF9"
C_BLUE_MID  = "#3C7FC0"
C_GREEN     = "#1A7A4A"
C_GREEN_LT  = "#E3F5EC"
C_AMBER     = "#7A5000"
C_AMBER_LT  = "#FDF3DC"
C_RED       = "#9B1C1C"
C_RED_LT    = "#FDEAEA"
C_TEXT      = "#1A2433"
C_TEXT_SEC  = "#3D5166"
C_TEXT_MUTE = "#5A7085"

PALETTE_CATS = [C_BLUE, C_GREEN, C_AMBER, C_RED, "#6B4FA0", "#A0522D"]

plt.rcParams.update({
    'figure.facecolor':  C_SURFACE,
    'axes.facecolor':    C_BG,
    'axes.edgecolor':    C_BORDER,
    'axes.labelcolor':   C_TEXT,
    'axes.titlecolor':   C_TEXT,
    'xtick.color':       C_TEXT_SEC,
    'ytick.color':       C_TEXT_SEC,
    'text.color':        C_TEXT,
    'grid.color':        C_BORDER,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.8,
    'legend.facecolor':  C_SURFACE,
    'legend.edgecolor':  C_BORDER,
    'legend.labelcolor': C_TEXT,
    'font.family':       'sans-serif',
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.2,
})

# ══════════════════════════════════════════════════════════════
# CSS ACESSÍVEL
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Source+Code+Pro:wght@400;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"],
.stApp, .main {{
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 17px !important;
    line-height: 1.65 !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}}

section[data-testid="stSidebar"] {{
    background: {C_SURFACE} !important;
    border-right: 2px solid {C_BORDER} !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 1.5rem 1rem 2rem;
}}

.brand-wrap {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}}
.brand-hex {{
    width: 40px; height: 40px; flex-shrink: 0;
    background: {C_BLUE};
    clip-path: polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 20px; font-weight: 700;
}}
.brand-name {{
    font-size: 20px; font-weight: 800;
    color: {C_TEXT}; letter-spacing: -0.01em;
    line-height: 1.1;
}}
.brand-sub {{
    font-size: 12px; color: {C_TEXT_MUTE};
    letter-spacing: 0.04em; line-height: 1.3;
}}

.sidebar-div {{
    border: none;
    border-top: 2px solid {C_BORDER};
    margin: 1rem 0;
}}

.nav-label {{
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.12em; color: {C_TEXT_MUTE};
    text-transform: uppercase;
    padding: 0 4px; margin-bottom: 6px;
}}

div[data-testid="stRadio"] > label {{ display: none !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    display: flex !important;
    align-items: center !important;
    padding: 11px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    cursor: pointer !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    color: {C_TEXT_SEC} !important;
    border: 2px solid transparent !important;
    transition: all .15s ease !important;
    min-height: 48px !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
    background: {C_BLUE_LT} !important;
    color: {C_BLUE} !important;
    border-color: {C_BLUE_MID} !important;
}}
div[data-testid="stRadio"] [aria-checked="true"] ~ div p {{
    color: {C_BLUE} !important;
    font-weight: 700 !important;
}}

.page-header {{
    border-bottom: 2px solid {C_BORDER};
    padding-bottom: 1.2rem;
    margin-bottom: 1.6rem;
}}
.page-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {C_BLUE};
    background: {C_BLUE_LT};
    border: 1.5px solid {C_BLUE_MID};
    border-radius: 6px;
    padding: 3px 12px;
    margin-bottom: 10px;
    text-transform: uppercase;
}}
.page-title {{
    font-size: 32px;
    font-weight: 800;
    color: {C_TEXT};
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0;
}}
.page-title span {{ color: {C_BLUE}; }}

.info-box {{
    background: {C_BLUE_LT};
    border: 1.5px solid {C_BLUE_MID};
    border-left: 4px solid {C_BLUE};
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 15px;
    color: {C_TEXT};
    margin: 10px 0 20px;
    line-height: 1.7;
}}
.info-box strong {{ color: {C_TEXT}; font-weight: 700; }}
.warn-box {{
    background: {C_AMBER_LT};
    border: 1.5px solid #D4A820;
    border-left: 4px solid {C_AMBER};
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 15px;
    color: {C_TEXT};
    margin: 10px 0;
    line-height: 1.7;
}}

.metric-row {{
    display: flex;
    gap: 10px;
    margin: 16px 0;
    flex-wrap: wrap;
}}
.metric-card {{
    flex: 1; min-width: 110px;
    background: {C_SURFACE};
    border: 1.5px solid {C_BORDER};
    border-top: 4px solid {C_BLUE};
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}}
.metric-label {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: {C_TEXT_SEC};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.metric-value {{
    font-size: 26px;
    font-weight: 800;
    color: {C_BLUE};
    line-height: 1;
}}
.metric-card.green {{ border-top-color: {C_GREEN}; }}
.metric-card.green .metric-value {{ color: {C_GREEN}; }}
.metric-card.amber {{ border-top-color: {C_AMBER}; }}
.metric-card.amber .metric-value {{ color: {C_AMBER}; }}
.metric-card.red {{ border-top-color: {C_RED}; }}
.metric-card.red .metric-value {{ color: {C_RED}; }}

.section-title {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {C_TEXT_SEC};
    text-transform: uppercase;
    margin: 1.6rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 2px;
    background: {C_BORDER};
}}

.stButton > button {{
    background: {C_BLUE} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    min-height: 48px !important;
    cursor: pointer !important;
    transition: background .15s ease !important;
    letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{
    background: #144D8C !important;
}}

[data-testid="stDownloadButton"] button {{
    background: {C_GREEN} !important;
    color: #fff !important;
    border: none !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    min-height: 44px !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploader"] {{
    background: {C_SURFACE} !important;
    border: 2px dashed {C_BLUE_MID} !important;
    border-radius: 10px !important;
    padding: 10px !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {C_BLUE} !important;
    background: {C_BLUE_LT} !important;
}}

[data-testid="stSelectbox"] > div > div {{
    background: {C_SURFACE} !important;
    border: 1.5px solid {C_BORDER_ST} !important;
    color: {C_TEXT} !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    min-height: 44px !important;
}}

[data-testid="stExpander"] {{
    background: {C_SURFACE} !important;
    border: 1.5px solid {C_BORDER} !important;
    border-radius: 10px !important;
    margin: 8px 0 !important;
}}

[data-testid="stDataFrame"] {{
    border: 1.5px solid {C_BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}

[data-testid="stProgress"] > div > div > div {{
    background: {C_BLUE} !important;
    height: 10px !important;
    border-radius: 5px !important;
}}

hr {{
    border: none !important;
    border-top: 2px solid {C_BORDER} !important;
    margin: 1.2rem 0 !important;
}}

.sidebar-footer {{
    margin-top: 2rem;
    font-size: 12px;
    color: {C_TEXT_MUTE};
    letter-spacing: 0.06em;
    border-top: 2px solid {C_BORDER};
    padding-top: 12px;
    line-height: 1.7;
}}

.pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 600;
}}
.pill-green {{
    background: {C_GREEN_LT};
    color: {C_GREEN};
    border: 1.5px solid {C_GREEN};
}}
.pill-red {{
    background: {C_RED_LT};
    color: {C_RED};
    border: 1.5px solid {C_RED};
}}

.cmp-badge {{
    background: {C_GREEN_LT};
    color: {C_GREEN};
    border: 1.5px solid {C_GREEN};
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    padding: 3px 10px;
    display: inline-block;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="brand-wrap">
        <div class="brand-hex">⬡</div>
        <div>
            <div class="brand-name">Incubadora ML</div>
            <div class="brand-sub">Univ. Kimpa Vita · IA II · Cap. III</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-div">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Módulos</div>', unsafe_allow_html=True)

    modulo = st.radio(
        "Navegar para",
        [
            "01 · Supervisionado",
            "02 · Não Supervisionado",
            "03 · Reforço Q-Learning",
            "04 · Agentes Songo",
            "05 · Treino Geral",
            "06 · Visão Computacional",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-div">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-footer">
        Carregue qualquer ficheiro CSV<br>
        ou imagens e explore os algoritmos.<br><br>
        Capítulo III · Aprendizagem Automática<br>
        v2.0 — Sistema Geral + Visão Comp.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════
def info(msg):
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-box">⚠ {msg}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def page_header(eyebrow, title, highlight=""):
    full = title.replace(highlight, f"<span>{highlight}</span>") if highlight else title
    st.markdown(f"""
    <div class="page-header">
        <div class="page-eyebrow">📌 {eyebrow}</div>
        <div class="page-title">{full}</div>
    </div>
    """, unsafe_allow_html=True)

def metrics_row(items):
    color_cycle = ["", "green", "amber", "red", "", "green"]
    html = '<div class="metric-row" role="list" aria-label="Métricas do modelo">'
    for i, (lbl, val) in enumerate(items):
        c = color_cycle[i % len(color_cycle)]
        html += f"""
        <div class="metric-card {c}" role="listitem">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def encode_dataframe(df):
    """Codifica automaticamente colunas categóricas e devolve df + dicionário de encoders."""
    df_enc = df.copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    return df_enc, le_dict

def plot_confusion_matrix(cm, label_names, title="Matriz de Confusão"):
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else C_TEXT)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [str(l)[:12] for l in label_names]
    ax.set_xticklabels(short, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# MÓDULO 01 — SUPERVISIONADO (CRÉDITO)
# ══════════════════════════════════════════════════════════════
if modulo == "01 · Supervisionado":
    page_header("Módulo 01", "Classificação de Crédito Bancário", "Crédito Bancário")

    info("Dataset: <strong>credito_bancario.csv</strong> — Idade, Pontuacao_Credito, "
         "Rendimento_Mensal, Anos_Historico, Divida_Total &nbsp;|&nbsp; Alvo: <strong>Aprovado</strong> (0 = Recusado, 1 = Aprovado)")

    uploaded = st.file_uploader(
        "Carregar credito_bancario.csv",
        type="csv", key="sup",
        help="Ficheiro CSV com os dados de crédito bancário"
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✔ Dataset carregado: {df.shape[0]} amostras × {df.shape[1]} colunas")

        with st.expander("📊 Explorar dados"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Primeiras linhas do dataset**")
                st.dataframe(df.head(8), use_container_width=True)
            with c2:
                st.markdown("**Estatísticas descritivas**")
                st.dataframe(df.describe().round(2), use_container_width=True)

            vc = df['Aprovado'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(
                ['Recusado (0)', 'Aprovado (1)'],
                vc.reindex([0, 1]).fillna(0).values,
                color=[C_RED, C_GREEN], edgecolor='none', width=0.45
            )
            for bar, val in zip(bars, vc.reindex([0, 1]).fillna(0).values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(int(val)), ha='center', va='bottom', fontsize=13, fontweight='bold', color=C_TEXT)
            ax.set_title("Distribuição da variável alvo", fontsize=14, pad=10, fontweight='bold')
            ax.set_ylabel("Número de amostras")
            ax.grid(axis='y', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        section("Configurar treino")
        ca, cb, cc = st.columns(3)
        modelo_sel = ca.selectbox(
            "Modelo de classificação",
            ["KNN", "Árvore de Decisão", "Random Forest", "SVM"],
        )
        test_size = cb.slider("Percentagem de dados de teste (%)", 10, 40, 30) / 100
        with cc:
            if modelo_sel == "KNN":
                k_val = st.slider("K vizinhos mais próximos", 1, 21, 5, step=2)
            elif modelo_sel == "Árvore de Decisão":
                depth_val = st.slider("Profundidade máxima da árvore", 1, 15, 5)
            elif modelo_sel == "Random Forest":
                n_trees  = st.slider("Número de árvores", 10, 200, 100, step=10)
                depth_rf = st.slider("Profundidade máxima", 1, 15, 5)
            else:
                c_val = st.select_slider("Parâmetro C", options=[0.01, 0.1, 1.0, 10.0], value=1.0)

        if st.button("▶ Treinar modelo", key="btn_sup"):
            with st.spinner("A treinar o modelo..."):
                X = df.drop('Aprovado', axis=1)
                y = df['Aprovado']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_train)
                X_te_sc = scaler.transform(X_test)

                if modelo_sel == "KNN":
                    model = KNeighborsClassifier(n_neighbors=k_val, metric='euclidean')
                    model.fit(X_tr_sc, y_train)
                    y_pred = model.predict(X_te_sc)
                    y_prob = model.predict_proba(X_te_sc)[:, 1]
                elif modelo_sel == "Árvore de Decisão":
                    model = DecisionTreeClassifier(max_depth=depth_val, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif modelo_sel == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth_rf, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    model = SVC(kernel='rbf', C=c_val, probability=True, random_state=42)
                    model.fit(X_tr_sc, y_train)
                    y_pred = model.predict(X_te_sc)
                    y_prob = model.predict_proba(X_te_sc)[:, 1]

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec  = recall_score(y_test, y_pred, zero_division=0)
                f1   = f1_score(y_test, y_pred, zero_division=0)
                auc  = roc_auc_score(y_test, y_prob)

            section("Métricas de avaliação")
            metrics_row([
                ("Acurácia",  f"{acc:.4f}"),
                ("Precisão",  f"{prec:.4f}"),
                ("Recall",    f"{rec:.4f}"),
                ("F1-Score",  f"{f1:.4f}"),
                ("AUC-ROC",   f"{auc:.4f}"),
            ])

            col_cm, col_roc = st.columns(2)
            with col_cm:
                section("Matriz de confusão")
                cm = confusion_matrix(y_test, y_pred)
                fig = plot_confusion_matrix(cm, ['Recusado', 'Aprovado'], modelo_sel)
                st.pyplot(fig); plt.close()

            with col_roc:
                section("Curva ROC")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.fill_between(fpr, tpr, alpha=0.12, color=C_BLUE)
                ax.plot(fpr, tpr, color=C_BLUE, lw=2.5, label=f'AUC = {auc:.3f}')
                ax.plot([0, 1], [0, 1], color=C_TEXT_MUTE, lw=1.5, linestyle='--', label='Aleatório')
                ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
                ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
                ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            if modelo_sel in ["Random Forest", "Árvore de Decisão"]:
                section("Importância das variáveis")
                feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
                fig, ax = plt.subplots(figsize=(7, 3))
                colors = [C_BLUE if v >= feat_imp.max() * 0.6 else C_BLUE_MID for v in feat_imp.values]
                ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='none', height=0.55)
                for v, lbl in zip(feat_imp.values, feat_imp.index):
                    ax.text(v + 0.002, lbl, f'{v:.3f}', va='center', fontsize=12, color=C_TEXT)
                ax.set_xlabel("Importância Gini")
                ax.set_title("Importância das Variáveis", fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.4); fig.tight_layout(); st.pyplot(fig); plt.close()

            model_bytes = pickle.dumps(model)
            st.download_button("⬇ Descarregar modelo (.pkl)", model_bytes,
                               f"modelo_credito_{modelo_sel.lower().replace(' ','_')}.pkl",
                               mime="application/octet-stream")
    else:
        warn("Carregue o ficheiro <strong>credito_bancario.csv</strong> para começar.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 02 — NÃO SUPERVISIONADO
# ══════════════════════════════════════════════════════════════
elif modulo == "02 · Não Supervisionado":
    page_header("Módulo 02", "K-Means — Segmentação de Clientes", "K-Means")

    info("Dataset: <strong>clientes_ecommerce.csv</strong> — Idade, Gasto_Mensal, "
         "Frequencia_Compras, Tempo_Plataforma_h &nbsp;|&nbsp; Aprendizagem <strong>não supervisionada</strong>")

    uploaded = st.file_uploader("Carregar clientes_ecommerce.csv", type="csv", key="unsup")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✔ {df.shape[0]} clientes × {df.shape[1]} variáveis carregados")

        with st.expander("📊 Explorar dados"):
            st.dataframe(df.head(8), use_container_width=True)
            st.dataframe(df.describe().round(2), use_container_width=True)

        section("Configurar K-Means")
        c1, c2, c3 = st.columns(3)
        k_clusters = c1.slider("Número de clusters (K)", 2, 8, 3)
        eixo_x = c2.selectbox("Variável eixo X", df.columns.tolist(), index=1)
        eixo_y = c3.selectbox("Variável eixo Y", df.columns.tolist(), index=0)

        if st.button("▶ Executar K-Means", key="btn_km"):
            with st.spinner("A executar K-Means..."):
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(df.select_dtypes(include=np.number))
                inertias, sil_scores = [], []
                for k in range(2, 9):
                    kmt = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmt.fit(X_sc)
                    inertias.append(kmt.inertia_)
                    sil_scores.append(silhouette_score(X_sc, kmt.labels_))
                km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X_sc)
                df['Cluster'] = labels
                sil = silhouette_score(X_sc, labels)
                ch  = calinski_harabasz_score(X_sc, labels)

            section("Resultados")
            metrics_row([
                ("Clusters K",       str(k_clusters)),
                ("Silhouette Score", f"{sil:.4f}"),
                ("Calinski-Harabasz",f"{ch:.1f}"),
            ])

            col_el, col_sil = st.columns(2)
            with col_el:
                section("Método do cotovelo")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(range(2, 9), inertias, color=C_BLUE, lw=2.5, marker='o',
                        markersize=8, markerfacecolor=C_SURFACE, markeredgewidth=2)
                ax.axvline(k_clusters, color=C_AMBER, lw=2, linestyle='--', label=f'K = {k_clusters}')
                ax.set_xlabel("K"); ax.set_ylabel("Inércia")
                ax.set_title("Método do Cotovelo", fontsize=14, fontweight='bold')
                ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            with col_sil:
                section("Silhouette por K")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(range(2, 9), sil_scores, color=C_GREEN, lw=2.5, marker='o',
                        markersize=8, markerfacecolor=C_SURFACE, markeredgewidth=2)
                ax.axvline(k_clusters, color=C_AMBER, lw=2, linestyle='--', label=f'K = {k_clusters}')
                ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
                ax.set_title("Silhouette por K", fontsize=14, fontweight='bold')
                ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            section("Dispersão dos clusters")
            fig, ax = plt.subplots(figsize=(8, 5))
            for c_idx in range(k_clusters):
                mask = df['Cluster'] == c_idx
                ax.scatter(df.loc[mask, eixo_x], df.loc[mask, eixo_y],
                           color=PALETTE_CATS[c_idx % len(PALETTE_CATS)],
                           alpha=0.75, s=60, edgecolors='none',
                           label=f'Cluster {c_idx} (n={mask.sum()})')
            ax.set_xlabel(eixo_x); ax.set_ylabel(eixo_y)
            ax.set_title(f"K-Means — K={k_clusters}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

            section("Perfil médio por cluster")
            profile = df.groupby('Cluster').mean().round(2)
            st.dataframe(profile, use_container_width=True)
    else:
        warn("Carregue o ficheiro <strong>clientes_ecommerce.csv</strong> para começar.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 03 — Q-LEARNING
# ══════════════════════════════════════════════════════════════
elif modulo == "03 · Reforço Q-Learning":
    page_header("Módulo 03", "Q-Learning — Grid World 5×5", "Q-Learning")

    info("Sem ficheiro CSV necessário. Ambiente Grid World pré-definido. "
         "Ajuste os hiperparâmetros e observe o agente aprender.")

    st.markdown(f"""
    <div style="background:{C_SURFACE};border:1.5px solid {C_BORDER};border-radius:10px;
                padding:18px 22px;font-size:16px;color:{C_TEXT};line-height:2.0;
                margin:12px 0 20px;font-family:'Source Code Pro',monospace;">
    <strong style="color:{C_BLUE};font-size:17px;">MAPA DO LABIRINTO (5×5)</strong><br>
    <span style="color:{C_BLUE};font-weight:700;">S</span> · · <span style="color:{C_RED};font-weight:700;">X</span> ·
    &emsp; S = Início (0,0)<br>
    · <span style="color:{C_RED};font-weight:700;">X</span> · · ·
    &emsp; G = Destino (4,4) — recompensa +100<br>
    · · · <span style="color:{C_RED};font-weight:700;">X</span> ·
    &emsp; X = Obstáculo — penalidade −10<br>
    · <span style="color:{C_RED};font-weight:700;">X</span> · · ·
    &emsp; · = Livre — penalidade −1<br>
    · · · · <span style="color:{C_GREEN};font-weight:700;">G</span>
    </div>
    """, unsafe_allow_html=True)

    section("Hiperparâmetros do agente")
    c1, c2, c3, c4 = st.columns(4)
    n_ep  = c1.slider("Episódios", 100, 2000, 500, step=100)
    alpha = c2.slider("α — aprendizagem", 0.01, 1.0, 0.1, step=0.01)
    gamma = c3.slider("γ — desconto", 0.5, 1.0, 0.95, step=0.05)
    eps_i = c4.slider("ε inicial", 0.5, 1.0, 1.0, step=0.1)

    if st.button("▶ Treinar agente Q-Learning", key="btn_ql"):
        GRID_ROWS, GRID_COLS = 5, 5
        START = (0, 0); GOAL = (4, 4)
        OBSTACLES = [(0, 3), (1, 1), (2, 3), (3, 1)]
        ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def is_valid(r, c):
            return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and (r, c) not in OBSTACLES

        def step_env(state, action):
            r, c = state; dr, dc = ACTIONS[action]
            nr, nc = r + dr, c + dc
            if not is_valid(nr, nc): return state, -1
            ns = (nr, nc)
            if ns == GOAL: return ns, 100
            return ns, -1

        Q = np.zeros((GRID_ROWS, GRID_COLS, 4))
        eps = eps_i; eps_min = 0.01
        eps_decay = (eps - eps_min) / n_ep
        rewards_hist, steps_hist = [], []
        prog = st.progress(0, text="A treinar o agente...")

        for ep in range(n_ep):
            state = START; total_r = 0; n_steps = 0; done = False
            while not done and n_steps < 200:
                r, c = state
                a = np.random.randint(4) if np.random.random() < eps else np.argmax(Q[r, c])
                ns, reward = step_env(state, a)
                nr, nc = ns
                Q[r, c, a] += alpha * (reward + gamma * np.max(Q[nr, nc]) - Q[r, c, a])
                state = ns; total_r += reward; n_steps += 1
                if state == GOAL: done = True
            eps = max(eps_min, eps - eps_decay)
            rewards_hist.append(total_r); steps_hist.append(n_steps)
            if ep % 50 == 0:
                prog.progress((ep + 1) / n_ep, text=f"Episódio {ep+1}/{n_ep}...")

        prog.progress(1.0, text="Treino concluído ✔")

        avg_last  = np.mean(rewards_hist[-50:])
        goal_rate = sum(1 for s in steps_hist[-100:] if s < 200) / min(100, len(steps_hist))

        section("Resultados do treino")
        metrics_row([
            ("Episódios", str(n_ep)),
            ("Recomp. média (últ.50)", f"{avg_last:.1f}"),
            ("Taxa de sucesso", f"{goal_rate:.0%}"),
        ])

        win = max(1, n_ep // 20)
        col_rw, col_st = st.columns(2)
        with col_rw:
            section("Recompensa por episódio")
            sm = pd.Series(rewards_hist).rolling(win).mean()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(rewards_hist, alpha=0.2, color=C_BLUE, lw=1)
            ax.plot(sm, color=C_BLUE, lw=2.5, label=f'Média ({win} ep.)')
            ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa total")
            ax.set_title("Curva de Aprendizagem", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col_st:
            section("Passos por episódio")
            sm_s = pd.Series(steps_hist).rolling(win).mean()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(steps_hist, alpha=0.2, color=C_RED, lw=1)
            ax.plot(sm_s, color=C_RED, lw=2.5, label='Média móvel')
            ax.set_xlabel("Episódio"); ax.set_ylabel("Passos")
            ax.set_title("Eficiência do Agente", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        section("Melhor rota aprendida — política greedy")
        state = START; path = [state]; vis = {state}
        for _ in range(50):
            r, c = state
            a = np.argmax(Q[r, c])
            ns, _ = step_env(state, a)
            if ns in vis: break
            path.append(ns); vis.add(ns); state = ns
            if state == GOAL: break

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        ax.set_title(f"Rota óptima — {len(path)-1} passos", fontsize=14, fontweight='bold')
        for r in range(5):
            for c in range(5):
                if (r, c) in OBSTACLES:
                    fc, ec, lbl, lc = C_RED_LT, C_RED, '✕', C_RED
                elif (r, c) == GOAL:
                    fc, ec, lbl, lc = C_GREEN_LT, C_GREEN, 'G', C_GREEN
                elif (r, c) == START:
                    fc, ec, lbl, lc = C_BLUE_LT, C_BLUE, 'S', C_BLUE
                else:
                    fc, ec, lbl, lc = C_SURFACE, C_BORDER, '', C_TEXT
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                     facecolor=fc, edgecolor=ec, lw=2, zorder=1)
                ax.add_patch(rect)
                if lbl:
                    ax.text(c, r, lbl, ha='center', va='center',
                            fontsize=18, fontweight='bold', color=lc, zorder=2)
        if len(path) > 1:
            px = [p[1] for p in path]; py = [p[0] for p in path]
            ax.plot(px, py, 'o-', color=C_AMBER, lw=3, markersize=10,
                    markerfacecolor=C_AMBER, zorder=3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        if path[-1] == GOAL:
            st.success(f"✔ Agente encontrou o destino em {len(path)-1} passos!")
        else:
            st.warning("⚠ O agente não convergiu. Tente aumentar os episódios.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 04 — AGENTES SONGO
# ══════════════════════════════════════════════════════════════
elif modulo == "04 · Agentes Songo":
    page_header("Módulo 04", "Segurança no Trânsito — Município do Songo", "Songo")

    info("Dataset: <strong>inquerito_songo_500.csv</strong> — Inquérito sobre IA e Segurança no Trânsito. "
         "Variável alvo: <strong>IA_Pode_Reduzir_Acidentes</strong>")

    uploaded = st.file_uploader("Carregar inquerito_songo_500.csv", type="csv", key="songo")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✔ {df.shape[0]} registos × {df.shape[1]} colunas carregados")
        TARGET = 'IA_Pode_Reduzir_Acidentes'

        if TARGET not in df.columns:
            st.error(f"❌ Coluna '{TARGET}' não encontrada.")
        else:
            with st.expander("📊 Explorar dados"):
                st.dataframe(df.head(8), use_container_width=True)
                vc = df[TARGET].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(vc.index.astype(str), vc.values,
                        color=PALETTE_CATS[:len(vc)], edgecolor='none', height=0.5)
                for i, v in enumerate(vc.values):
                    ax.text(v + 1, i, str(v), va='center', fontsize=13, fontweight='bold', color=C_TEXT)
                ax.set_title(f"Distribuição — {TARGET}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Respostas"); ax.grid(axis='x', alpha=0.4)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            section("Configurar modelo")
            c1, c2 = st.columns(2)
            modelo_s = c1.selectbox("Modelo", ["Random Forest", "Árvore de Decisão", "Regressão Logística"])
            test_size_s = c2.slider("Percentagem de teste (%)", 10, 40, 30, key="ts_s") / 100

            if st.button("▶ Treinar modelo Songo", key="btn_songo"):
                with st.spinner("A preparar e treinar o modelo..."):
                    df_enc, le_dict = encode_dataframe(df)
                    drop_cols = (['ID'] if 'ID' in df_enc.columns else []) + [TARGET]
                    X = df_enc.drop(columns=drop_cols)
                    y = df_enc[TARGET]
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size_s, random_state=42, stratify=y)

                    if modelo_s == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                        model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
                    elif modelo_s == "Árvore de Decisão":
                        model = DecisionTreeClassifier(max_depth=6, random_state=42)
                        model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
                    else:
                        sc_s = StandardScaler()
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(sc_s.fit_transform(X_tr), y_tr)
                        y_pred = model.predict(sc_s.transform(X_te))

                    acc  = accuracy_score(y_te, y_pred)
                    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
                    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
                    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)

                section("Métricas")
                metrics_row([("Acurácia", f"{acc:.4f}"), ("F1-Score", f"{f1:.4f}"),
                              ("Precisão", f"{prec:.4f}"), ("Recall", f"{rec:.4f}")])

                labels_u = sorted(y.unique())
                le_t = le_dict.get(TARGET)
                label_names = le_t.classes_.tolist() if le_t else [str(l) for l in labels_u]
                cm = confusion_matrix(y_te, y_pred, labels=labels_u)
                fig = plot_confusion_matrix(cm, label_names, modelo_s)
                st.pyplot(fig); plt.close()

                if hasattr(model, 'feature_importances_'):
                    section("Importância das variáveis — top 10")
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(10)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(fi.index, fi.values, color=C_BLUE, edgecolor='none', height=0.6)
                    ax.set_xlabel("Importância Gini")
                    ax.set_title("Top 10 Variáveis", fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.4); fig.tight_layout(); st.pyplot(fig); plt.close()

                model_bytes = pickle.dumps(model)
                st.download_button("⬇ Descarregar modelo (.pkl)", model_bytes,
                                   f"modelo_songo_{modelo_s.replace(' ','_').lower()}.pkl",
                                   mime="application/octet-stream")
    else:
        warn("Carregue o ficheiro <strong>inquerito_songo_500.csv</strong> para começar.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 05 — TREINO GERAL (QUALQUER CSV)
# ══════════════════════════════════════════════════════════════
elif modulo == "05 · Treino Geral":
    page_header("Módulo 05", "Treino Geral — Qualquer Dataset", "Qualquer Dataset")

    info("Carregue <strong>qualquer ficheiro CSV</strong>. O sistema detecta automaticamente "
         "as colunas, o tipo de problema (classificação / regressão / clustering) "
         "e oferece todos os algoritmos disponíveis.")

    uploaded = st.file_uploader(
        "Carregar qualquer ficheiro CSV",
        type="csv", key="general",
        help="Ficheiro CSV com dados numéricos e/ou categóricos"
    )

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✔ {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas carregadas")

        with st.expander("📊 Explorar dados"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Primeiras linhas**")
                st.dataframe(df_raw.head(10), use_container_width=True)
            with c2:
                st.markdown("**Tipos e valores em falta**")
                info_df = pd.DataFrame({
                    'Tipo': df_raw.dtypes,
                    'Nulos': df_raw.isnull().sum(),
                    'Únicos': df_raw.nunique()
                })
                st.dataframe(info_df, use_container_width=True)
            st.markdown("**Estatísticas descritivas**")
            st.dataframe(df_raw.describe(include='all').round(3), use_container_width=True)

        st.markdown("---")
        section("Configurar problema")

        task_type = st.radio(
            "Tipo de tarefa",
            ["🔵 Classificação", "🟢 Regressão", "🟠 Clustering (não supervisionado)"],
            horizontal=True
        )

        # ── PRÉ-PROCESSAMENTO ──────────────────────────────────
        section("Pré-processamento")
        pc1, pc2, pc3 = st.columns(3)

        # Estratégia de valores em falta
        fill_strategy = pc1.selectbox(
            "Valores em falta",
            ["Média (numérico) / Moda (cat.)", "Mediana / Moda", "Remover linhas com nulos"]
        )

        # Scaler
        scaler_choice = pc2.selectbox(
            "Normalização / Escala",
            ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "Nenhum"]
        )

        # Remover colunas de ID
        cols_to_drop_user = pc3.multiselect(
            "Colunas a remover (IDs, etc.)",
            df_raw.columns.tolist()
        )

        if "🔵" in task_type or "🟢" in task_type:
            # ── COLUNA ALVO ────────────────────────────────────
            target_col = st.selectbox(
                "Coluna alvo (variável dependente / label)",
                [c for c in df_raw.columns if c not in cols_to_drop_user]
            )

        # ── ALGORITMOS ────────────────────────────────────────
        section("Seleccionar algoritmo")

        if "🔵" in task_type:
            algo = st.selectbox("Algoritmo de classificação", [
                "Random Forest",
                "Árvore de Decisão",
                "Gradient Boosting",
                "AdaBoost",
                "KNN",
                "Regressão Logística",
                "SVM (RBF)",
                "Naive Bayes",
            ])
        elif "🟢" in task_type:
            algo = st.selectbox("Algoritmo de regressão", [
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "Regressão Linear",
                "Ridge Regression",
                "Lasso Regression",
                "KNN Regressor",
                "Árvore de Decisão Regressor",
            ])
        else:
            algo = st.selectbox("Algoritmo de clustering", [
                "K-Means",
                "DBSCAN",
                "Agglomerative Clustering",
            ])

        # ── HIPERPARÂMETROS DINÂMICOS ──────────────────────────
        section("Hiperparâmetros")
        ha, hb, hc = st.columns(3)

        hyper = {}
        if algo in ["Random Forest", "Random Forest Regressor"]:
            hyper['n_estimators'] = ha.slider("Nº de árvores", 10, 500, 100, step=10)
            hyper['max_depth'] = hb.slider("Profundidade máx.", 1, 30, 8)
            hyper['min_samples_split'] = hc.slider("Min. amostras split", 2, 20, 2)
        elif algo in ["Gradient Boosting", "Gradient Boosting Regressor"]:
            hyper['n_estimators'] = ha.slider("Nº de estimadores", 50, 500, 100, step=50)
            hyper['learning_rate'] = hb.select_slider("Taxa de aprendizagem", [0.01,0.05,0.1,0.2,0.5], value=0.1)
            hyper['max_depth'] = hc.slider("Profundidade máx.", 1, 10, 3)
        elif algo in ["AdaBoost"]:
            hyper['n_estimators'] = ha.slider("Nº de estimadores", 10, 300, 50, step=10)
            hyper['learning_rate'] = hb.select_slider("Taxa de aprendizagem", [0.01,0.1,0.5,1.0], value=1.0)
        elif algo in ["KNN", "KNN Regressor"]:
            hyper['n_neighbors'] = ha.slider("K vizinhos", 1, 31, 5, step=2)
            hyper['metric'] = hb.selectbox("Métrica", ['euclidean','manhattan','minkowski'])
            hyper['weights'] = hc.selectbox("Pesos", ['uniform','distance'])
        elif algo in ["Árvore de Decisão", "Árvore de Decisão Regressor"]:
            hyper['max_depth'] = ha.slider("Profundidade máx.", 1, 20, 5)
            hyper['criterion'] = hb.selectbox("Critério", ['gini','entropy'] if "🔵" in task_type else ['squared_error','friedman_mse'])
        elif algo == "SVM (RBF)":
            hyper['C'] = ha.select_slider("C (regularização)", [0.01,0.1,1.0,10.0,100.0], value=1.0)
            hyper['gamma'] = hb.selectbox("Gamma", ['scale','auto'])
        elif algo == "Regressão Logística":
            hyper['C'] = ha.select_slider("C", [0.01,0.1,1.0,10.0], value=1.0)
            hyper['max_iter'] = hb.slider("Max iterações", 100, 2000, 1000, step=100)
        elif algo in ["Ridge Regression"]:
            hyper['alpha'] = ha.select_slider("Alpha", [0.01,0.1,1.0,10.0,100.0], value=1.0)
        elif algo in ["Lasso Regression"]:
            hyper['alpha'] = ha.select_slider("Alpha", [0.001,0.01,0.1,1.0,10.0], value=0.1)
        elif algo == "K-Means":
            hyper['n_clusters'] = ha.slider("K clusters", 2, 15, 3)
            hyper['n_init'] = hb.slider("N inicializações", 5, 30, 10)
        elif algo == "DBSCAN":
            hyper['eps'] = ha.slider("Epsilon (eps)", 0.1, 5.0, 0.5, step=0.1)
            hyper['min_samples'] = hb.slider("Min. amostras", 2, 20, 5)
        elif algo == "Agglomerative Clustering":
            hyper['n_clusters'] = ha.slider("K clusters", 2, 15, 3)
            hyper['linkage'] = hb.selectbox("Ligação", ['ward','complete','average','single'])

        # ── SPLIT ────────────────────────────────────────────
        if "🟠" not in task_type:
            test_size_g = st.slider("Percentagem de dados de teste (%)", 10, 40, 25, key="ts_g") / 100
            cv_folds    = st.slider("Folds validação cruzada", 3, 10, 5, key="cv_g")

        # ── TREINAR ──────────────────────────────────────────
        if st.button("▶ Treinar / Executar", key="btn_general"):
            with st.spinner("A pré-processar e treinar..."):

                # --- Cópia + remover colunas
                df_work = df_raw.drop(columns=cols_to_drop_user, errors='ignore').copy()

                # --- Valores em falta
                for col in df_work.columns:
                    if df_work[col].isnull().any():
                        if fill_strategy == "Remover linhas com nulos":
                            df_work = df_work.dropna()
                            break
                        elif df_work[col].dtype == object:
                            df_work[col].fillna(df_work[col].mode()[0], inplace=True)
                        else:
                            if "Mediana" in fill_strategy:
                                df_work[col].fillna(df_work[col].median(), inplace=True)
                            else:
                                df_work[col].fillna(df_work[col].mean(), inplace=True)

                # --- Codificação
                df_enc, le_dict = encode_dataframe(df_work)

                # --- CLUSTERING ─────────────────────────────
                if "🟠" in task_type:
                    X_cl = df_enc.values
                    if scaler_choice != "Nenhum":
                        sc = StandardScaler() if "Standard" in scaler_choice else MinMaxScaler()
                        X_cl = sc.fit_transform(X_cl)

                    if algo == "K-Means":
                        model_cl = KMeans(**hyper, random_state=42)
                    elif algo == "DBSCAN":
                        model_cl = DBSCAN(**hyper)
                    else:
                        model_cl = AgglomerativeClustering(**hyper)

                    labels_cl = model_cl.fit_predict(X_cl)
                    df_work['Cluster'] = labels_cl
                    n_clusters_found = len(set(labels_cl)) - (1 if -1 in labels_cl else 0)

                    try:
                        sil = silhouette_score(X_cl, labels_cl)
                    except:
                        sil = float('nan')
                    try:
                        ch = calinski_harabasz_score(X_cl, labels_cl)
                    except:
                        ch = float('nan')

                    section("Resultados do Clustering")
                    metrics_row([
                        ("Clusters encontrados", str(n_clusters_found)),
                        ("Silhouette Score",     f"{sil:.4f}" if not np.isnan(sil) else "N/A"),
                        ("Calinski-Harabasz",    f"{ch:.1f}"  if not np.isnan(ch)  else "N/A"),
                        ("Total de amostras",    str(len(labels_cl))),
                    ])

                    section("Distribuição de clusters")
                    vc_cl = pd.Series(labels_cl).value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.bar([f"Cluster {k}" for k in vc_cl.index], vc_cl.values,
                           color=PALETTE_CATS[:len(vc_cl)], edgecolor='none', width=0.5)
                    for i, v in enumerate(vc_cl.values):
                        ax.text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold', color=C_TEXT)
                    ax.set_title("Distribuição por Cluster", fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.4); fig.tight_layout(); st.pyplot(fig); plt.close()

                    if X_cl.shape[1] >= 2:
                        section("Projecção PCA 2D")
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_cl)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        for c_idx in sorted(set(labels_cl)):
                            mask = labels_cl == c_idx
                            lbl = f"Ruído" if c_idx == -1 else f"Cluster {c_idx}"
                            color = C_TEXT_MUTE if c_idx == -1 else PALETTE_CATS[c_idx % len(PALETTE_CATS)]
                            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color,
                                       alpha=0.7, s=50, edgecolors='none', label=lbl)
                        ve = pca.explained_variance_ratio_
                        ax.set_xlabel(f"PC1 — {ve[0]*100:.1f}%"); ax.set_ylabel(f"PC2 — {ve[1]*100:.1f}%")
                        ax.set_title("Clusters — Projecção PCA", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    section("Perfil médio por cluster")
                    numeric_cols = df_work.select_dtypes(include=np.number).columns.tolist()
                    if 'Cluster' in numeric_cols:
                        numeric_cols.remove('Cluster')
                    profile_cl = df_work.groupby('Cluster')[numeric_cols].mean().round(3)
                    st.dataframe(profile_cl, use_container_width=True)

                # --- CLASSIFICAÇÃO / REGRESSÃO ──────────────
                else:
                    target_col_enc = target_col
                    X = df_enc.drop(columns=[target_col_enc], errors='ignore')
                    y = df_enc[target_col_enc]

                    is_regression = "🟢" in task_type
                    stratify_y = y if not is_regression and y.nunique() <= 20 else None

                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size_g, random_state=42, stratify=stratify_y)

                    if scaler_choice != "Nenhum":
                        sc = StandardScaler() if "Standard" in scaler_choice else MinMaxScaler()
                        X_tr = sc.fit_transform(X_tr)
                        X_te = sc.transform(X_te)

                    # Instanciar modelo
                    if algo == "Random Forest":
                        model = RandomForestClassifier(**hyper, random_state=42, n_jobs=-1)
                    elif algo == "Random Forest Regressor":
                        model = RandomForestClassifier(**{k:v for k,v in hyper.items()}, random_state=42, n_jobs=-1)
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(**hyper, random_state=42, n_jobs=-1)
                    elif algo == "Gradient Boosting":
                        model = GradientBoostingClassifier(**hyper, random_state=42)
                    elif algo == "Gradient Boosting Regressor":
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(**hyper, random_state=42)
                    elif algo == "AdaBoost":
                        model = AdaBoostClassifier(**hyper, random_state=42)
                    elif algo == "KNN":
                        model = KNeighborsClassifier(**hyper)
                    elif algo == "KNN Regressor":
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(**{k:v for k,v in hyper.items() if k!='weights' or True})
                    elif algo == "Árvore de Decisão":
                        model = DecisionTreeClassifier(**hyper, random_state=42)
                    elif algo == "Árvore de Decisão Regressor":
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(**hyper, random_state=42)
                    elif algo == "Regressão Logística":
                        model = LogisticRegression(**hyper, random_state=42)
                    elif algo == "SVM (RBF)":
                        model = SVC(**hyper, probability=True, random_state=42)
                    elif algo == "Naive Bayes":
                        model = GaussianNB()
                    elif algo == "Regressão Linear":
                        model = LinearRegression()
                    elif algo == "Ridge Regression":
                        model = Ridge(**hyper)
                    elif algo == "Lasso Regression":
                        model = Lasso(**hyper)

                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)

                    # ── MÉTRICAS ─────────────────────────────
                    if is_regression:
                        mse  = mean_squared_error(y_te, y_pred)
                        rmse = np.sqrt(mse)
                        mae  = mean_absolute_error(y_te, y_pred)
                        r2   = r2_score(y_te, y_pred)

                        section("Métricas de Regressão")
                        metrics_row([
                            ("R² Score",  f"{r2:.4f}"),
                            ("RMSE",      f"{rmse:.4f}"),
                            ("MAE",       f"{mae:.4f}"),
                            ("MSE",       f"{mse:.4f}"),
                        ])

                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            section("Real vs Previsto")
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.scatter(y_te, y_pred, alpha=0.5, color=C_BLUE, s=30, edgecolors='none')
                            mn = min(y_te.min(), y_pred.min()); mx = max(y_te.max(), y_pred.max())
                            ax.plot([mn, mx], [mn, mx], color=C_RED, lw=2, linestyle='--', label='Ideal')
                            ax.set_xlabel("Valor Real"); ax.set_ylabel("Valor Previsto")
                            ax.set_title("Real vs Previsto", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                            fig.tight_layout(); st.pyplot(fig); plt.close()

                        with col_r2:
                            section("Distribuição dos Resíduos")
                            residuals = y_te - y_pred
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.hist(residuals, bins=30, color=C_BLUE, edgecolor=C_SURFACE, alpha=0.85)
                            ax.axvline(0, color=C_RED, lw=2, linestyle='--')
                            ax.set_xlabel("Resíduo (Real − Previsto)"); ax.set_ylabel("Frequência")
                            ax.set_title("Distribuição dos Resíduos", fontsize=14, fontweight='bold')
                            ax.grid(axis='y', alpha=0.4); fig.tight_layout(); st.pyplot(fig); plt.close()

                        # Cross-val para regressão
                        section(f"Validação cruzada — {cv_folds} folds (R²)")
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')
                        fig, ax = plt.subplots(figsize=(7, 2.8))
                        bar_col = [C_BLUE if s >= cv_scores.mean() else C_BLUE_MID for s in cv_scores]
                        ax.bar(range(1, cv_folds+1), cv_scores, color=bar_col, width=0.5, edgecolor='none')
                        ax.axhline(cv_scores.mean(), color=C_AMBER, lw=2, linestyle='--',
                                   label=f'Média = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
                        for i, s in enumerate(cv_scores):
                            ax.text(i+1, s+0.01, f'{s:.3f}', ha='center', fontsize=11, fontweight='bold', color=C_TEXT)
                        ax.set_xlabel("Fold"); ax.set_ylabel("R²")
                        ax.set_title("Validação Cruzada", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    else:
                        # CLASSIFICAÇÃO
                        acc  = accuracy_score(y_te, y_pred)
                        f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
                        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
                        rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)

                        section("Métricas de Classificação")
                        metrics_row([
                            ("Acurácia", f"{acc:.4f}"),
                            ("F1-Score", f"{f1:.4f}"),
                            ("Precisão", f"{prec:.4f}"),
                            ("Recall",   f"{rec:.4f}"),
                        ])

                        # Matriz de confusão
                        labels_u = sorted(y.unique())
                        le_t = le_dict.get(target_col)
                        label_names_g = le_t.classes_.tolist() if le_t else [str(l) for l in labels_u]
                        cm = confusion_matrix(y_te, y_pred, labels=labels_u)

                        col_cm_g, col_report = st.columns(2)
                        with col_cm_g:
                            section("Matriz de Confusão")
                            fig = plot_confusion_matrix(cm, label_names_g, algo)
                            st.pyplot(fig); plt.close()
                        with col_report:
                            section("Relatório de Classificação")
                            report = classification_report(y_te, y_pred, target_names=label_names_g,
                                                           output_dict=True, zero_division=0)
                            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

                        # ROC se binário
                        if y.nunique() == 2 and hasattr(model, 'predict_proba'):
                            try:
                                y_prob_g = model.predict_proba(X_te)[:, 1]
                                auc = roc_auc_score(y_te, y_prob_g)
                                fpr, tpr, _ = roc_curve(y_te, y_prob_g)
                                section("Curva ROC")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.fill_between(fpr, tpr, alpha=0.12, color=C_BLUE)
                                ax.plot(fpr, tpr, color=C_BLUE, lw=2.5, label=f'AUC = {auc:.3f}')
                                ax.plot([0,1],[0,1], color=C_TEXT_MUTE, lw=1.5, linestyle='--', label='Aleatório')
                                ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
                                ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
                                ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                                fig.tight_layout(); st.pyplot(fig); plt.close()
                            except:
                                pass

                        # Cross-val classificação
                        section(f"Validação cruzada — {cv_folds} folds (Acurácia)")
                        skf_g = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if y.nunique() <= 20 else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        try:
                            cv_scores_g = cross_val_score(model, X_tr, y_tr, cv=skf_g, scoring='accuracy')
                            fig, ax = plt.subplots(figsize=(7, 2.8))
                            bar_col = [C_BLUE if s >= cv_scores_g.mean() else C_BLUE_MID for s in cv_scores_g]
                            ax.bar(range(1, cv_folds+1), cv_scores_g, color=bar_col, width=0.5, edgecolor='none')
                            ax.axhline(cv_scores_g.mean(), color=C_AMBER, lw=2, linestyle='--',
                                       label=f'Média = {cv_scores_g.mean():.4f} ± {cv_scores_g.std():.4f}')
                            for i, s in enumerate(cv_scores_g):
                                ax.text(i+1, s+0.005, f'{s:.3f}', ha='center', fontsize=11, fontweight='bold', color=C_TEXT)
                            ax.set_ylim(0, 1.1)
                            ax.set_xlabel("Fold"); ax.set_ylabel("Acurácia")
                            ax.set_title("Validação Cruzada", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.4)
                            fig.tight_layout(); st.pyplot(fig); plt.close()
                        except Exception as e:
                            st.warning(f"Validação cruzada não disponível: {e}")

                    # Importância de variáveis
                    if hasattr(model, 'feature_importances_'):
                        section("Importância das variáveis — Top 15")
                        feat_names = X.columns if hasattr(X, 'columns') else [f'var_{i}' for i in range(X.shape[1])]
                        fi = pd.Series(model.feature_importances_, index=feat_names).sort_values().tail(15)
                        fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.4)))
                        colors_fi = [C_BLUE if v >= fi.max() * 0.6 else C_BLUE_MID for v in fi.values]
                        ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', height=0.65)
                        for v, lbl in zip(fi.values, fi.index):
                            ax.text(v + 0.001, lbl, f'{v:.3f}', va='center', fontsize=11, color=C_TEXT)
                        ax.set_xlabel("Importância")
                        ax.set_title("Top 15 Variáveis Mais Importantes", fontsize=14, fontweight='bold')
                        ax.grid(axis='x', alpha=0.4); fig.tight_layout(); st.pyplot(fig); plt.close()

                    # Download
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        "⬇ Descarregar modelo treinado (.pkl)", model_bytes,
                        f"modelo_geral_{algo.lower().replace(' ','_')}.pkl",
                        mime="application/octet-stream"
                    )

                    section("Previsão interactiva — nova amostra")
                    feat_cols = X.columns if hasattr(X, 'columns') else [f'var_{i}' for i in range(X.shape[1])]
                    # Mostrar em grid de 4 colunas
                    user_inputs = {}
                    cols_input = st.columns(min(4, len(feat_cols)))
                    for idx, col_name in enumerate(feat_cols):
                        col_widget = cols_input[idx % len(cols_input)]
                        orig_col = df_raw[col_name] if col_name in df_raw.columns else None
                        if orig_col is not None and orig_col.dtype == object:
                            options = orig_col.dropna().unique().tolist()
                            val_sel = col_widget.selectbox(col_name, options, key=f"inp_{col_name}")
                            le_c = le_dict.get(col_name)
                            if le_c:
                                try:
                                    val_enc = le_c.transform([val_sel])[0]
                                except:
                                    val_enc = 0
                            else:
                                val_enc = 0
                            user_inputs[col_name] = val_enc
                        else:
                            min_v = float(df_raw[col_name].min()) if col_name in df_raw.columns else 0.0
                            max_v = float(df_raw[col_name].max()) if col_name in df_raw.columns else 100.0
                            med_v = float(df_raw[col_name].mean()) if col_name in df_raw.columns else 0.0
                            user_inputs[col_name] = col_widget.number_input(
                                col_name, min_value=min_v, max_value=max_v, value=med_v, key=f"inp_{col_name}")

                    if st.button("▶ Prever nova amostra", key="btn_predict_new"):
                        try:
                            nova = np.array([[user_inputs[c] for c in feat_cols]])
                            if scaler_choice != "Nenhum":
                                nova = sc.transform(nova)
                            pred_new = model.predict(nova)[0]
                            if not is_regression and le_dict.get(target_col):
                                pred_label = le_dict[target_col].inverse_transform([int(pred_new)])[0]
                            else:
                                pred_label = pred_new
                            if is_regression:
                                st.success(f"📊 Previsão: **{pred_label:.4f}**")
                            else:
                                st.success(f"🏷️ Classe prevista: **{pred_label}**")
                                if hasattr(model, 'predict_proba'):
                                    probas = model.predict_proba(nova)[0]
                                    le_t2 = le_dict.get(target_col)
                                    class_names = le_t2.classes_.tolist() if le_t2 else [str(i) for i in range(len(probas))]
                                    prob_df = pd.DataFrame({'Classe': class_names, 'Probabilidade': probas}).sort_values('Probabilidade', ascending=False)
                                    fig, ax = plt.subplots(figsize=(6, 2.5))
                                    ax.barh(prob_df['Classe'], prob_df['Probabilidade'],
                                            color=PALETTE_CATS[:len(prob_df)], edgecolor='none', height=0.5)
                                    ax.set_xlim(0, 1); ax.set_xlabel("Probabilidade")
                                    ax.set_title("Probabilidades por classe", fontsize=13, fontweight='bold')
                                    fig.tight_layout(); st.pyplot(fig); plt.close()
                        except Exception as ex:
                            st.error(f"Erro na previsão: {ex}")
    else:
        warn("Carregue <strong>qualquer ficheiro CSV</strong> para começar. "
             "O módulo suporta classificação, regressão e clustering com detecção automática.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 06 — VISÃO COMPUTACIONAL
# ══════════════════════════════════════════════════════════════
elif modulo == "06 · Visão Computacional":
    page_header("Módulo 06", "Visão Computacional — CNN com TensorFlow/Keras", "CNN")

    info("Treine uma <strong>Rede Neuronal Convolucional (CNN)</strong> com as suas próprias imagens. "
         "Carregue imagens organizadas por classes, configure a arquitectura e treine o modelo. "
         "Suporta qualquer número de classes.")

    # Verificar se TensorFlow está disponível
    tf_available = False
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.utils import to_categorical
        tf_available = True
    except ImportError:
        st.error("⚠️ TensorFlow não está instalado. Execute: `pip install tensorflow pillow`")

    if tf_available:
        st.markdown(f"""
        <div style="background:{C_SURFACE};border:1.5px solid {C_BORDER};border-radius:10px;
                    padding:16px 20px;margin:10px 0 20px;font-size:15px;color:{C_TEXT_SEC};">
        <strong style="color:{C_TEXT};">Como usar este módulo:</strong><br>
        1️⃣ &nbsp;Carregue imagens (JPG/PNG) de <strong>várias classes</strong> — o nome do ficheiro deve conter o nome da classe (ex: <code>gato_01.jpg</code>, <code>cao_05.png</code>)<br>
        2️⃣ &nbsp;Ou organize por separador de classe manual abaixo.<br>
        3️⃣ &nbsp;Configure a arquitectura CNN e hiperparâmetros.<br>
        4️⃣ &nbsp;Treine e analise os resultados.<br>
        5️⃣ &nbsp;Use a secção de inferência para classificar novas imagens.
        </div>
        """, unsafe_allow_html=True)

        # ── Configuração das classes ───────────────────────────
        section("1 · Definir classes e carregar imagens")

        num_classes_input = st.number_input(
            "Número de classes", min_value=2, max_value=20, value=2, step=1
        )

        class_names_input = []
        class_images = {}
        cols_cls = st.columns(min(4, int(num_classes_input)))

        for i in range(int(num_classes_input)):
            col_c = cols_cls[i % len(cols_cls)]
            cls_name = col_c.text_input(f"Nome da classe {i+1}", value=f"classe_{i+1}", key=f"cls_{i}")
            class_names_input.append(cls_name)
            imgs = col_c.file_uploader(
                f"Imagens — {cls_name}",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                key=f"imgs_{i}"
            )
            class_images[cls_name] = imgs if imgs else []

        # Contar total de imagens
        total_imgs = sum(len(v) for v in class_images.values())
        classes_with_data = [c for c, imgs in class_images.items() if len(imgs) > 0]

        if total_imgs > 0:
            st.success(f"✔ {total_imgs} imagens carregadas em {len(classes_with_data)} classe(s): "
                       + ", ".join([f"{c} ({len(class_images[c])})" for c in classes_with_data]))

            # ── Configurar CNN ─────────────────────────────────
            section("2 · Configurar arquitectura CNN")

            ca1, ca2, ca3 = st.columns(3)
            img_size   = ca1.select_slider("Tamanho da imagem (px)", [32, 48, 64, 96, 128, 224], value=64)
            epochs     = ca2.slider("Épocas de treino", 3, 100, 15, step=1)
            batch_size = ca3.select_slider("Batch size", [8, 16, 32, 64, 128], value=16)

            cb1, cb2, cb3 = st.columns(3)
            lr         = cb1.select_slider("Taxa de aprendizagem", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            dropout    = cb2.slider("Dropout (regularização)", 0.0, 0.7, 0.3, step=0.05)
            test_split = cb3.slider("% Teste", 10, 40, 20) / 100

            section("Arquitectura das camadas convolucionais")
            cc1, cc2, cc3 = st.columns(3)
            n_conv_blocks = cc1.slider("Nº de blocos convolucionais", 1, 4, 2)
            filters_base  = cc2.select_slider("Filtros no 1º bloco", [16, 32, 64, 128], value=32)
            dense_units   = cc3.select_slider("Nós na camada densa", [64, 128, 256, 512], value=128)

            augment = st.checkbox("Activar Data Augmentation (flip, rotação, zoom)", value=True)

            section("3 · Treinar modelo CNN")

            if st.button("▶ Treinar CNN", key="btn_cnn"):
                with st.spinner("A carregar e pré-processar imagens..."):
                    X_all, y_all = [], []
                    label_map = {cls: idx for idx, cls in enumerate(classes_with_data)}

                    for cls_name in classes_with_data:
                        for img_file in class_images[cls_name]:
                            try:
                                pil_img = Image.open(img_file).convert('RGB')
                                pil_img = pil_img.resize((img_size, img_size))
                                arr = img_to_array(pil_img) / 255.0
                                X_all.append(arr)
                                y_all.append(label_map[cls_name])
                            except Exception as e:
                                st.warning(f"Erro ao carregar {img_file.name}: {e}")

                X_arr = np.array(X_all, dtype=np.float32)
                y_arr = np.array(y_all, dtype=np.int32)
                n_classes_actual = len(classes_with_data)

                st.success(f"✔ {len(X_arr)} imagens processadas — shape: {X_arr.shape}")

                if len(X_arr) < 4:
                    st.error("❌ São necessárias pelo menos 4 imagens para treinar.")
                else:
                    # Split treino/teste
                    X_tr_cnn, X_te_cnn, y_tr_cnn, y_te_cnn = train_test_split(
                        X_arr, y_arr, test_size=test_split, random_state=42,
                        stratify=y_arr if len(set(y_arr)) > 1 else None
                    )

                    y_tr_cat = to_categorical(y_tr_cnn, n_classes_actual)
                    y_te_cat = to_categorical(y_te_cnn, n_classes_actual)

                    # Construir modelo CNN
                    model_cnn = models.Sequential()

                    # Data augmentation integrada
                    if augment:
                        model_cnn.add(layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)))
                        model_cnn.add(layers.RandomRotation(0.1))
                        model_cnn.add(layers.RandomZoom(0.1))
                        first_conv_input = False
                    else:
                        first_conv_input = True

                    # Blocos convolucionais
                    for block in range(n_conv_blocks):
                        n_filters = filters_base * (2 ** block)
                        if block == 0 and first_conv_input:
                            model_cnn.add(layers.Conv2D(n_filters, (3, 3), activation='relu',
                                                         padding='same',
                                                         input_shape=(img_size, img_size, 3)))
                        else:
                            model_cnn.add(layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same'))
                        model_cnn.add(layers.BatchNormalization())
                        model_cnn.add(layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same'))
                        model_cnn.add(layers.MaxPooling2D((2, 2)))
                        model_cnn.add(layers.Dropout(dropout * 0.5))

                    model_cnn.add(layers.GlobalAveragePooling2D())
                    model_cnn.add(layers.Dense(dense_units, activation='relu'))
                    model_cnn.add(layers.Dropout(dropout))

                    if n_classes_actual == 2:
                        model_cnn.add(layers.Dense(1, activation='sigmoid'))
                        loss_fn = 'binary_crossentropy'
                        y_tr_fit = y_tr_cnn
                        y_te_fit = y_te_cnn
                    else:
                        model_cnn.add(layers.Dense(n_classes_actual, activation='softmax'))
                        loss_fn = 'categorical_crossentropy'
                        y_tr_fit = y_tr_cat
                        y_te_fit = y_te_cat

                    model_cnn.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss=loss_fn,
                        metrics=['accuracy']
                    )

                    # Resumo do modelo
                    with st.expander("🔎 Arquitectura do modelo CNN"):
                        summary_lines = []
                        model_cnn.summary(print_fn=lambda x: summary_lines.append(x))
                        st.code('\n'.join(summary_lines), language='text')

                    # Treino com progresso
                    prog_cnn = st.progress(0, text="A treinar CNN...")
                    history_data = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

                    # Callback personalizado para barra de progresso
                    class StreamlitCallback(keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            prog_cnn.progress((epoch + 1) / epochs,
                                              text=f"Época {epoch+1}/{epochs} — "
                                                   f"loss: {logs.get('loss',0):.4f} — "
                                                   f"acc: {logs.get('accuracy',0):.4f} — "
                                                   f"val_acc: {logs.get('val_accuracy',0):.4f}")
                            for k in history_data:
                                if k in logs:
                                    history_data[k].append(logs[k])

                    early_stop = keras.callbacks.EarlyStopping(
                        monitor='val_accuracy', patience=10, restore_best_weights=True)
                    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

                    hist = model_cnn.fit(
                        X_tr_cnn, y_tr_fit,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_te_cnn, y_te_fit),
                        callbacks=[StreamlitCallback(), early_stop, reduce_lr],
                        verbose=0
                    )
                    prog_cnn.progress(1.0, text="Treino concluído ✔")

                    # Avaliar no conjunto de teste
                    eval_results = model_cnn.evaluate(X_te_cnn, y_te_fit, verbose=0)
                    test_loss    = eval_results[0]
                    test_acc     = eval_results[1]

                    # Previsões
                    preds_raw = model_cnn.predict(X_te_cnn, verbose=0)
                    if n_classes_actual == 2:
                        y_pred_cnn = (preds_raw[:, 0] > 0.5).astype(int)
                    else:
                        y_pred_cnn = np.argmax(preds_raw, axis=1)

                    f1_cnn  = f1_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)
                    prec_cnn= precision_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)
                    rec_cnn = recall_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)

                    section("Métricas finais")
                    metrics_row([
                        ("Acurácia Teste", f"{test_acc:.4f}"),
                        ("F1-Score",       f"{f1_cnn:.4f}"),
                        ("Precisão",       f"{prec_cnn:.4f}"),
                        ("Recall",         f"{rec_cnn:.4f}"),
                        ("Loss Teste",     f"{test_loss:.4f}"),
                    ])

                    # Curvas de aprendizagem
                    section("Curvas de aprendizagem")
                    col_loss, col_acc = st.columns(2)
                    h = hist.history

                    with col_loss:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(h['loss'], color=C_BLUE, lw=2, label='Treino')
                        ax.plot(h['val_loss'], color=C_RED, lw=2, linestyle='--', label='Validação')
                        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
                        ax.set_title("Curva de Loss", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    with col_acc:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(h['accuracy'], color=C_GREEN, lw=2, label='Treino')
                        ax.plot(h['val_accuracy'], color=C_AMBER, lw=2, linestyle='--', label='Validação')
                        ax.set_xlabel("Época"); ax.set_ylabel("Acurácia")
                        ax.set_title("Curva de Acurácia", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    # Matriz de confusão CNN
                    section("Matriz de confusão")
                    cm_cnn = confusion_matrix(y_te_cnn, y_pred_cnn)
                    fig = plot_confusion_matrix(cm_cnn, classes_with_data, "CNN")
                    st.pyplot(fig); plt.close()

                    # Relatório
                    section("Relatório de classificação")
                    report_cnn = classification_report(y_te_cnn, y_pred_cnn,
                                                        target_names=classes_with_data,
                                                        output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report_cnn).T.round(3), use_container_width=True)

                    # Visualizar predições
                    section("Amostras do conjunto de teste com predições")
                    n_show = min(12, len(X_te_cnn))
                    indices_show = np.random.choice(len(X_te_cnn), n_show, replace=False)
                    n_cols_show = min(6, n_show)
                    n_rows_show = (n_show + n_cols_show - 1) // n_cols_show
                    fig, axes = plt.subplots(n_rows_show, n_cols_show,
                                             figsize=(n_cols_show * 2.5, n_rows_show * 2.8))
                    if n_rows_show == 1 and n_cols_show == 1:
                        axes = np.array([[axes]])
                    elif n_rows_show == 1:
                        axes = axes[np.newaxis, :]
                    elif n_cols_show == 1:
                        axes = axes[:, np.newaxis]

                    for idx_plot, idx_data in enumerate(indices_show):
                        r_p = idx_plot // n_cols_show
                        c_p = idx_plot % n_cols_show
                        ax = axes[r_p, c_p]
                        ax.imshow(X_te_cnn[idx_data])
                        true_cls  = classes_with_data[y_te_cnn[idx_data]]
                        pred_cls  = classes_with_data[y_pred_cnn[idx_data]]
                        correct   = true_cls == pred_cls
                        color_border = C_GREEN if correct else C_RED
                        for spine in ax.spines.values():
                            spine.set_edgecolor(color_border); spine.set_linewidth(3)
                        ax.set_title(f"Real: {true_cls}\nPrev: {pred_cls}",
                                     fontsize=9,
                                     color=C_GREEN if correct else C_RED,
                                     fontweight='bold')
                        ax.axis('off')

                    # Ocultar axes extras
                    for idx_extra in range(n_show, n_rows_show * n_cols_show):
                        r_p = idx_extra // n_cols_show
                        c_p = idx_extra % n_cols_show
                        axes[r_p, c_p].axis('off')

                    fig.suptitle("Predições no conjunto de teste (🟩 Correcto | 🟥 Errado)",
                                 fontsize=13, fontweight='bold', color=C_TEXT)
                    fig.tight_layout(); st.pyplot(fig); plt.close()

                    # Guardar modelo
                    st.session_state['cnn_model']       = model_cnn
                    st.session_state['cnn_classes']     = classes_with_data
                    st.session_state['cnn_img_size']    = img_size

                    # Download modelo keras
                    try:
                        model_path_cnn = "/tmp/modelo_cnn.keras"
                        model_cnn.save(model_path_cnn)
                        with open(model_path_cnn, 'rb') as f:
                            st.download_button("⬇ Descarregar modelo CNN (.keras)", f.read(),
                                               "modelo_cnn.keras", mime="application/octet-stream")
                    except Exception:
                        pass

            # ── INFERÊNCIA ────────────────────────────────────
            st.markdown("---")
            section("4 · Inferência — Classificar nova imagem")

            if 'cnn_model' in st.session_state:
                inf_img = st.file_uploader(
                    "Carregar imagem para classificar",
                    type=["jpg","jpeg","png","bmp","webp"],
                    key="inf_img"
                )
                if inf_img:
                    m_inf   = st.session_state['cnn_model']
                    cls_inf = st.session_state['cnn_classes']
                    sz_inf  = st.session_state['cnn_img_size']

                    pil_inf = Image.open(inf_img).convert('RGB')
                    pil_rs  = pil_inf.resize((sz_inf, sz_inf))
                    arr_inf = img_to_array(pil_rs) / 255.0
                    arr_inf = np.expand_dims(arr_inf, axis=0)

                    preds_inf = m_inf.predict(arr_inf, verbose=0)
                    if len(cls_inf) == 2:
                        prob_inf = float(preds_inf[0][0])
                        pred_idx = int(prob_inf > 0.5)
                        probas_inf = [1 - prob_inf, prob_inf]
                    else:
                        probas_inf = preds_inf[0].tolist()
                        pred_idx   = int(np.argmax(probas_inf))

                    col_img_inf, col_res_inf = st.columns([1, 2])
                    with col_img_inf:
                        st.image(pil_inf, caption="Imagem carregada", use_container_width=True)
                    with col_res_inf:
                        st.markdown(f"""
                        <div style="background:{C_GREEN_LT};border:2px solid {C_GREEN};
                                    border-radius:10px;padding:16px 20px;margin:10px 0;">
                            <div style="font-size:14px;color:{C_GREEN};font-weight:700;letter-spacing:0.08em;
                                        text-transform:uppercase;margin-bottom:6px;">Resultado</div>
                            <div style="font-size:28px;font-weight:800;color:{C_TEXT};">
                                🏷️ {cls_inf[pred_idx]}
                            </div>
                            <div style="font-size:15px;color:{C_TEXT_SEC};margin-top:4px;">
                                Confiança: {probas_inf[pred_idx]*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        section("Probabilidades por classe")
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        colors_p = [C_GREEN if i == pred_idx else C_BLUE_MID for i in range(len(cls_inf))]
                        ax.barh(cls_inf, probas_inf, color=colors_p, edgecolor='none', height=0.5)
                        for i, v in enumerate(probas_inf):
                            ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
                        ax.set_xlim(0, 1.15)
                        ax.set_xlabel("Probabilidade")
                        ax.set_title("Distribuição de probabilidades", fontsize=12, fontweight='bold')
                        ax.grid(axis='x', alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()
            else:
                warn("Treine primeiro o modelo CNN (passo 3) para activar a inferência.")

        else:
            warn("Carregue imagens nas classes definidas acima para continuar.")