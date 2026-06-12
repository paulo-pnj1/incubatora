"""
DataForge EDU — Módulo de Autenticação
Login, registo, perfil do utilizador
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from modules.utils import (
    inject_css, C_ACCENT, C_SURFACE, C_BORDER, C_TEXT, C_TEXT_SEC,
    C_TEXT_MUTE, C_BG, C_SURFACE2, C_GREEN, C_AMBER, get_user_progress, badge
)

CONFIG_PATH = "config.yaml"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        _create_default_config()
    with open(CONFIG_PATH) as f:
        return yaml.load(f, Loader=SafeLoader)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _create_default_config():
    import bcrypt
    def hash_pw(pw):
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

    config = {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@dataforge.ao",
                    "name": "Administrador",
                    "password": hash_pw("admin123"),
                    "role": "admin"
                },
                "professor": {
                    "email": "prof@dataforge.ao",
                    "name": "Professor Demo",
                    "password": hash_pw("prof123"),
                    "role": "professor"
                },
                "aluno": {
                    "email": "aluno@dataforge.ao",
                    "name": "Aluno Demo",
                    "password": hash_pw("aluno123"),
                    "role": "aluno"
                }
            }
        },
        "cookie": {
            "expiry_days": 7,
            "key": "dataforge_edu_2025",
            "name": "dataforge_session"
        },
        "preauthorized": {"emails": []}
    }
    save_config(config)


def get_authenticator():
    config = load_config()
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    ), config


def render_login_page():
    """Renderiza a página de login com design personalizado"""
    inject_css()

    # Cabeçalho — HTML completo e fechado, sem widgets dentro
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:2.4rem;">
        <div style="width:72px;height:72px;background:linear-gradient(135deg,{C_ACCENT},{C_SURFACE2});
        border-radius:20px;display:flex;align-items:center;justify-content:center;
        font-size:36px;margin:0 auto 1rem;border:1px solid {C_BORDER};">🧠</div>
        <div style="font-size:30px;font-weight:800;color:{C_TEXT};letter-spacing:-.5px;">DataForge EDU</div>
        <div style="font-size:15px;color:{C_TEXT_MUTE};margin-top:6px;">
            Plataforma de Aprendizagem de Machine Learning
        </div>
    </div>
    <div style="text-align:center;margin-bottom:1.2rem;">
        <span style="font-size:18px;font-weight:700;color:{C_TEXT};">Entrar na plataforma</span>
    </div>
    """, unsafe_allow_html=True)

    # Formulário de login — widgets Streamlit directamente, sem HTML em volta
    authenticator, config = get_authenticator()

    # Centralizar o formulário com colunas
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            authenticator.login(location="main", key="login_form")
        except Exception:
            authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    username    = st.session_state.get("username")

    # Mensagens de estado — HTML completo e fechado
    if auth_status is False:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background:rgba(231,76,60,.1);border:1px solid rgba(231,76,60,.3);
            border-radius:8px;padding:.7rem 1rem;color:#E74C3C;font-size:14px;margin-top:.8rem;">
            ❌ Utilizador ou palavra-passe incorrectos.</div>
            """, unsafe_allow_html=True)

    if auth_status is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="font-size:13px;color:{C_TEXT_MUTE};text-align:center;margin-top:1rem;">
            Conta demo: <code>aluno</code> / <code>aluno123</code></div>
            """, unsafe_allow_html=True)

    # Recursos da plataforma — HTML completo e fechado
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;
    max-width:400px;margin:1.6rem auto 0;">
        <div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:12px;
        padding:.8rem;text-align:center;">
            <div style="font-size:22px;">📚</div>
            <div style="font-size:11px;color:{C_TEXT_MUTE};margin-top:4px;">Teoria + Prática</div>
        </div>
        <div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:12px;
        padding:.8rem;text-align:center;">
            <div style="font-size:22px;">🏆</div>
            <div style="font-size:11px;color:{C_TEXT_MUTE};margin-top:4px;">Certificados</div>
        </div>
        <div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:12px;
        padding:.8rem;text-align:center;">
            <div style="font-size:22px;">🤖</div>
            <div style="font-size:11px;color:{C_TEXT_MUTE};margin-top:4px;">50+ Algoritmos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return authenticator, auth_status, username, config


def render_sidebar_user(name: str, username: str, role: str, authenticator):
    """Renderiza o bloco do utilizador na sidebar"""
    initial = name[0].upper() if name else "U"
    role_labels = {"admin": ("Admin", "red"), "professor": ("Professor", "amber"), "aluno": ("Aluno", "blue")}
    role_label, role_color = role_labels.get(role, ("Utilizador", "blue"))

    progress = get_user_progress(username)
    pontos = progress.get("pontos", 0)

    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-name">DataForge EDU</div>
            <div class="sidebar-brand-sub">Machine Learning</div>
        </div>
    </div>

    <div class="user-pill">
        <div class="user-avatar">{initial}</div>
        <div style="flex:1;min-width:0;">
            <div class="user-name" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
            <div class="user-role">{badge(role_label, role_color)} &nbsp;⭐ {pontos} pts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sair", key="logout_btn", use_container_width=True):
        authenticator.logout(location="unrendered")
        st.rerun()


def get_user_role(username: str, config: dict) -> str:
    try:
        return config["credentials"]["usernames"][username].get("role", "aluno")
    except Exception:
        return "aluno"


def registar_utilizador_demo():
    """Form de registo rápido (para demo/extensão futura)"""
    with st.expander("➕ Criar nova conta"):
        st.markdown(f'<div style="font-size:13px;color:{C_TEXT_SEC};">Registo disponível em breve. Usa a conta demo para explorar.</div>', unsafe_allow_html=True)