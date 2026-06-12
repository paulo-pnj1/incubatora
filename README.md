# 🧠 DataForge EDU

**Plataforma de Aprendizagem de Machine Learning para Universidades**

Desenvolvida em Angola 🇦🇴 para o contexto universitário lusófono.

---

## 🚀 Instalação rápida

### 1. Clonar / descompactar o projecto

```bash
cd dataforge-edu
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependências

```bash
# Instalação completa (inclui PyTorch, XGBoost, LightGBM, etc.)
pip install -r requirements.txt

# Instalação mínima para testar rapidamente (sem deep learning)
pip install streamlit pandas numpy matplotlib plotly scikit-learn seaborn \
            streamlit-authenticator tinydb bcrypt pyyaml xgboost lightgbm \
            joblib fpdf2 qrcode
```

### 4. Gerar o ficheiro de configuração de utilizadores

```bash
python -c "
import bcrypt, yaml, os

def hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

config = {
    'credentials': {
        'usernames': {
            'admin':     {'email': 'admin@dataforge.ao',   'name': 'Administrador',  'password': hash_pw('admin123'),   'role': 'admin'},
            'professor': {'email': 'prof@dataforge.ao',    'name': 'Professor Demo', 'password': hash_pw('prof123'),    'role': 'professor'},
            'aluno':     {'email': 'aluno@dataforge.ao',   'name': 'Aluno Demo',     'password': hash_pw('aluno123'),   'role': 'aluno'},
        }
    },
    'cookie': {'expiry_days': 7, 'key': 'dataforge_edu_2025', 'name': 'dataforge_session'},
    'preauthorized': {'emails': []}
}

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print('✅ config.yaml criado!')
"
```

### 5. Executar

```bash
streamlit run app.py
```

Acede em: **http://localhost:8501**

---

## 👤 Contas de demonstração

| Utilizador | Password | Papel |
|-----------|----------|-------|
| `aluno` | `aluno123` | Aluno |
| `professor` | `prof123` | Professor |
| `admin` | `admin123` | Administrador |

---

## 📁 Estrutura do projecto

```
dataforge-edu/
├── app.py                          # Entry point
├── requirements.txt
├── config.yaml                     # Utilizadores (gerado no passo 4)
├── README.md
│
├── modules/
│   ├── utils.py                    # Paleta, CSS, componentes, datasets, teoria
│   ├── auth.py                     # Login e perfis
│   ├── dashboard.py                # Dashboard e percurso de aprendizagem
│   │
│   └── supervisionado/
│       ├── classificacao.py        # ✅ Fase 1 — 13+ algoritmos
│       └── regressao.py            # ✅ Fase 1 — 12+ algoritmos
│
└── data/
    ├── db.json                     # TinyDB (criado automaticamente)
    └── datasets/                   # CSVs locais adicionais
```

---

## 🗺️ Roadmap de Fases

| Fase | Módulos | Estado |
|------|---------|--------|
| **Fase 1** | Dashboard, Auth, Classificação, Regressão | ✅ **Completo** |
| **Fase 2** | Clustering, Redução Dim., Anomalias, MLP, Visão, Comparador | 🔜 |
| **Fase 3** | NLP, Séries Temporais, Modelos Generativos | 🔜 |
| **Fase 4** | Aprendizagem por Reforço | 🔜 |
| **Fase 5** | Desafios, Certificados, Monetização | 🔜 |

---

## 🎯 Algoritmos disponíveis — Fase 1

### Classificação (13 algoritmos)
KNN · Decision Tree · Random Forest · Extra Trees · Gradient Boosting ·
AdaBoost · SVM · Logistic Regression · Naive Bayes (Gaussian/Bernoulli) ·
SGD · LDA · Bagging · XGBoost* · LightGBM*

### Regressão (14 algoritmos)
Linear · Ridge · Lasso · ElasticNet · Huber · Bayesian Ridge ·
KNN Regressor · Decision Tree · Random Forest · Extra Trees ·
Gradient Boosting · AdaBoost · SVR · SGD · XGBoost* · LightGBM*

*Se instalados

---

## 💡 Funcionalidades da Fase 1

- ✅ Autenticação com cookies persistentes
- ✅ Dashboard com percurso de aprendizagem visual
- ✅ Sistema de pontos por actividade
- ✅ Teoria integrada por algoritmo (analogia + como funciona + quando usar + cuidados)
- ✅ Modo Guiado (passo a passo) + Modo Livre
- ✅ 10 datasets educativos embutidos (Iris, Titanic, Vinho, etc.)
- ✅ Diagnóstico automático (overfitting, underfitting, variância)
- ✅ Validação cruzada com visualização por fold
- ✅ Matriz de confusão, Curva ROC, Importância de features
- ✅ Previsão interactiva com novos dados
- ✅ Exportação de modelos (.pkl)
- ✅ Historial de modelos treinados
- ✅ Design profissional responsivo
