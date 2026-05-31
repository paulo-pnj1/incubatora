"""
=============================================================
GUIÃO PRÁTICO - CAPÍTULO III: APRENDIZAGEM AUTOMÁTICA (ML)
SECÇÃO 2: MODELOS NÃO SUPERVISIONADOS — K-MEANS
Universidade Kimpa Vita - Curso de Engenharia Informática
Disciplina: Inteligência Artificial II
=============================================================

OBJECTIVOS:
  - Aplicar K-Means para segmentação de clientes de e-commerce
  - Determinar o número ideal de clusters com 3 métricas:
    1. Método do Cotovelo (Elbow Method / WCSS)
    2. Índice de Silhouette
    3. Índice de Calinski-Harabasz (CH Score)
  - Interpretar e visualizar os clusters encontrados
  - Compreender as métricas de validação interna

DATASET: clientes_ecommerce.csv (simulado)
  Variáveis: Idade, Gasto_Mensal, Frequencia_Compras,
             Tempo_Plataforma_h
  (Sem rótulos — aprendizagem NÃO supervisionada)
"""

# ================================================================
# PASSO 0: IMPORTAÇÕES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')
FIGDIR = "./figuras"
os.makedirs(FIGDIR, exist_ok=True)

print("=" * 65)
print("  SECÇÃO 2 — MODELOS NÃO SUPERVISIONADOS: K-MEANS")
print("  Capítulo III: Aprendizagem Automática")
print("=" * 65)

# ================================================================
# PASSO 1: CARREGAR E EXPLORAR OS DADOS
# ================================================================
# Explicação: No aprendizado não supervisionado, os dados NÃO têm
# rótulos (labels). O algoritmo deve descobrir a estrutura
# intrínseca dos dados por conta própria.
# K-Means agrupa os dados em K clusters, minimizando a soma das
# distâncias quadráticas intra-cluster (WCSS).

print("\n[PASSO 1] Carregamento dos Dados")
print("-" * 50)

df = pd.read_csv("../datasets/clientes_ecommerce.csv")
print(f"→ Dimensão: {df.shape[0]} clientes × {df.shape[1]} variáveis")
print(f"→ Não há variável target (sem rótulos!)")
print(f"\n→ Primeiras 5 linhas:")
print(df.head())
print(f"\n→ Estatísticas descritivas:")
print(df.describe().round(2))

# Visualização exploratória: pairplot
print("\n→ Gerando pairplot exploratório...")
fig = plt.figure(figsize=(12, 10))
pd.plotting.scatter_matrix(df, alpha=0.4, figsize=(12, 10), diagonal='kde',
                            color='#3498db')
plt.suptitle("Matriz de Dispersão — Dataset Clientes E-Commerce", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/2a_pairplot_clientes.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"  Figura salva: 2a_pairplot_clientes.png")

# ================================================================
# PASSO 2: PRÉ-PROCESSAMENTO
# ================================================================
# Explicação: K-Means usa distância euclidiana para medir
# similaridade entre pontos. Sem normalização, variáveis com
# maior escala (ex: Gasto_Mensal em centenas) dominam o cálculo
# de distância, distorcendo os clusters.
# StandardScaler garante que todas as variáveis contribuem
# igualmente para a formação dos clusters.

print("\n[PASSO 2] Normalização dos Dados")
print("-" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

print(f"→ Dados normalizados (µ≈0, σ≈1) para todas as variáveis:")
print(f"  Médias:         {X_scaled.mean(axis=0).round(4)}")
print(f"  Desvios padrão: {X_scaled.std(axis=0).round(4)}")
print(f"→ JUSTIFICAÇÃO: K-Means é sensível à escala — normalizar")
print(f"  garante equidade entre as variáveis no cálculo de distância.")

# ================================================================
# PASSO 3: DETERMINAÇÃO DO NÚMERO IDEAL DE CLUSTERS (K)
# ================================================================
# Explicação: K é o único hiperparâmetro do K-Means.
# Escolhê-lo correctamente é crucial. Usamos 3 métricas:
#   1. WCSS (Elbow): queremos o "cotovelo" — ponto onde adicionar
#      mais clusters já não traz grande redução do WCSS.
#   2. Silhouette Score: varia de -1 a 1. Mede quão bem cada
#      amostra se encaixa no seu cluster vs vizinhos. Máximo = melhor.
#   3. Calinski-Harabasz: razão dispersão inter/intra-cluster.
#      Valores maiores = clusters mais compactos e separados.

print("\n[PASSO 3] Determinação do K Ideal — 3 Métricas de Validação")
print("-" * 50)

k_range = range(2, 11)
wcss_vals, sil_vals, ch_vals, db_vals = [], [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss_vals.append(km.inertia_)
    sil_vals.append(silhouette_score(X_scaled, labels))
    ch_vals.append(calinski_harabasz_score(X_scaled, labels))
    db_vals.append(davies_bouldin_score(X_scaled, labels))
    print(f"  K={k}: WCSS={km.inertia_:.2f} | Silhouette={sil_vals[-1]:.4f} | "
          f"CH={ch_vals[-1]:.2f} | DB={db_vals[-1]:.4f}")

# Plotar as 3 métricas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Métricas para Determinação do Número Ideal de Clusters (K)",
             fontsize=14, fontweight='bold')

# 1. Método do Cotovelo (WCSS)
axes[0].plot(list(k_range), wcss_vals, 'o-', color='#e74c3c', lw=2.5, markersize=8)
axes[0].set_title("Método do Cotovelo\n(WCSS — Within-Cluster Sum of Squares)", fontsize=10)
axes[0].set_xlabel("Número de Clusters (K)")
axes[0].set_ylabel("WCSS")
axes[0].axvline(x=3, color='gray', linestyle='--', alpha=0.7, label='K sugerido=3')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

# 2. Índice de Silhouette
best_sil_k = list(k_range)[np.argmax(sil_vals)]
axes[1].plot(list(k_range), sil_vals, 's-', color='#3498db', lw=2.5, markersize=8)
axes[1].set_title(f"Índice de Silhouette\n(Máximo = melhor | K={best_sil_k})", fontsize=10)
axes[1].set_xlabel("Número de Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=best_sil_k, color='gray', linestyle='--', alpha=0.7, label=f'K={best_sil_k}')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

# 3. Calinski-Harabasz
best_ch_k = list(k_range)[np.argmax(ch_vals)]
axes[2].plot(list(k_range), ch_vals, '^-', color='#2ecc71', lw=2.5, markersize=8)
axes[2].set_title(f"Índice Calinski-Harabasz\n(Máximo = melhor | K={best_ch_k})", fontsize=10)
axes[2].set_xlabel("Número de Clusters (K)")
axes[2].set_ylabel("CH Score")
axes[2].axvline(x=best_ch_k, color='gray', linestyle='--', alpha=0.7, label=f'K={best_ch_k}')
axes[2].legend()
axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/2b_metricas_validacao_clusters.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura salva: 2b_metricas_validacao_clusters.png")

print(f"\n  ★ ANÁLISE DAS MÉTRICAS:")
print(f"    Cotovelo sugere: K≈3 (inflexão visual)")
print(f"    Silhouette máximo em K={best_sil_k} (valor={max(sil_vals):.4f})")
print(f"    Calinski-Harabasz máximo em K={best_ch_k} (valor={max(ch_vals):.2f})")
print(f"    DECISÃO: Usar K=3 (consenso das métricas e interpretabilidade)")

# ================================================================
# PASSO 4: TREINAR K-MEANS COM K=3
# ================================================================
# Explicação: Usamos 'k-means++' para inicialização inteligente
# dos centróides — reduz o risco de convergir para mínimos locais.
# n_init=10: executa 10 vezes com inicializações diferentes e
# escolhe a melhor (menor WCSS). random_state=42: reproducibilidade.

print("\n[PASSO 4] Treinamento Final — K-Means com K=3")
print("-" * 50)

K_FINAL = 3
kmeans = KMeans(n_clusters=K_FINAL, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

print(f"→ K-Means treinado com K={K_FINAL}")
print(f"→ WCSS final: {kmeans.inertia_:.4f}")
print(f"→ Iterações até convergência: {kmeans.n_iter_}")

# Avaliação final com K=3
sil_final = silhouette_score(X_scaled, labels)
ch_final  = calinski_harabasz_score(X_scaled, labels)
db_final  = davies_bouldin_score(X_scaled, labels)
print(f"\n  Métricas do modelo final (K=3):")
print(f"    Silhouette Score:       {sil_final:.4f}  (quanto maior, melhor)")
print(f"    Calinski-Harabasz:      {ch_final:.2f}  (quanto maior, melhor)")
print(f"    Davies-Bouldin Index:   {db_final:.4f}  (quanto menor, melhor)")

# ================================================================
# PASSO 5: ANÁLISE DOS CLUSTERS
# ================================================================

print("\n[PASSO 5] Caracterização dos Clusters")
print("-" * 50)

perfis = df.groupby('Cluster').agg({
    'Idade': ['mean','std'],
    'Gasto_Mensal': ['mean','std'],
    'Frequencia_Compras': ['mean','std'],
    'Tempo_Plataforma_h': ['mean','std'],
    'Cluster': 'count'
}).round(2)
perfis.columns = ['Idade_Média','Idade_DP','Gasto_Média','Gasto_DP',
                  'Freq_Média','Freq_DP','Tempo_Média','Tempo_DP','N_Clientes']
print(perfis)

# Distribuição dos clusters
dist = df['Cluster'].value_counts().sort_index()
for c in range(K_FINAL):
    print(f"\n  ► Cluster {c} ({dist[c]} clientes, {dist[c]/len(df)*100:.1f}%):")
    row = perfis.loc[c]
    print(f"    Idade média: {row['Idade_Média']:.1f} anos")
    print(f"    Gasto mensal médio: ${row['Gasto_Média']:.2f}")
    print(f"    Frequência de compras: {row['Freq_Média']:.1f} compras/mês")
    print(f"    Tempo na plataforma: {row['Tempo_Média']:.2f}h/dia")

# ================================================================
# PASSO 6: VISUALIZAÇÕES DOS CLUSTERS
# ================================================================

print("\n[PASSO 6] Visualizações dos Clusters")
print("-" * 50)

colors_cl = {0:'#e74c3c', 1:'#3498db', 2:'#2ecc71'}
labels_cl  = {0:'Cluster 0', 1:'Cluster 1', 2:'Cluster 2'}

# Figura 1: Scatter plots principais
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Visualização dos Clusters K-Means (K=3)", fontsize=14, fontweight='bold')

pares = [('Gasto_Mensal','Frequencia_Compras'),
         ('Idade','Gasto_Mensal'),
         ('Tempo_Plataforma_h','Gasto_Mensal'),
         ('Idade','Frequencia_Compras')]
for ax, (x_col, y_col) in zip(axes.flat, pares):
    for c in range(K_FINAL):
        mask = df['Cluster'] == c
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   color=colors_cl[c], label=labels_cl[c], alpha=0.7, s=50, edgecolors='white')
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f'{x_col} vs {y_col}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/2c_scatter_clusters.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura salva: 2c_scatter_clusters.png")

# Figura 2: Perfil dos clusters (radar/barras)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Barras comparativas
features_plot = ['Idade','Gasto_Mensal','Frequencia_Compras','Tempo_Plataforma_h']
medias = df.groupby('Cluster')[features_plot].mean()
medias_norm = (medias - medias.min()) / (medias.max() - medias.min())

x_pos = np.arange(len(features_plot))
width = 0.25
for i, c in enumerate(range(K_FINAL)):
    axes[0].bar(x_pos + i*width, medias_norm.loc[c], width,
                label=f'Cluster {c}', color=list(colors_cl.values())[i], alpha=0.8, edgecolor='white')
axes[0].set_xticks(x_pos + width)
axes[0].set_xticklabels(features_plot, rotation=15, ha='right', fontsize=9)
axes[0].set_ylabel('Valor Normalizado (0-1)')
axes[0].set_title('Perfil Comparativo dos Clusters\n(Valores Normalizados)', fontsize=11)
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)

# Distribuição de tamanho dos clusters
sizes = [dist[c] for c in range(K_FINAL)]
wedge_colors = [colors_cl[c] for c in range(K_FINAL)]
axes[1].pie(sizes, labels=[f'Cluster {c}\n({s} clientes)' for c, s in enumerate(sizes)],
            colors=wedge_colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10}, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Distribuição de Clientes por Cluster', fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/2d_perfil_clusters.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura salva: 2d_perfil_clusters.png")

# Figura 3: PCA para visualização 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1','PC2'])
df_pca['Cluster'] = labels

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(K_FINAL):
    mask = df_pca['Cluster'] == c
    ax.scatter(df_pca.loc[mask,'PC1'], df_pca.loc[mask,'PC2'],
               color=colors_cl[c], label=f'Cluster {c}', alpha=0.7, s=60, edgecolors='white')
# Plotar centróides transformados
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:,0], centroids_pca[:,1], c='black', marker='X',
           s=200, zorder=5, label='Centróides')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)', fontsize=11)
ax.set_title(f'Clusters K-Means visualizados via PCA 2D\n'
             f'Variância total explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/2e_clusters_pca2d.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura salva: 2e_clusters_pca2d.png")

# ================================================================
# PASSO 7: RELATÓRIO FINAL
# ================================================================

print("\n" + "=" * 65)
print("  RELATÓRIO FINAL — SECÇÃO 2")
print("=" * 65)
print(f"\n  Dataset: Clientes E-Commerce")
print(f"  Amostras: {len(df)} | Features: 4 | Clusters: {K_FINAL}")
print(f"\n  Métricas do modelo K-Means (K=3):")
print(f"    WCSS:             {kmeans.inertia_:.4f}")
print(f"    Silhouette Score: {sil_final:.4f}")
print(f"    Calinski-Harabasz:{ch_final:.2f}")
print(f"    Davies-Bouldin:   {db_final:.4f}")
print(f"\n  ★ INTERPRETAÇÃO DOS CLUSTERS:")
for c in range(K_FINAL):
    row = perfis.loc[c]
    print(f"\n  Cluster {c} ({dist[c]} clientes):")
    print(f"    Idade média: {row['Idade_Média']:.0f} | Gasto: ${row['Gasto_Média']:.0f}/mês | "
          f"Freq: {row['Freq_Média']:.0f}/mês")
print("=" * 65)
print("\n✓ Secção 2 concluída. Todas as figuras salvas em ../figuras/")
