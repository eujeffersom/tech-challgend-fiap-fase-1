#!/usr/bin/env python
# coding: utf-8

# # 01 - EDA e Baselines
#
# Analise exploratoria da base Telco Customer Churn (Kaggle) e primeiros baselines.
#
# O que eu queria responder aqui:
#
# 1. A base esta pronta para modelar? (volume, nulos, duplicados, tipos)
# 2. Quanto de churn existe e o que isso muda na escolha de metrica?
# 3. Quais variaveis separam quem cancela de quem fica?
# 4. Tem alguma variavel redundante ou que precise de cuidado no pre-processamento?
# 5. Qual o desempenho minimo que um modelo precisa bater?
#
# O notebook usa o mesmo pacote `churn` do codigo de producao (`src/churn`), entao a leitura
# e a limpeza dos dados sao exatamente as mesmas da API e do treino.

# In[1]:


# O mlflow importa o tqdm, que reclama da falta do ipywidgets. Nao afeta nada aqui.
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")


# In[2]:


import hashlib
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# Funciona tanto rodando da raiz do projeto quanto de dentro de notebooks/.
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn.config import RANDOM_SEED  # noqa: E402
from churn.data import load_churn_csv, split_features_target  # noqa: E402
from churn.features import build_preprocessor  # noqa: E402
from churn.metrics import binary_classification_metrics  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
DATASET_SOURCE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DATASET_ORIGINAL_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Mesmo banco do MLflow que o make mlflow abre, mesmo rodando de dentro de notebooks/.
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
# Sem isso o mlflow enche a saida com log de criacao de tabela.
logging.getLogger("mlflow").setLevel(logging.WARNING)

CHURN_COLOR = "#E4572E"
STAY_COLOR = "#4C78A8"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 3)


# ## 1. Leitura dos dados

# In[3]:


df = load_churn_csv(DATA_PATH)
print(f"{df.shape[0]} clientes e {df.shape[1]} colunas")
df.head()


# O `load_churn_csv` valida o schema com Pandera e converte o alvo `Churn` de Yes/No para
# 1/0. Daqui pra frente, `Churn = 1` e o cliente que cancelou.

# ## 2. Qualidade dos dados

# In[4]:


quality = pd.DataFrame(
    {
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "valores_unicos": df.nunique(),
    }
)
quality


# In[5]:


readiness = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "duplicated_rows": int(df.duplicated().sum()),
    "duplicated_ids": int(df["customerID"].duplicated().sum()),
    "missing_cells": int(df.isna().sum().sum()),
    "missing_rate": float(df.isna().sum().sum() / df.size),
    "target_churn_rate": float(df["Churn"].mean()),
}
readiness


# In[6]:


# Quem sao os clientes sem TotalCharges?
df.loc[df["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]


# **O que a qualidade mostrou**
#
# - 7.043 clientes, nenhuma linha duplicada e nenhum `customerID` repetido.
# - O unico problema e o `TotalCharges`: 11 clientes vem em branco no CSV original. Todos tem
#   `tenure = 0`, ou seja, acabaram de entrar e ainda nao tiveram a primeira cobranca. Nao e erro
#   de dado, e cliente novo.
# - Sao 0,16% das linhas. Em vez de jogar fora, o pipeline preenche com a mediana
#   (`SimpleImputer` em `features.py`), e a API aceita `TotalCharges` vazio pelo mesmo motivo.
# - As categoricas tem no maximo 4 valores, entao one-hot encoding resolve sem explodir colunas.
# - `customerID` e so identificador e sai antes do treino.

# ## 3. Distribuicao do alvo

# In[7]:


counts = df["Churn"].value_counts().sort_index()
shares = counts / counts.sum()

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(["Ficou (0)", "Cancelou (1)"], counts.values, color=[STAY_COLOR, CHURN_COLOR])
for bar, share in zip(bars, shares.values, strict=True):
    ax.annotate(
        f"{share:.1%}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
    )
ax.set_ylim(0, counts.max() * 1.12)
ax.set_title("Distribuicao do churn")
ax.set_ylabel("Clientes")
plt.tight_layout()
plt.show()

shares.rename("proporcao")


# 26,5% dos clientes cancelaram (1.869 de 7.043). E um desbalanceamento moderado, mas ja
# basta para a acuracia enganar: um modelo que responde "ninguem vai sair" acerta 73,5% e nao
# serve para nada.
#
# Por isso as metricas principais do projeto sao **F1**, **recall** e **PR-AUC**, que olham
# para a classe de churn, e o threshold de decisao e escolhido na validacao cruzada em vez de
# ficar fixo em 0,5.

# ## 4. Variaveis numericas

# In[8]:


numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
df.groupby("Churn")[numeric].median().rename(index={0: "ficou", 1: "cancelou"})


# In[9]:


fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, column in zip(axes, numeric, strict=True):
    for label, color, name in [(0, STAY_COLOR, "Ficou"), (1, CHURN_COLOR, "Cancelou")]:
        values = df.loc[df["Churn"] == label, column].dropna()
        ax.hist(values, bins=30, alpha=0.55, density=True, color=color, label=name)
    ax.set_title(column)
    ax.set_yticks([])
axes[0].legend()
fig.suptitle("Distribuicao das numericas por churn (densidade)")
plt.tight_layout()
plt.show()


# In[10]:


tenure_band = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 meses", "13-24", "25-48", "49-72"],
)
by_tenure = df.groupby(tenure_band, observed=True)["Churn"].agg(taxa_churn="mean", clientes="size")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(by_tenure.index.astype(str), by_tenure["taxa_churn"], color=CHURN_COLOR)
ax.axhline(df["Churn"].mean(), color="gray", linestyle="--", linewidth=1, label="media geral")
ax.set_title("Taxa de churn por tempo de casa")
ax.set_ylabel("Taxa de churn")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend()
plt.tight_layout()
plt.show()

first_year_share = df.loc[df["tenure"] <= 12, "Churn"].sum() / df["Churn"].sum()
print(f"Dos clientes que cancelaram, {first_year_share:.1%} estavam no primeiro ano.")
by_tenure


# **O que as numericas mostraram**
#
# - **Tempo de casa e o sinal mais forte.** Quem cancela tem mediana de 10 meses de casa, contra
#   38 de quem fica. No primeiro ano a taxa de churn e 47,4%, e cai para 9,5% depois de 4 anos.
#   Mais da metade (55,5%) dos cancelamentos acontece no primeiro ano.
# - **Quem cancela paga mais por mes**: mediana de 79,65 contra 64,43.
# - `TotalCharges` e menor para quem cancela, mas isso e reflexo do tempo de casa (pouco tempo,
#   pouco acumulado). A secao 6 confirma que ela e quase uma combinacao das outras duas.

# ## 5. Variaveis categoricas

# In[11]:


categorical = [
    column for column in df.columns if df[column].dtype == "object" and column != "customerID"
] + ["SeniorCitizen"]


def churn_by(column: str) -> pd.DataFrame:
    return df.groupby(column)["Churn"].agg(taxa_churn="mean", clientes="size")


# Amplitude = maior taxa de churn menos a menor taxa entre as categorias da coluna.
# Quanto maior, mais a coluna separa quem sai de quem fica.
spread = pd.Series(
    {
        column: churn_by(column)["taxa_churn"].agg(lambda s: s.max() - s.min())
        for column in categorical
    }
).sort_values(ascending=False)
spread.rename("amplitude_taxa_churn").to_frame()


# In[12]:


top = spread.index[:8]
overall = df["Churn"].mean()

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, column in zip(axes.flat, top, strict=True):
    table = churn_by(column).sort_values("taxa_churn")
    ax.barh(table.index.astype(str), table["taxa_churn"], color=CHURN_COLOR)
    ax.axvline(overall, color="gray", linestyle="--", linewidth=1)
    ax.set_title(column)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", labelsize=8)
fig.suptitle("Taxa de churn por categoria (linha tracejada = media geral de 26,5%)")
plt.tight_layout()
plt.show()


# In[13]:


# Contrato e tipo de internet juntos: onde o churn se concentra?
segment = df.pivot_table(
    index="Contract", columns="InternetService", values="Churn", aggfunc="mean"
)
segment


# In[14]:


hot = df[(df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic")]
print(
    f"Mensal + fibra: {len(hot)} clientes ({len(hot) / len(df):.1%} da base), "
    f"churn de {hot['Churn'].mean():.1%} e {hot['Churn'].sum() / df['Churn'].sum():.1%} "
    "de todos os cancelamentos."
)


# **O que as categoricas mostraram**
#
# - **Contrato e a variavel que mais separa**: 42,7% de churn no mensal, 11,3% no anual e 2,8%
#   no de dois anos.
# - **Fibra otica cancela muito mais** (41,9%) que DSL (19,0%). Quem nao tem internet quase nao
#   sai (7,4%).
# - **Servicos de protecao seguram o cliente**: sem `OnlineSecurity` ou sem `TechSupport` o churn
#   fica em torno de 42%; com eles, cai para cerca de 15%.
# - **Boleto eletronico** (`Electronic check`) tem 45,3% de churn, contra 15% a 19% nos outros
#   meios de pagamento.
# - **Clientes idosos** (`SeniorCitizen = 1`) cancelam 41,7%, contra 23,6% dos demais.
# - Nas colunas de servico, parte da amplitude vem da categoria `No internet service` (7,4%).
#   Em `StreamingTV` e `StreamingMovies`, ter ou nao o servico quase nao muda nada (cerca de 30%
#   contra 34%); o que pesa ali e ter internet.
# - **Genero nao faz diferenca** (26,9% vs 26,2%), e `PhoneService`/`MultipleLines` quase nao
#   mudam nada.
# - O cruzamento mostra onde esta o problema: **mensal + fibra** sao 30% da base, tem 54,6% de
#   churn e respondem por 62,2% de todos os cancelamentos. E o segmento obvio para a primeira
#   campanha de retencao.

# ## 6. Correlacao entre as numericas

# In[15]:


corr = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        value = corr.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(image, ax=ax, shrink=0.8)
ax.set_title("Correlacao (Pearson)")
plt.tight_layout()
plt.show()

estimated_total = df["tenure"] * df["MonthlyCharges"]
total_corr = df["TotalCharges"].corr(estimated_total)
print(f"Correlacao entre TotalCharges e tenure x MonthlyCharges: {total_corr:.4f}")


# `TotalCharges` tem correlacao de 0,9996 com `tenure x MonthlyCharges`, ou seja, quase nao traz
# informacao nova. Mantive a coluna porque a API recebe esse campo e a regularizacao da Regressao
# Logistica lida bem com essa redundancia, mas ela e candidata natural a sair numa proxima versao.
#
# Fora isso, nenhuma numerica tem correlacao alta o bastante com o churn para resolver sozinha
# (a mais forte e `tenure`, com -0,35). O sinal vem da combinacao de contrato, tempo de casa,
# servicos e forma de pagamento.

# ## 7. Resumo para a modelagem
#
# | Achado | O que muda no modelo |
# | --- | --- |
# | 26,5% de churn | Avaliar por F1, recall e PR-AUC, nunca so acuracia. Threshold escolhido na CV |
# | `TotalCharges` vazio em 11 clientes novos | Imputar com a mediana dentro do pipeline, sem descartar linhas |
# | Categoricas com ate 4 valores | One-hot encoding, `handle_unknown="ignore"` para nao quebrar em producao |
# | Escalas bem diferentes (`tenure` ate 72, `TotalCharges` ate ~8.700) | `StandardScaler` nas numericas, importante para Regressao Logistica e MLP |
# | `TotalCharges` quase igual a `tenure x MonthlyCharges` | Redundante. Mantida por ora, modelos regularizados lidam bem |
# | Sinais fortes e bem comportados (contrato, tenure, fibra, servicos) | Um modelo linear deve ir bem. Vale comparar com arvores e MLP antes de concluir |
# | Genero sem efeito; idosos com churn bem maior | Ponto de atencao de vies para o Model Card |
# | `customerID` | Removido antes do treino |

# ## 8. Baselines
#
# Dois pontos de referencia: um modelo ingenuo (`DummyClassifier`, sempre chuta a classe
# majoritaria) e uma Regressao Logistica balanceada. Separacao 80/20 estratificada, e a
# validacao cruzada roda so dentro dos 80% de treino.
#
# A comparacao completa com Random Forest, Gradient Boosting e MLP fica em
# `src/churn/compare_models.py` (`make compare-models`).

# In[16]:


x, y = split_features_target(df)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_SEED,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

models = {
    "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
}

rows = []
for model_name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", estimator),
        ]
    )
    cv_results = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring)
    pipeline.fit(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    test_metrics = binary_classification_metrics(y_test, y_prob)
    row = {"model": model_name}
    row.update({f"cv_{metric}": float(np.mean(cv_results[f"test_{metric}"])) for metric in scoring})
    row.update({f"test_{metric}": value for metric, value in test_metrics.items()})
    rows.append(row)

results = pd.DataFrame(rows).set_index("model")
results.T


# O Dummy confirma o que a secao 3 dizia: 73,5% de acuracia com F1 zero, porque nunca acerta
# um cancelamento. A Regressao Logistica balanceada ja passa de 0,84 de ROC-AUC e serve de
# referencia minima para os outros modelos.

# ## 9. Registro no MLflow
#
# Cada baseline vai para o MLflow com a origem e o hash do dataset, para dar para saber depois
# exatamente com qual arquivo o resultado foi gerado.

# In[17]:


dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
mlflow.set_experiment("churn_eda_baselines")

for model_name, row in results.iterrows():
    with mlflow.start_run(run_name=f"eda_{model_name}"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_source": DATASET_SOURCE,
                "dataset_original_file": DATASET_ORIGINAL_FILE,
                "dataset_local_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
                "dataset_sha256": dataset_hash,
                "cv_folds": 5,
                "split": "80_20_stratified",
                "seed": RANDOM_SEED,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in row.items()})
        mlflow.log_metrics({f"data_{key}": float(value) for key, value in readiness.items()})

print(f"{len(results)} runs registrados no experimento churn_eda_baselines")
