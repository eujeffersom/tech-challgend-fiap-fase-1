# Dados

Coloque aqui os dados do projeto.

- `raw/`: dados brutos, como `churn.csv`.
- `processed/`: dados transformados, quando necessario.

Os arquivos de dados reais ficam fora do Git por padrao. Use DVC ou outro storage versionado para
datasets grandes.

## Dataset Original

Fonte:

```text
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Arquivo original no site: WA_Fn-UseC_-Telco-Customer-Churn.csv
Renomear para: data/raw/churn.csv
```

O arquivo deve ser salvo exatamente em:

```text
data/raw/churn.csv
```

Se o arquivo baixado estiver como `WA_Fn-UseC_-Telco-Customer-Churn.csv`, renomeie ou copie para
`data/raw/churn.csv`.

Exemplo no macOS:

```bash
mkdir -p data/raw
cp ~/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/churn.csv
```

Opcionalmente, use o comando do projeto para baixar via KaggleHub:

```bash
uv run --no-editable --python 3.14.4 churn-download-data --output data/raw/churn.csv
```
