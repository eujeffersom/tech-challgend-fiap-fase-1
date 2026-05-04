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
Arquivo recebido/anexado: Churn1.csv
Nome esperado no projeto: churn.csv
```

O arquivo deve ser salvo exatamente em:

```text
data/raw/churn.csv
```

Se o arquivo baixado/anexado estiver como `Churn1.csv`, renomeie ou copie para `data/raw/churn.csv`.

Exemplo no macOS:

```bash
mkdir -p data/raw
cp ~/Downloads/Churn1.csv data/raw/churn.csv
```
