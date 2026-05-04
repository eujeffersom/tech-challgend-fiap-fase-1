# Dados

Coloque aqui os dados do projeto.

- `raw/`: dados brutos, como `churn.csv`.
- `processed/`: dados transformados, quando necessario.

Os arquivos de dados reais ficam fora do Git por padrao. Use DVC ou outro storage versionado para
datasets grandes.

## Dataset Original

Fonte:

```text
https://github.com/albayraktaroglu/Datasets/blob/master/churn.csv
```

Download direto:

```bash
mkdir -p data/raw
curl -L https://raw.githubusercontent.com/albayraktaroglu/Datasets/master/churn.csv \
  -o data/raw/churn.csv
```

O arquivo deve ser salvo exatamente em:

```text
data/raw/churn.csv
```
