# Notebooks

Analise exploratoria e experimentos. O codigo que vai para producao (treino, comparacao e API)
fica em `src/churn/`. O notebook importa esse mesmo pacote, entao a leitura e a limpeza dos dados
sao as mesmas.

## 01_eda_e_baselines

- `01_eda_e_baselines.ipynb`: EDA completa e baselines (Dummy e Regressao Logistica). Fica salvo
  ja executado, entao da para ler os graficos e tabelas direto no GitHub.
- `01_eda_e_baselines.py`: mesmo conteudo em texto puro, gerado a partir do notebook. Serve para
  revisar diferencas no Git. Nao edite esse arquivo, edite o `.ipynb`.

Para reexecutar tudo e regenerar o `.py` (precisa do `data/raw/churn.csv`):

```bash
make notebook
```

Para abrir no VS Code ou no Jupyter, escolha o kernel do `.venv` do projeto.
