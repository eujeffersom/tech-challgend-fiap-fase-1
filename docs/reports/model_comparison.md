| Modelo | ROC-AUC (CV) | PR-AUC (CV) | F1 (CV) | Recall (CV) | Threshold | F1 (teste) | ROC-AUC (teste) | Valor liquido (teste) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **logistic_regression** (campeao) | 0.8459 ± 0.0124 | 0.6600 ± 0.0195 | 0.6406 ± 0.0216 | 0.7411 ± 0.0521 | 0.58 | 0.6186 | 0.8413 | 56,050 |
| random_forest | 0.8415 ± 0.0081 | 0.6464 ± 0.0223 | 0.6375 ± 0.0115 | 0.7472 ± 0.0631 | 0.45 | 0.6274 | 0.8378 | 57,650 |
| gradient_boosting | 0.8478 ± 0.0122 | 0.6671 ± 0.0231 | 0.6374 ± 0.0166 | 0.7385 ± 0.0262 | 0.32 | 0.6257 | 0.8434 | 57,250 |
| pytorch_mlp | 0.8458 ± 0.0126 | 0.6577 ± 0.0169 | 0.6366 ± 0.0241 | 0.7739 ± 0.0321 | 0.53 | 0.6222 | 0.8427 | 57,700 |
| mlp_classifier | 0.8410 ± 0.0138 | 0.6549 ± 0.0213 | 0.6331 ± 0.0235 | 0.7144 ± 0.0694 | 0.32 | 0.6195 | 0.8385 | 55,700 |

Campeao: `logistic_regression`, maior F1 medio na validacao cruzada entre os modelos Scikit-Learn. Salvo em `models/champion.joblib` e servido pela API.
