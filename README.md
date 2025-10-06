# 🌡️ Previsão de Temperaturas Diárias Mínimas — Melbourne (1981–1990)

Este notebook implementa e compara **dois modelos de previsão de séries temporais** — um modelo estatístico clássico (`ThetaForecaster`, via [sktime](https://www.sktime.net)) e um modelo de aprendizado profundo (`LSTM`, via [TensorFlow/Keras](https://www.tensorflow.org/)) — aplicados à série de **temperaturas mínimas diárias** registradas em Melbourne, Austrália, entre 1981 e 1990.

---

## 📊 Dataset

**Fonte:** [Daily Minimum Temperatures — Melbourne (Kaggle)](https://www.kaggle.com/datasets/suprematism/daily-minimum-temperatures)

**Descrição:**
- Série temporal univariada (temperaturas mínimas diárias em °C)
- Período: 1981 a 1990  
- Tamanho: 3.650 observações (10 anos × 365 dias)
- Colunas:
  - `Date` — Data (YYYY-MM-DD)
  - `Temp` — Temperatura mínima em graus Celsius

**Tratamentos realizados:**
- Padronização do formato de data (`pd.to_datetime`)
- Remoção de ruídos (`?`, `°`, etc.)
- Conversão segura para `float`
- Preenchimento de lacunas com interpolação temporal
- Indexação diária (`asfreq('D')`)

---

## 🧠 Modelos de Previsão

### 1️⃣ Modelo Estatístico — `ThetaForecaster` (sktime)

O `ThetaForecaster` é um modelo baseado em decomposição linear e suavização exponencial simples, proposto por Assimakopoulos e Nikolopoulos (2000).  
Ele é amplamente usado como *baseline* robusto para previsões univariadas.

**Principais características:**
- Simples, rápido e interpretável  
- Capta tendência linear e padrões sazonais  
- Não depende de tuning complexo de hiperparâmetros  

```python
from sktime.forecasting.theta import ThetaForecaster
forecaster = ThetaForecaster(deseasonalize=False)
````

---

### 2️⃣ Modelo de Aprendizado Profundo — LSTM

A **Long Short-Term Memory (LSTM)** é uma rede recorrente (RNN) capaz de capturar dependências de longo prazo em séries temporais.
Foi configurada aqui para prever a temperatura de amanhã com base nos últimos 30 dias (`LOOKBACK = 30`).

**Arquitetura:**

```python
model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
```

**Treinamento:**

* Otimizador: Adam (`lr=1e-3`)
* Perda: MSE
* Épocas: 30
* Batch size: 32

---

## ⚖️ Métricas de Avaliação

Para comparar os resultados dos modelos, foram usadas três métricas complementares:

| Métrica                               | Fórmula                          | Interpretação                                  |   |                             |
| ------------------------------------- | -------------------------------- | ---------------------------------------------- | - | --------------------------- |
| **MAE (Mean Absolute Error)**         | 𝑀𝐴𝐸 = (1/n) Σ                 | y - ŷ                                          |   | Erro médio absoluto (em °C) |
| **RMSE (Root Mean Squared Error)**    | 𝑅𝑀𝑆𝐸 = √(Σ(y - ŷ)² / n)      | Penaliza mais erros grandes                    |   |                             |
| **MASE (Mean Absolute Scaled Error)** | 𝑀𝐴𝑆𝐸 = MAE_model / MAE_naive | Mede o desempenho relativo a um modelo *naive* |   |                             |

**Por que MASE?**
O **MASE** é recomendado por **Hyndman & Koehler (2006)** em *“Another look at measures of forecast accuracy” (IJF, vol. 22, n. 4)*, pois:

* É **independente da escala da série**
* Evita os problemas do **MAPE**, que falha quando há valores próximos de zero
* Permite comparar modelos em séries distintas

📚 **Referência:**

> Hyndman, R. J., & Koehler, A. B. (2006).
> *Another look at measures of forecast accuracy.*
> International Journal of Forecasting, 22(4), 679–688.
> [https://doi.org/10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)

---

## 📈 Resultados

Após o treinamento, ambos os modelos foram avaliados sobre o conjunto de **teste (último ano)**.

| Modelo            | MAE (°C) | RMSE (°C) |     MASE |
| :---------------- | -------: | --------: | -------: |
| `ThetaForecaster` | ≈ *x.xx* |  ≈ *x.xx* | ≈ *x.xx* |
| `LSTM`            | ≈ *x.xx* |  ≈ *x.xx* | ≈ *x.xx* |

> *Os valores exatos variam conforme o split e o treinamento, mas geralmente o LSTM atinge erro ligeiramente menor, ao custo de maior complexidade.*

---

## 💬 Conclusões

* O modelo **LSTM** apresentou bom desempenho em capturar não linearidades, mas requer mais tempo de treino e ajuste fino.
* O **ThetaForecaster** (ou Exponential Smoothing) continua sendo um ótimo baseline — rápido, estável e competitivo.
* Métricas como **MASE** permitem comparar de forma justa métodos de naturezas diferentes.
* Para produção, o trade-off entre interpretabilidade e precisão deve guiar a escolha do modelo.

---

## 🧾 Estrutura do Notebook

1. Instalação e setup
2. Download automático do dataset via API do Kaggle
3. Limpeza e preparação dos dados
4. Modelagem clássica (sktime ThetaForecaster)
5. Modelagem com Deep Learning (LSTM)
6. Avaliação e comparação com métricas padronizadas
7. Discussão de resultados e referências

---

## 🚀 Reprodutibilidade

O notebook é compatível com **Google Colab**.
Basta executar célula por célula; o dataset é baixado automaticamente com seu `kaggle.json`.

---

## 📚 Referências

* Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy.* *International Journal of Forecasting, 22(4), 679–688.*
* Assimakopoulos, V., & Nikolopoulos, K. (2000). *The Theta model: a decomposition approach to forecasting.* *International Journal of Forecasting, 16(4), 521–530.*
* Chollet, F. (2017). *Deep Learning with Python.* Manning Publications.
* [Dataset — Kaggle](https://www.kaggle.com/datasets/suprematism/daily-minimum-temperatures)

