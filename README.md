# TP_Finanzas_Malacalza
Trabajo practico de Finanzas y Control de Franco Malacalza
Objetivo. Integrar análisis de riesgo, correlaciones, optimización de cartera, análisis técnico y valuación fundamental sobre una cartera tech/fintech con foco en MELI (comparables: AMZN, BABA, SHOP, NU; benchmark: SPY).


## 📚 Material teórico 

Este repositorio incluye una **síntesis conceptual** creada por el autor y usada como base del análisis. La versión extendida está en `docs/metodologia.md`.
A continuación, los puntos clave (resumen para el revisor):

### 1) Riesgo y retorno

* **Retorno logarítmico**: $r_t=\ln(P_t/P_{t-1})$. Evita sesgos por composición.
* **Volatilidad anualizada**: $\sigma_{ann}=\sigma_{diaria}\sqrt{252}$.
* **Sharpe**: $(E[R]-r_f)/\sigma$. Mide retorno por unidad de riesgo.

### 2) Pérdidas extremas

* **VaR 95% (histórico)**: percentil 5% de $r_t$.
* **CVaR 95% (ES)**: media de retornos en la cola $r_t \le VaR$.
* **Máx Drawdown**: caída pico-a-valle más profunda en la curva de capital.

### 3) Correlación y diversificación

* **Matriz de correlaciones** para cuantificar co-movimientos.
* **Efecto crisis**: en estrés, suben las correlaciones → cae la diversificación.
* **Rolling-corr** (60 días) para ver cómo cambia en el tiempo.

### 4) Optimización de cartera

* **Frontera eficiente (Montecarlo con restricciones)**:

  * Límites realistas: **5%–40%** por activo, **costos 0.5%** (impactan retorno).
  * Carteras reportadas: **Máx Sharpe**, **Mín Vol**, **Elegida (perfil)**.

### 5) Análisis técnico (resumen de reglas)

* **Tendencia**: SMA 50/200 (Golden/Death Cross).
* **Momentum**: MACD (12/26/9), **RSI 14** (30/70), **ADX 14** (umbral 25).
* **Bollinger**: media 20 ± 2σ; compresión (squeeze) como alerta de ruptura.
* **Backtesting**: Sharpe, Calmar, Win Rate, Máx DD; sólo long para simplicidad.

### 6) Valuación DCF (flujo a accionista)

* **FCF** ≈ (EBIT×(1–t) + DA – Capex – ΔWC).
* **WACC** con componentes explícitos (CoE por CAPM, CoD neto, pesos E/D).
* **Terminal** (Gordon): $\text{TV} = \dfrac{FCF_{t+1}}{WACC-g_\infty}$.
* **Sensibilidades**: WACC × $g_\infty$ × margen FCF (matrices de calor).

> Toda la teoría anterior fue **redactada por el autor** y referenciada en cada celda del notebook donde se aplica. Las cifras y elecciones de parámetros se justifican en el propio notebook (ver secciones “Metodología”, “Hallazgos” y “DCF”).

---

## ✍️ Declaración de originalidad y uso de IA

* El **análisis, selección de supuestos, interpretación de resultados y conclusiones** son de **autoría propia**.
* Cualquier uso de herramientas de IA se limitó a:

  1. **Formateo de código y gráficos** (p. ej., estilos de Seaborn/Matplotlib).
  2. **Corrección de errores de Python** (mensajes, imports, manejo de columnas de Yahoo).
  3. **Mejora de redacción** sin alterar el **criterio de inversión**.
* No se copiaron respuestas automáticas ni se delegó la **toma de decisiones** (supuestos del DCF, parámetros de backtest, elección de cartera, etc.).
* Las decisiones clave están **explicadas y defendidas** en el informe (por qué esos rangos de WACC/g, por qué esos límites de pesos, por qué esa cartera “Elegida”).

> Entiendo que el TP **penaliza** el uso no declarado de IA. Por eso, **dejo explícito** lo anterior y mantengo el código/notebook **reproducible** para que cualquier evaluador verifique los resultados.

---

## 🧱 Estructura del repositorio

```
finanzas-trabajo-practico-utn-2025/
│
├── README.md                      # Este archivo (síntesis + cómo correr)
├── CONSIGNAS.md                   # Consignas del TP (opcional)
├── requirements.txt               # Dependencias exactas
├── .gitignore                     # Ignora venv, checkpoints y outputs
│
├── notebooks/
│   ├── 01_obtencion_datos.ipynb
│   ├── 02_analisis_cartera.ipynb
│   ├── 03_analisis_tecnico.ipynb
│   ├── 04_analisis_fundamental.ipynb
│   └── 05_integracion_final.ipynb  # Notebook “corré todo”
│
├── src/                           # Utilidades (módulos Python)
│   ├── __init__.py
│   ├── data_utils.py
│   ├── portfolio_utils.py
│   ├── technical_analysis.py
│   └── fundamental_analysis.py
│
├── data/                          # CSVs generados (agregado al .gitignore)
├── outputs/                       # Figuras y tablas para el informe
│
└── docs/
    ├── metodologia.md             # Material teórico ampliado y reproducibilidad
    └── recursos.md                # Notas, links y bibliografía
```

---

## ⚙️ Reproducibilidad (cómo correr)

1. **Clonar** el repo y entrar a la carpeta.
2. (Opcional) Crear y activar **venv**.
3. `pip install -r requirements.txt`
4. Abrir `notebooks/05_integracion_final.ipynb` → **Restart & Run All**.
5. Las figuras/tablas se guardan en `outputs/` y se referencian en el informe.

---

## 🧭 Criterio personal (qué se evalúa aquí)

* **Elección de comparables**: sector, mercado, tamaño y relevancia para un inversor local.
* **Supuestos del DCF**: márgenes y crecimiento consistentes con históricos/industria; WACC desagregado.
* **Gestión de riesgo**: límites por activo, costos, lectura de correlaciones en crisis.
* **Estrategia técnica coherente**: reglas simples, backtest transparente y métricas claras.
* **Síntesis y toma de posición**: recomendación documentada y defendida con sensibilidad.

---

## 📄 Licencia académica

Este repo se publica **exclusivamente** con fines académicos. Se prohíbe su reutilización como entrega de terceros sin autorización explícita del autor y de la cátedra.

---

> Si sos revisor: todo el pipeline es **determinístico** (seed fijada) y los notebooks corren de punta a punta sin intervención manual. Cualquier inconsistencia o duda puede trazarse a las celdas de “Metodología” y a los módulos en `src/`.
