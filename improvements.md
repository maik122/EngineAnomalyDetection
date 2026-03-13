# Engine Anomaly Detection (GRU) — Project Roadmap
## MoSCoW Prioritisation

**Priority Key:** 🔴 Must Have | 🟠 Should Have | 🟡 Could Have | ⚪ Won't Have (now)

---

### 🐛 Known Bugs (Fix Before Anything Else)

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | `app.py` `/predict` | `reshape(1, -1)` feeds shape `(1, 14)` — GRU expects `(1, 30, 18)`. Prediction runs but has zero temporal context. | Accept last 30 cycles as CSV upload; reshape to `(1, 30, 18)` before `model.predict()` |
| 2 | `app.py` `/predict` | `sensor_columns` lists 14 sensors only. Scaler was fitted on 19 columns: `Unit + Time + Op_Setting_1/2/3 + 14 sensors`. Scaler transform on 14 columns silently produces wrong values. | Pass all 19 columns in original order, or refit scaler on sensor-only columns and resave |
| 3 | `app.py` `/predict` | `sensor_columns` omits `Op_Setting_1/2/3` and `Time`, which are features in the trained model (`create_sequences` keeps `Time`; drops only `Unit` and `RUL`). | Include `Time, Op_Setting_1, Op_Setting_2, Op_Setting_3` in the prediction payload |
| 4 | Notebook | `train_data_cleaned` (Sensor_9 & Sensor_14 dropped) was never used for training — `create_sequences` runs on `train_data_split`. Model includes both sensors. Document this or retrain on cleaned data. | Either document as intentional or retrain; do not drop them in the Flask input |

> **Model input shape confirmed:** `(batch=1, timesteps=30, features=18)` — Time + Op_Setting_1/2/3 + Sensor_2/3/4/7/8/9/11/12/13/14/15/17/20/21

---

### 📂 Dataset Context (CMAPSS — NASA)

| Dataset | Train Engines | Test Engines | Op Conditions | Fault Modes | Current Status |
|---------|--------------|-------------|---------------|-------------|----------------|
| FD001 | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) | ✅ Trained |
| FD002 | 260 | 259 | 6 | 1 (HPC Degradation) | ⏳ Not started |
| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan Degradation) | ⏳ Not started |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan Degradation) | ⏳ Not started |

> **Implication:** The current model is trained and evaluated on FD001 only — the simplest subset (1 condition, 1 fault). It will generalise poorly to FD002–FD004 without retraining or a multi-dataset approach. The Op_Setting columns become critical for FD002/FD004 where 6 operating conditions introduce substantial sensor variability — the zero-variance feature selection done on FD001 may not hold there.

> **"Try Sample Engine" button:** Pull any of the 100 test engines from `test_FD001.txt` (last 30 cycles), run `/predict`, compare against ground truth from `RUL_FD001.txt`.

---

### Project Structure & Setup

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Modular `src/` layout (`data/`, `models/`, `api/`, `utils/`) | ⏳ Todo |
| 🔴 Must | `pyproject.toml` / `setup.cfg` — make project importable as package | ⏳ Todo |
| 🔴 Must | Docker containerisation + `docker-compose.yml` | ⏳ Todo |
| 🔴 Must | Structured logging with JSON formatter (structlog) | ⏳ Todo |
| 🔴 Must | Unit tests for data pipeline — pytest, coverage >80% | ⏳ Todo |
| 🟠 Should | CI/CD pipeline (GitHub Actions — lint, test, build, push) | ⏳ Todo |

---

### Data & Preprocessing

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Feature selection — drop 7 zero-variance sensors; keep 14 + 3 Op_Settings + Time | ✅ Done |
| 🔴 Must | MinMaxScaler fit on 19 columns (Unit + Time + Op_Settings + sensors, excl. RUL) | ✅ Done |
| 🔴 Must | Sliding-window sequence creation — 30-cycle windows per engine, label = RUL at t+30 | ✅ Done |
| 🔴 Must | Unit-based train/val split — 80 engines train, 20 engines val (not row-level split) | ✅ Done |
| 🔴 Must | **Fix scaler column mismatch** — refit scaler on 18 inference columns (no Unit) and resave | 🔥 Bug fix |
| 🔴 Must | Input validation for 30-cycle CSV upload — shape check, dtype check, missing value check | ⏳ Next |
| 🟡 Could | Real-time streaming ingestion (Kafka / MQTT) | ⏳ Later |

---

### Modelling & Evaluation

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | GRU model — 64 units, tanh, Dropout 0.2, Dense 32 relu, Dense 1 output | ✅ Done |
| 🔴 Must | Validation metrics: MAE 18.09 / RMSE 24.96 / R² 0.81 (FD001 val, 20 engines) | ✅ Done |
| 🔴 Must | Evaluate on FD001 test set vs `RUL_FD001.txt` ground truth (not just val split) | ⏳ Next |
| 🔴 Must | Model versioning — save `gru_model.keras` + corrected `scaler.pkl` under `models/v1/` with `model_card.json` | ⏳ Next |
| 🟠 Should | Experiment tracking — log seq_length, features, dropout, epochs per run (MLflow) | ⏳ Todo |
| 🟠 Should | SHAP feature importance — per-prediction bar chart of 18 features | ⏳ Todo |
| 🟠 Should | Data drift detection — KS-test on incoming 30-cycle windows vs FD001 training distribution (Evidently) | ⏳ Todo |
| 🟡 Could | Retrain on FD002 — 6 op conditions require condition-normalisation before feature selection | ⏳ Later |
| 🟡 Could | Retrain on FD003/FD004 — multi-fault adds Fan Degradation class; may need separate model per fault mode | ⏳ Later |
| 🟡 Could | Hyperparameter tuning — seq_length (20/30/50), GRU units, dropout (Optuna) | ⏳ Later |

---

### API & Deployment

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Flask `/predict` endpoint — skeleton exists | ✅ Done (broken) |
| 🔴 Must | **Fix `/predict`** — accept 30-cycle CSV, reshape to `(1, 30, 18)`, use corrected scaler | 🔥 Bug fix |
| 🔴 Must | **Redesign frontend** — RUL gauge (0–125), green/amber/red status, sensor trend chart, sample engine button | ⏳ Next |
| 🟠 Should | Migrate to FastAPI — OpenAPI `/docs`, async, Pydantic validation | ⏳ Todo |
| 🟠 Should | RUL alerting — webhook / email when RUL drops below threshold | ⏳ Todo |
| 🟡 Could | Cloud deployment (AWS / GCP / Heroku) | ⏳ Later |
| ⚪ Won't | Edge / embedded deployment (TFLite / ONNX on engine controller) | 🚫 Out of scope |

---

### Monitoring & MLOps

| Priority | Task | Status |
|----------|------|--------|
| 🟠 Should | Automated retraining pipeline triggered on drift signal | ⏳ Todo |
| 🟡 Could | Monitoring dashboard — rolling RUL per engine, alert history (Grafana) | ⏳ Later |
| 🟡 Could | Airflow / Prefect ETL + retraining DAG | ⏳ Later |
| ⚪ Won't | Digital twin integration (physics-based simulation) | 🚫 Out of scope |
| ⚪ Won't | Full Kubernetes auto-scaling infrastructure | 🚫 Out of scope |

---

### Next Steps (Priority Order)

1. **Fix scaler** — refit on 18 columns (drop Unit, keep Time + Op_Settings + 14 sensors); resave `scaler.pkl`
2. **Fix `/predict`** — accept CSV upload of last 30 cycles; reshape to `(1, 30, 18)`; include Time + Op_Settings in payload
3. **Evaluate on FD001 test set** — run the fixed pipeline against all 100 test engines; compare predicted RUL to `RUL_FD001.txt` ground truth; report MAE/RMSE/R²
4. **"Try Sample Engine" button** — serve last 30 cycles of a random FD001 test engine directly; show predicted vs actual RUL
5. **Redesign frontend** — RUL gauge (0–125), colour-coded status (green ≥60 / amber 30–59 / red <30), sensor trend line chart
6. **Model versioning** — bundle `gru_model.keras` + corrected `scaler.pkl` + `model_card.json` (dataset=FD001, metrics, features, seq_length=30) under `models/v1/`
7. **Structured logging** — JSON log per prediction: input shape, dataset, predicted RUL, latency
8. **Docker + `docker-compose`** — reproducible runtime; single `docker-compose up` to run the full app
9. **Unit tests** — pytest: sequence creation, scaler shape, `/predict` with mock 30-cycle FD001 input
10. **MLflow experiment tracking** — retroactively log FD001 run; required before attempting FD002/FD003 retraining

