# Musanze Harvest Intelligence Service

**Student:** Christine Ndatimana  
**Registration number:** 25/27201  
**Institutional username:** ug2527201@ines.ac.rw *(verify against your official INES account before submission)*  
**Verification code:** AI-A2-7201  
**Stakeholder:** Musanze potato cooperative field officer

## 1. Purpose

Musanze Harvest Intelligence Service is a classroom decision-support prototype designed to estimate potato harvest quantity, predict high production risk, and group farms according to their conditions.

**Important:** The dataset used in this project is synthetic. It was generated for educational purposes and is not real agricultural evidence.

## 2. Requirements and Environment

- **Python version tested:** Python 3.12.10
- **Operating system:** Windows with PowerShell
- **Main technologies:** Python, pandas, NumPy, scikit-learn, Matplotlib, FastAPI, and pytest

### Install dependencies

Open PowerShell in the project directory and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `.venv` environment must already exist. If it does not, create it using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Reproduce All Results

Run the complete data-generation, training, evaluation, and reporting pipeline:

```powershell
.\.venv\Scripts\python.exe run_all.py --student-id "25/27201" --seed 7201
```

The pipeline generates the synthetic dataset, validates and processes the data, trains and evaluates models, and saves the resulting models, metrics, plots, fingerprint, predictions, and technical report.

The generated dataset contains 269 rows before duplicate removal and 266 rows after duplicate removal for the recorded seed and configuration.

## 4. Command-Line Predictions

Run a prediction using the valid example:

```powershell
.\.venv\Scripts\python.exe predict.py --json samples/valid.json
```

Check input validation using the invalid example:

```powershell
.\.venv\Scripts\python.exe predict.py --json samples/invalid.json
```

The invalid example is intended to demonstrate input validation. An invalid request should be rejected rather than treated as a valid prediction.

## 5. REST API

Start the API server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation in a browser:

http://127.0.0.1:8000/docs

Available endpoints include:

- `GET /health` — checks whether the API is running.
- `GET /model-info` — provides model information.
- `POST /predict` — returns predictions for validated input.

Stop the server with `Ctrl + C` in the terminal running it.

## 6. Run Automated Tests

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The previously recorded test run passed all 8 tests, with one deprecation warning related to the test client. Run the command again if the code changes, and use the latest actual result in your submitted evidence.

## 7. Project Structure

- `app.py` — FastAPI application and API endpoints.
- `predict.py` — command-line prediction interface.
- `run_all.py` — complete pipeline entry point.
- `requirements.txt` — Python dependencies.
- `data/generate_data.py` — deterministic synthetic data generation.
- `data/AI_A2_2527201.csv` — generated dataset.
- `src/` — pipeline, inference, and report-generation helpers.
- `models/` — trained models and preprocessing artifacts.
- `artifacts/` — evaluation metrics, predictions, fingerprint, cluster report, and plots.
- `samples/` — valid and invalid example inputs.
- `tests/` — automated tests.
- `docs/` — generated technical report.
- `evidence/` — AI-use disclosure, test log, declaration, and demonstration evidence.

## 8. Repository and Submission Evidence

**GitHub repository:**  
https://github.com/christine-ndatiman/Musanze-Harvest-Intelligence-Service

**Branch:** `main`

**Final commit hash:** Obtain the latest hash after all final changes have been committed by running:

```powershell
git rev-parse HEAD
```

The submission ZIP should be named `AI_A2_2527201.zip`. It should contain the required source code, dataset, trained models, artifacts, report, tests, and evidence files. Do not include virtual environment folders.

The demonstration video must be recorded from the actual running project. Do not fabricate a video, test result, repository history, or other evidence.

## 9. Limitations and Responsible Use

This project uses synthetic data that was not collected from real farmers. Its predictions and evaluation results are for learning and prototyping only.

The system must not be used as the sole basis for decisions about credit, insurance, public services, or other high-impact opportunities. Any real agricultural use would require representative field data, external validation, model calibration, monitoring, security review, and consultation with qualified local agricultural experts.

The clustering results describe patterns in the generated dataset and should not be interpreted as established agricultural facts.