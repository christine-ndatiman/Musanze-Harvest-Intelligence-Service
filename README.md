# Musanze Harvest Intelligence Service

**Student:** Christine Ndatimana  
**Registration number:** 25/27201  
**Institutional username:** ug2527201@ines.ac.rw *(verify this format against your official INES account before submission)*  
**Verification code:** AI-A2-7201  
**Stakeholder:** Musanze potato cooperative field officer

## Purpose
A classroom decision-support prototype that estimates potato harvest quantity, flags high production risk, and groups farms by conditions. The dataset is synthetic and must not be treated as real agronomic evidence.

## Requirements
Python 3.10 or newer is recommended. Tested OS: record the operating system you actually test on. Install dependencies with:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows Command Prompt: .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Reproduce all results
Replace the example student ID only if your official institutional username differs:

```bash
python run_all.py --student-id ug2527201@ines.ac.rw --seed 7201
```

The command generates the CSV, cleans and validates data, fits the models, evaluates them, saves plots/reports/models, and creates `docs/AI_A2_2527201_REPORT.pdf`.

## CLI prediction
```bash
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
```

## API
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs`. Endpoints: `GET /health`, `GET /model-info`, and `POST /predict`.

## Tests
```bash
pytest -q
```

## Project structure
- `data/generate_data.py`: deterministic student-specific data generation
- `src/`: validation, preprocessing, training, inference, and report helpers
- `models/`: fitted preprocessing/model artifacts created by `run_all.py`
- `artifacts/`: metrics, predictions, fingerprint, plots and cluster report
- `samples/`: valid and invalid example input
- `tests/`: automated tests
- `docs/`: generated technical report
- `evidence/`: add your own demo video, test-log PDF, declaration PDF, and AI-use disclosure

## Repository and submission evidence
Repository URL: add your actual repository URL.  
Final commit hash: record it using `git rev-parse HEAD` after committing.  
Do not invent a repository, commit hash, test run, or demonstration evidence.

## Limitations and responsible use
The dataset is synthetic, not collected from farmers. Results are for learning and prototyping only. Do not use the predictions to decide access to credit, insurance, public services, or other high-impact opportunities. An agronomist must review any real-world decision. Real deployment would require representative field data, external validation, calibration, monitoring, security review, and consultation with local agricultural experts.
