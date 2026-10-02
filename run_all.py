"""One-command reproducible training and artifact generation."""
import argparse, hashlib
from pathlib import Path
from data.generate_data import generate_data
from src.pipeline import validate_and_clean, train_all
from src.make_report import make_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--student-id',required=True); p.add_argument('--seed',required=True,type=int); a=p.parse_args()
    digits=''.join(ch for ch in a.student_id if ch.isdigit())
    if not digits: raise SystemExit('Student ID must contain digits.')
    expected=int(digits[-4:])
    if a.seed != expected: raise SystemExit(f'Seed rejected: seed must equal the last four digits of student ID ({expected}).')
    code=f'AI-A2-{expected:04d}'; root=Path(__file__).resolve().parent
    data_path=root/'data'/'AI_A2_2527201.csv'; raw,fingerprint=generate_data(a.seed,data_path)
    clean,validation=validate_and_clean(raw)
    (root/'artifacts').mkdir(exist_ok=True); (root/'artifacts'/'data_fingerprint.txt').write_text(f'Student ID: {a.student_id}\nVerification code: {code}\nSHA-256: {fingerprint}\nRows in generated CSV: {len(raw)}\nRows after duplicate removal: {len(clean)}\n')
    result=train_all(clean,a.seed,code,root/'artifacts')
    make_report(root, a.student_id, code, a.seed, fingerprint, validation, result)
    print(f'Student ID: {a.student_id}\nVerification code: {code}\nDataset SHA-256: {fingerprint}\nRows generated: {len(raw)}; rows after deduplication: {len(clean)}')
    print('Artifacts generated: artifacts/*.json, artifacts/test_predictions.csv, artifacts/data_fingerprint.txt, artifacts/plots/*.png, models/*.joblib, docs/AI_A2_2527201_REPORT.pdf')
    print('IMPORTANT: Run tests and personally review the generated metrics/report before submitting.')
if __name__=='__main__': main()
