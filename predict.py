import argparse, json
from pathlib import Path
import joblib
from src.inference import predict_record

def main():
    p=argparse.ArgumentParser(); p.add_argument('--json',required=True); a=p.parse_args(); root=Path(__file__).resolve().parent
    info=joblib.load(root/'models'/'model_info.joblib')
    record=json.loads(Path(a.json).read_text())
    try: print(json.dumps(predict_record(record,root,info),indent=2))
    except (ValueError,KeyError,TypeError) as e: raise SystemExit(f'Invalid input: {e}')
if __name__=='__main__': main()
