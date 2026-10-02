import json, subprocess, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference import validate_record
VALID=json.loads((ROOT/'samples'/'valid.json').read_text())
def test_valid_input(): assert validate_record(VALID)['season']=='A'
def test_missing_field():
    x=dict(VALID); x.pop('rainfall_mm')
    with pytest.raises(ValueError): validate_record(x)
def test_wrong_type():
    x=dict(VALID); x['rainfall_mm']='wet'
    with pytest.raises(TypeError): validate_record(x)
def test_range_error():
    x=dict(VALID); x['rainfall_mm']=2000
    with pytest.raises(ValueError): validate_record(x)
def test_invalid_season():
    x=dict(VALID); x['season']='C'
    with pytest.raises(ValueError): validate_record(x)
def test_extra_field():
    x=dict(VALID); x['yield_tons']=20
    with pytest.raises(ValueError): validate_record(x)
def test_seed_rule_rejects_wrong_seed():
    r=subprocess.run([sys.executable,str(ROOT/'run_all.py'),'--student-id','ug2527201@ines.ac.rw','--seed','1234'],capture_output=True,text=True)
    assert r.returncode != 0 and 'Seed rejected' in (r.stdout+r.stderr)
def test_api_health():
    from fastapi.testclient import TestClient
    import app
    c=TestClient(app.app)
    assert c.get('/health').status_code==200
