from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.inference import predict_record
ROOT=Path(__file__).resolve().parent
app=FastAPI(title='Musanze Harvest Intelligence Service',version='1.0.0',description='Synthetic-data classroom decision-support prototype.')
class FarmRecord(BaseModel):
    model_config=ConfigDict(extra='forbid')
    rainfall_mm: float=Field(ge=600,le=1400)
    avg_temp_c: float=Field(ge=12,le=26)
    soil_moisture_pct: float=Field(ge=20,le=80)
    fertilizer_kg: float=Field(ge=0,le=250)
    field_size_ha: float=Field(ge=0.2,le=4.0)
    season: str
    cooperative_support: int=Field(ge=0,le=1)
    @field_validator('season')
    @classmethod
    def season_allowed(cls,v):
        if v not in ('A','B'): raise ValueError("season must be 'A' or 'B'")
        return v
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/model-info')
def model_info():
    p=ROOT/'models'/'model_info.joblib'
    if not p.exists(): raise HTTPException(status_code=503,detail='Models not trained. Run run_all.py first.')
    info=joblib.load(p)
    return {'verification_code':info['verification_code'],'seed':info['seed'],'regression_model':info['selected_regression'],'classification_model':info['selected_classification'],'selected_k':info['selected_k'],'features':info['features'],'data_status':'synthetic'}
@app.post('/predict')
def predict(record:FarmRecord):
    if not (ROOT/'models'/'model_info.joblib').exists(): raise HTTPException(status_code=503,detail='Models not trained. Run run_all.py first.')
    try: return predict_record(record.model_dump(),ROOT)
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(status_code=422,detail=str(e))
