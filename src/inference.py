from pathlib import Path
import numpy as np, joblib
from data.generate_data import FEATURES
RANGES={"rainfall_mm":(600,1400),"avg_temp_c":(12,26),"soil_moisture_pct":(20,80),"fertilizer_kg":(0,250),"field_size_ha":(0.2,4.0),"cooperative_support":(0,1)}
def validate_record(record):
    if not isinstance(record,dict): raise TypeError('JSON input must be an object.')
    missing=[f for f in FEATURES if f not in record]
    extra=[f for f in record if f not in FEATURES]
    if missing: raise ValueError(f'Missing required field(s): {missing}')
    if extra: raise ValueError(f'Unexpected field(s): {extra}')
    for f,(lo,hi) in RANGES.items():
        v=record[f]
        if isinstance(v,bool) or not isinstance(v,(int,float)): raise TypeError(f'{f} must be numeric.')
        if not np.isfinite(v) or v<lo or v>hi: raise ValueError(f'{f} must be between {lo} and {hi}.')
    if record['cooperative_support'] not in (0,1): raise ValueError('cooperative_support must be 0 or 1.')
    if record['season'] not in ('A','B'): raise ValueError('season must be A or B.')
    return {k:record[k] for k in FEATURES}
def predict_record(record,root:Path,info=None):
    row=validate_record(record); root=Path(root); info=info or joblib.load(root/'models'/'model_info.joblib')
    import pandas as pd
    X=pd.DataFrame([row],columns=FEATURES)
    reg=joblib.load(root/'models'/'regression.joblib'); clf=joblib.load(root/'models'/'classification.joblib'); prep=joblib.load(root/'models'/'cluster_preprocessor.joblib'); km=joblib.load(root/'models'/'kmeans.joblib')
    prob=float(clf.predict_proba(X)[0,1]); risk=int(clf.predict(X)[0]); z=prep.transform(X)
    if hasattr(z,'toarray'): z=z.toarray()
    cluster=int(km.predict(z)[0])
    return {"yield_estimate_tons":round(float(reg.predict(X)[0]),4),"high_risk_prediction":risk,"high_risk_probability":round(prob,4),"cluster":cluster,"verification_code":info['verification_code']}
