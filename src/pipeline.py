"""Validation, modeling, evaluation, and artifact generation."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, silhouette_score
from data.generate_data import FEATURES, COLUMNS

NUMERIC = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "cooperative_support"]
CATEGORICAL = ["season"]

def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_cols = sorted(set(COLUMNS) - set(df.columns))
    if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")
    duplicates = int(df.duplicated().sum())
    clean = df.drop_duplicates().copy()
    missing_before = {c: int(v) for c,v in clean.isna().sum().items() if v}
    # Required features are imputed by train-fitted pipelines; targets cannot be missing.
    if clean[["yield_tons", "high_risk"]].isna().any().any():
        raise ValueError("Target columns must not contain missing values.")
    checks = {
      "rainfall_mm": (600,1400), "avg_temp_c": (12,26), "soil_moisture_pct": (20,80),
      "fertilizer_kg": (0,250), "field_size_ha": (0.2,4.0), "cooperative_support": (0,1), "yield_tons": (0.5,None)}
    for col,(lo,hi) in checks.items():
        vals=clean[col].dropna()
        if (vals < lo).any() or (hi is not None and (vals > hi).any()): raise ValueError(f"Out-of-range values in {col}.")
    if not set(clean['season'].dropna().unique()).issubset({'A','B'}): raise ValueError('season must be A or B.')
    if not set(clean['high_risk'].unique()).issubset({0,1}): raise ValueError('high_risk must be binary 0/1.')
    return clean, {"rows_before_deduplication":len(df),"duplicate_rows_removed":duplicates,"rows_after_deduplication":len(clean),"missing_values_after_deduplication":missing_before,"target_leakage_features_excluded":["yield_tons","high_risk","yield_per_ha"]}

def _preprocessor():
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),NUMERIC),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)])

def train_all(df: pd.DataFrame, seed: int, verification_code: str, out: Path):
    out.mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(exist_ok=True); (out.parent/'models').mkdir(exist_ok=True)
    row_ids=np.arange(len(df)); X=df[FEATURES].copy(); y=df['yield_tons'].astype(float); risk=df['high_risk'].astype(int)
    idx_train,idx_test=train_test_split(row_ids,test_size=0.25,random_state=seed)
    Xtr,Xte=X.iloc[idx_train],X.iloc[idx_test]; ytr,yte=y.iloc[idx_train],y.iloc[idx_test]; rtr,rte=risk.iloc[idx_train],risk.iloc[idx_test]
    reg_models={"mean_baseline":DummyRegressor(strategy='mean'),"ridge":Ridge(alpha=1.0),"random_forest":RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=seed)}
    reg_results={}; fitted_reg={}
    for name,model in reg_models.items():
        pipe=Pipeline([('prep',_preprocessor()),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        reg_results[name]={"mae":float(mean_absolute_error(yte,pred)),"rmse":float(np.sqrt(mean_squared_error(yte,pred))),"r2":float(r2_score(yte,pred))}
        fitted_reg[name]=pipe
    selected_reg=min([n for n in reg_results if n!='mean_baseline'],key=lambda n:reg_results[n]['rmse'])
    reg_results['selected_model']=selected_reg; reg_results['selection_reason']='Lowest held-out RMSE among the two candidate models.'
    clf_models={"majority_baseline":DummyClassifier(strategy='most_frequent'),"logistic_regression":LogisticRegression(max_iter=1500,random_state=seed),"random_forest":RandomForestClassifier(n_estimators=160,min_samples_leaf=2,random_state=seed,class_weight='balanced')}
    clf_results={}; fitted_clf={}; probs={}
    for name,model in clf_models.items():
        pipe=Pipeline([('prep',_preprocessor()),('model',model)]); pipe.fit(Xtr,rtr); pred=pipe.predict(Xte)
        prob=pipe.predict_proba(Xte)[:,1] if hasattr(pipe[-1],'predict_proba') and len(np.unique(rtr))==2 else None
        met={"precision":float(precision_score(rte,pred,zero_division=0)),"recall":float(recall_score(rte,pred,zero_division=0)),"f1":float(f1_score(rte,pred,zero_division=0)),"confusion_matrix":confusion_matrix(rte,pred,labels=[0,1]).tolist(),"roc_auc":float(roc_auc_score(rte,prob)) if prob is not None and len(np.unique(rte))==2 else None}
        clf_results[name]=met; fitted_clf[name]=pipe; probs[name]=prob
    selected_clf=max([n for n in clf_results if n!='majority_baseline'],key=lambda n:clf_results[n]['f1'])
    clf_results['selected_model']=selected_clf; clf_results['selection_reason']='Highest held-out F1 among the two candidate classifiers.'
    # Clustering uses features only (no targets), with train-only imputation/scaling.
    cluster_prep=_preprocessor(); cluster_prep.fit(Xtr); Ztr=cluster_prep.transform(Xtr); Zte=cluster_prep.transform(Xte); Zall=cluster_prep.transform(X)
    if hasattr(Zall,'toarray'): Zall=Zall.toarray()
    if hasattr(Ztr,'toarray'): Ztr=Ztr.toarray()
    scores={}; candidates={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,random_state=seed,n_init=10); labels=km.fit_predict(Ztr)
        score=float(silhouette_score(Ztr,labels)) if len(set(labels))>1 else -1.0
        scores[str(k)]=score; candidates[k]=km
    best_k=max(range(2,7),key=lambda k:scores[str(k)])
    # Refit selected K on all transformed feature rows for final inference; K selection evidence remains train-only.
    final_km=KMeans(n_clusters=best_k,random_state=seed,n_init=10).fit(Zall)
    labels_all=final_km.labels_
    # Cluster profiles are means of original feature values, ignoring missing values.
    prof=pd.DataFrame({c:df[c] for c in NUMERIC}); prof['season']=df['season']; prof['cluster']=labels_all
    profiles=[]
    for cl,g in prof.groupby('cluster'):
        profiles.append({"cluster":int(cl),"size":int(len(g)),"mean_rainfall_mm":round(float(g.rainfall_mm.mean()),2),"mean_temp_c":round(float(g.avg_temp_c.mean()),2),"mean_soil_moisture_pct":round(float(g.soil_moisture_pct.mean()),2),"mean_fertilizer_kg":round(float(g.fertilizer_kg.mean()),2),"mean_field_size_ha":round(float(g.field_size_ha.mean()),2),"support_rate":round(float(g.cooperative_support.mean()),3),"season_A_rate":round(float((g.season=='A').mean()),3),"interpretation":"Describe this group using its average rainfall, soil moisture, fertilizer, field size, and support rate; validate with local agronomy expertise."})
    cluster_report={"silhouette_scores_by_k":scores,"selected_k":best_k,"selection_reason":"Highest silhouette score among K=2 through K=6 on the training split.","profiles":profiles,"features_used":FEATURES,"targets_excluded":True}
    # Save final models and preprocessor. Regression/classification pipelines include their own preprocessing.
    models=out.parent/'models'; joblib.dump(fitted_reg[selected_reg],models/'regression.joblib'); joblib.dump(fitted_clf[selected_clf],models/'classification.joblib'); joblib.dump(cluster_prep,models/'cluster_preprocessor.joblib'); joblib.dump(final_km,models/'kmeans.joblib')
    joblib.dump({"seed":seed,"verification_code":verification_code,"selected_regression":selected_reg,"selected_classification":selected_clf,"selected_k":best_k,"features":FEATURES},models/'model_info.joblib')
    # Predictions on held-out rows.
    reg_pred=fitted_reg[selected_reg].predict(Xte); clf_pred=fitted_clf[selected_clf].predict(Xte)
    clf_prob=fitted_clf[selected_clf].predict_proba(Xte)[:,1] if len(np.unique(rtr))==2 else np.full(len(Xte),np.nan)
    test_clusters=final_km.predict(Zte)
    predictions=pd.DataFrame({"row_id":idx_test,"actual_yield_tons":yte.to_numpy(),"predicted_yield_tons":reg_pred,"yield_error":yte.to_numpy()-reg_pred,"actual_high_risk":rte.to_numpy(),"predicted_high_risk":clf_pred,"high_risk_probability":clf_prob,"cluster":test_clusters})
    predictions.to_csv(out/'test_predictions.csv',index=False,float_format='%.4f')
    (out/'regression_metrics.json').write_text(json.dumps(reg_results,indent=2)); (out/'classification_metrics.json').write_text(json.dumps(clf_results,indent=2)); (out/'clustering_report.json').write_text(json.dumps(cluster_report,indent=2))
    # plots
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(yte,reg_pred); ax.set_xlabel('Actual yield (tons)'); ax.set_ylabel('Predicted yield (tons)'); ax.set_title('Held-out regression predictions'); fig.tight_layout(); fig.savefig(out/'plots'/'residuals.png',dpi=140); plt.close(fig)
    cm=np.array(clf_results[selected_clf]['confusion_matrix']); fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm); ax.set_xticks([0,1],labels=['Predicted low risk','Predicted high risk']); ax.set_yticks([0,1],labels=['Actual low risk','Actual high risk']); ax.set_title('Confusion matrix');
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.tight_layout(); fig.savefig(out/'plots'/'confusion_matrix.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); scatter=ax.scatter(Zall[:,0],Zall[:,1],c=labels_all,s=18); ax.set_title(f'Farm clusters (K={best_k}; first two transformed dimensions)'); ax.set_xlabel('Transformed feature 1'); ax.set_ylabel('Transformed feature 2'); fig.tight_layout(); fig.savefig(out/'plots'/'clusters.png',dpi=140); plt.close(fig)
    return {"regression":reg_results,"classification":clf_results,"clustering":cluster_report,"predictions":predictions,"models_dir":str(models)}
