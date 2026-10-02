from pathlib import Path
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def make_report(root: Path, student_id: str, code: str, seed: int, fingerprint: str, validation: dict, result: dict):
    docs=root/'docs'; docs.mkdir(exist_ok=True); path=docs/'AI_A2_2527201_REPORT.pdf'
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=8,leading=10))
    story=[]
    def para(s,style='BodyText'): story.append(Paragraph(str(s),styles[style])); story.append(Spacer(1,6))
    para('Musanze Harvest Intelligence Service','Title'); para('SWE 3513 — Individual Assignment 2','Heading2')
    para(f'Student: Christine Ndatimana | Registration: 25/27201 | Student ID supplied: {student_id}'); para(f'Verification code: {code} | Random seed: {seed}'); para('This report is generated from the current run. Verify all outputs and add your own interpretation before submission.')
    para('1. Identity and decision problem','Heading1'); para('The stakeholder is a Musanze potato cooperative field officer. The prototype estimates harvest quantity, predicts a binary high-risk label, and groups farms with similar input conditions. The decision-support output is not a substitute for agronomist judgement.')
    para('2. Data and reproducibility','Heading1'); para(f'The data are synthetic. CSV SHA-256: {fingerprint}. Rows before duplicate removal: {validation["rows_before_deduplication"]}; duplicates removed: {validation["duplicate_rows_removed"]}; rows after cleaning: {validation["rows_after_deduplication"]}. Missing values found after deduplication: {json.dumps(validation["missing_values_after_deduplication"])}.')
    para('3. Pipeline design and leakage controls','Heading1'); para('A fixed 75/25 train/test split uses the personal seed. Numeric imputation/scaling and categorical imputation/one-hot encoding are contained in scikit-learn pipelines and fitted on training data for regression and classification. Yield and risk targets are excluded from model input. Clustering uses input features only; its selected final model is refit on all transformed input rows after K selection on the training split.')
    para('4. Regression','Heading1'); rr=result['regression']; rows=[['Model','MAE','RMSE','R²']]
    for name,met in rr.items():
        if isinstance(met,dict): rows.append([name,f'{met["mae"]:.3f}',f'{met["rmse"]:.3f}',f'{met["r2"]:.3f}'])
    t=Table(rows,repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)])); story.extend([t,Spacer(1,8)]); para(f'Selected regression model: {rr["selected_model"]}. Selection rule: {rr["selection_reason"]}'); para('Interpretation prompt: inspect artifacts/test_predictions.csv, find a row with a large absolute yield_error, and explain that error in your own words. Do not claim causes that the synthetic features cannot establish.')
    para('5. Classification','Heading1'); cr=result['classification']; rows=[['Model','Precision','Recall','F1','ROC-AUC']]
    for name,met in cr.items():
        if isinstance(met,dict): rows.append([name,f'{met["precision"]:.3f}',f'{met["recall"]:.3f}',f'{met["f1"]:.3f}',('N/A' if met['roc_auc'] is None else f'{met["roc_auc"]:.3f}')])
    t=Table(rows,repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)])); story.extend([t,Spacer(1,8)]); para(f'Selected classifier: {cr["selected_model"]}. {cr["selection_reason"]}'); para('Interpretation prompt: use the confusion matrix and test_predictions.csv to discuss one false positive or false negative. Explain the operational consequence cautiously.')
    para('6. Clustering','Heading1'); cl=result['clustering']; para(f'Silhouette scores: {json.dumps(cl["silhouette_scores_by_k"])}. Selected K={cl["selected_k"]}. {cl["selection_reason"]}')
    for prof in cl['profiles']: para(f'Cluster {prof["cluster"]}: n={prof["size"]}, mean rainfall={prof["mean_rainfall_mm"]}, mean soil moisture={prof["mean_soil_moisture_pct"]}, mean fertilizer={prof["mean_fertilizer_kg"]}, mean field size={prof["mean_field_size_ha"]}, support rate={prof["support_rate"]}. Write a grounded operational interpretation after inspecting this profile.')
    para('7. API, testing, and responsible AI','Heading1'); para('The service exposes GET /health, GET /model-info, and POST /predict. Input validation rejects missing fields, invalid categories, and values outside expected ranges. Run pytest -q and insert the actual passing-test count and evidence in your final version.'); para('Limitations: synthetic data do not represent measured Musanze farms; the generated risk label is defined from synthetic yield per hectare; model performance is not evidence of real-world performance. Do not use this tool to decide credit, insurance, or public-service eligibility. Human review by a qualified agricultural expert is required. With real data, obtain consent and governance approvals, validate representativeness, evaluate fairness and calibration, and monitor drift.')
    for fname in ['residuals.png','confusion_matrix.png','clusters.png']:
        f=root/'artifacts'/'plots'/fname
        if f.exists():
            story.append(PageBreak()); para(f'Appendix: {fname}','Heading1'); story.append(Image(str(f),width=440,height=250,kind='proportional'))
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=40).build(story)
