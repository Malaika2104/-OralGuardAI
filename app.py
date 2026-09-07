import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HN Cancer Dashboard",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.big-num   { font-size:2.8rem; font-weight:700; line-height:1.1; }
.label-txt { font-size:0.78rem; text-transform:uppercase;
             letter-spacing:.06em; color:#94a3b8; margin-bottom:4px; }
.card      { background:#1e2130; border-radius:14px; padding:22px 26px;
             border-left:5px solid; margin-bottom:6px; }
.card-teal { border-color:#10b981; }
.card-red  { border-color:#ef4444; }
.bar-wrap  { margin-bottom:9px; }
.bar-label { display:flex; justify-content:space-between;
             font-size:.82rem; color:#cbd5e0; margin-bottom:3px; }
.bar-bg    { background:#2d3748; border-radius:4px; height:11px; }
.bar-fill-teal { background:linear-gradient(90deg,#10b981,#3b82f6);
                 height:11px; border-radius:4px; }
.bar-fill-red  { background:linear-gradient(90deg,#ef4444,#f97316);
                 height:11px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─── FEATURE LISTS ──────────────────────────────────────────────────────────
CLINICAL = [
    'year_of_initial_diagnosis','age_at_initial_diagnosis','sex','smoking_status',
    'primarily_metastasis','days_to_last_information','first_treatment_intent',
    'first_treatment_modality','days_to_first_treatment','adjuvant_treatment_intent',
    'adjuvant_radiotherapy','adjuvant_radiotherapy_modality','adjuvant_systemic_therapy',
    'adjuvant_systemic_therapy_modality','adjuvant_radiochemotherapy'
]
PATHOLOGY = [
    'primary_tumor_site','pT_stage','pN_stage','grading','hpv_association_p16',
    'number_of_positive_lymph_nodes','number_of_resected_lymph_nodes','perinodal_invasion',
    'lymphovascular_invasion_L','vascular_invasion_V','perineural_invasion_Pn',
    'resection_status','resection_status_carcinoma_in_situ','carcinoma_in_situ',
    'closest_resection_margin_in_cm','histologic_type','infiltration_depth_in_mm'
]
BLOOD = [
    'Basophils','Basophils %','CRP','Calcium','Chloride','Creatinine','Eosinophils',
    'Eosinophils %','Erythrocytes','Glomerular filtration rate','Glucose','Granulocytes',
    'Granulocytes %','Hematocrit','Hemoglobin','INR','Leukocytes','Lymphocytes',
    'Lymphocytes %','MCH','MCV','MHCH','MPV','Magnesium','Monocytes','Monocytes %',
    'Platelets','Potassium','RDW','Sodium','Urea','aPPT'
]
ALL_FEATURES = CLINICAL + PATHOLOGY + BLOOD

NICE_NAMES = {
    'year_of_initial_diagnosis':'Year of Diagnosis',
    'age_at_initial_diagnosis':'Age at Diagnosis',
    'sex':'Sex','smoking_status':'Smoking Status',
    'primarily_metastasis':'Primary Metastasis',
    'days_to_last_information':'Days to Last Info',
    'first_treatment_intent':'Treatment Intent',
    'first_treatment_modality':'Treatment Modality',
    'days_to_first_treatment':'Days to 1st Treatment',
    'adjuvant_treatment_intent':'Adjuvant Intent',
    'adjuvant_radiotherapy':'Adjuvant Radiotherapy',
    'adjuvant_radiotherapy_modality':'RT Modality',
    'adjuvant_systemic_therapy':'Systemic Therapy',
    'adjuvant_systemic_therapy_modality':'Systemic Modality',
    'adjuvant_radiochemotherapy':'Radiochemotherapy',
    'primary_tumor_site':'Tumor Site',
    'pT_stage':'Tumor Stage (pT)','pN_stage':'Node Stage (pN)',
    'grading':'Tumor Grading','hpv_association_p16':'HPV/p16 Status',
    'number_of_positive_lymph_nodes':'Positive Lymph Nodes',
    'number_of_resected_lymph_nodes':'Resected Lymph Nodes',
    'perinodal_invasion':'Perinodal Invasion',
    'lymphovascular_invasion_L':'Lymphovascular Invasion',
    'vascular_invasion_V':'Vascular Invasion',
    'perineural_invasion_Pn':'Perineural Invasion',
    'resection_status':'Resection Status',
    'resection_status_carcinoma_in_situ':'Resection (CIS)',
    'carcinoma_in_situ':'Carcinoma in Situ',
    'closest_resection_margin_in_cm':'Closest Margin (cm)',
    'histologic_type':'Histologic Type',
    'infiltration_depth_in_mm':'Infiltration Depth (mm)',
    'Basophils':'Basophils','Basophils %':'Basophils %',
    'CRP':'CRP','Calcium':'Calcium','Chloride':'Chloride',
    'Creatinine':'Creatinine','Eosinophils':'Eosinophils',
    'Eosinophils %':'Eosinophils %','Erythrocytes':'Erythrocytes',
    'Glomerular filtration rate':'GFR','Glucose':'Glucose',
    'Granulocytes':'Granulocytes','Granulocytes %':'Granulocytes %',
    'Hematocrit':'Hematocrit','Hemoglobin':'Hemoglobin','INR':'INR',
    'Leukocytes':'Leukocytes','Lymphocytes':'Lymphocytes',
    'Lymphocytes %':'Lymphocytes %','MCH':'MCH','MCV':'MCV',
    'MHCH':'MHCH','MPV':'MPV','Magnesium':'Magnesium',
    'Monocytes':'Monocytes','Monocytes %':'Monocytes %',
    'Platelets':'Platelets','Potassium':'Potassium','RDW':'RDW',
    'Sodium':'Sodium','Urea':'Urea','aPPT':'aPTT',
}

# ─── LOAD & TRAIN (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Gradient Boosting models on HANCOCK data…")
def load_and_train():
    df = pd.read_csv("patients.csv")
    df.columns = df.columns.str.strip()

    X = df[ALL_FEATURES].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == 'string']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    y_surv = (df['survival_status'] == 'deceased').astype(int)
    y_rec  = (df['recurrence'] == 'yes').astype(int)

    gb_s = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42)
    gb_r = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42)
    gb_s.fit(X_imp, y_surv)
    gb_r.fit(X_imp, y_rec)

    return df, gb_s, gb_r, le_dict, imp, cat_cols

df, gb_s, gb_r, le_dict, imp, cat_cols = load_and_train()

# ─── PREDICT ────────────────────────────────────────────────────────────────
def predict(patient_row):
    xp = pd.DataFrame([patient_row[ALL_FEATURES]])
    for col in cat_cols:
        xp[col] = xp[col].astype(str)
        if col in le_dict:
            try:    xp[col] = le_dict[col].transform(xp[col])
            except: xp[col] = 0
        else:
            xp[col] = 0
    xp_imp = imp.transform(xp)

    sp = gb_s.predict_proba(xp_imp)[0]   # [not deceased, deceased]
    rp = gb_r.predict_proba(xp_imp)[0]   # [no recurrence, recurrence]

    living_pct = round((1 - sp[1]) * 100, 1)
    rec_pct    = round(rp[1] * 100, 1)

    fi_s = sorted(zip(ALL_FEATURES, gb_s.feature_importances_), key=lambda x: -x[1])[:10]
    fi_r = sorted(zip(ALL_FEATURES, gb_r.feature_importances_), key=lambda x: -x[1])[:10]

    top_s = [(NICE_NAMES.get(f, f), round(v*100, 2)) for f, v in fi_s]
    top_r = [(NICE_NAMES.get(f, f), round(v*100, 2)) for f, v in fi_r]

    return living_pct, rec_pct, top_s, top_r

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("## 🧬 Head & Neck Cancer Prognostic Dashboard")
st.caption("FYP2025-HealthTech · UET Peshawar · HANCOCK Dataset · Gradient Boosting Model")
st.divider()

# ─── TOP METRICS ────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Patients", len(df))
m2.metric("Avg Age at Diagnosis", round(df["age_at_initial_diagnosis"].mean(), 1))
m3.metric("Overall Survival Rate",
          f"{round(df['survival_status'].eq('living').mean()*100,1)}%")
m4.metric("Recurrence Rate",
          f"{round(df['recurrence'].eq('yes').mean()*100,1)}%")
st.divider()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Patient Lookup")
st.sidebar.caption("Enter Patient ID to get predictions")
pid = st.sidebar.text_input("Patient ID", placeholder="e.g. 1, 42, 250")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Gradient Boosting")
st.sidebar.markdown("**Modalities:** Clinical · Pathological · Blood")
st.sidebar.markdown(f"**Features used:** {len(ALL_FEATURES)}")
st.sidebar.markdown(f"**Training samples:** {len(df)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset split**")
st.sidebar.write(f"Living: {df['survival_status'].eq('living').sum()} patients")
st.sidebar.write(f"Deceased: {df['survival_status'].eq('deceased').sum()} patients")
st.sidebar.write(f"Recurrence Yes: {df['recurrence'].eq('yes').sum()} patients")
st.sidebar.write(f"Recurrence No: {df['recurrence'].eq('no').sum()} patients")

# ─── MAIN: PATIENT RESULTS ──────────────────────────────────────────────────
if pid:
    pid = pid.strip()
    patient = df[df["patient_id"].astype(str) == pid]

    if patient.empty:
        st.error(f"❌ Patient ID **{pid}** not found. Valid IDs: 1 – {len(df)}")
    else:
        p = patient.iloc[0]

        with st.spinner("Running Gradient Boosting predictions…"):
            living_pct, rec_pct, top_s, top_r = predict(p)

        st.subheader(f"👤 Patient {pid} — Gradient Boosting Prediction")

        # ── Prediction cards ──────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card card-teal">
                <div class="label-txt">Survival Probability</div>
                <div class="big-num" style="color:#10b981">{living_pct}%</div>
                <div style="color:#64748b;font-size:.88rem;margin-top:6px">
                    Probability patient is <b>living</b> at last follow-up
                </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            rc = "#10b981" if rec_pct < 20 else "#f59e0b" if rec_pct <= 45 else "#ef4444"
            st.markdown(f"""
            <div class="card card-red">
                <div class="label-txt">Recurrence Risk</div>
                <div class="big-num" style="color:{rc}">{rec_pct}%</div>
                <div style="color:#64748b;font-size:.88rem;margin-top:6px">
                    Probability of cancer <b>recurrence</b> after treatment
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Risk badge ────────────────────────────────────────────────────
        if rec_pct < 20:
            st.success("🟢 LOW RECURRENCE RISK")
        elif rec_pct <= 45:
            st.warning("🟡 MODERATE RECURRENCE RISK")
        else:
            st.error("🔴 HIGH RECURRENCE RISK")

        st.divider()

        # ── Feature importance bars ───────────────────────────────────────
        fi1, fi2 = st.columns(2)

        with fi1:
            st.markdown("##### 📊 Top Factors → Survival Prediction")
            max_s = top_s[0][1] if top_s else 1
            for name, score in top_s:
                w = int((score / max_s) * 100)
                st.markdown(f"""
                <div class="bar-wrap">
                  <div class="bar-label">
                    <span>{name}</span>
                    <span style="color:#10b981">{score:.1f}%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill-teal" style="width:{w}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with fi2:
            st.markdown("##### 📊 Top Factors → Recurrence Prediction")
            max_r = top_r[0][1] if top_r else 1
            for name, score in top_r:
                w = int((score / max_r) * 100)
                st.markdown(f"""
                <div class="bar-wrap">
                  <div class="bar-label">
                    <span>{name}</span>
                    <span style="color:#ef4444">{score:.1f}%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill-red" style="width:{w}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Clinical Summary ──────────────────────────────────────────────
        st.subheader("🧾 Clinical Summary")
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("**Patient Info**")
            st.write(f"🎂 Age: **{p.get('age_at_initial_diagnosis','N/A')}**")
            st.write(f"⚧ Sex: **{p.get('sex','N/A')}**")
            st.write(f"🚬 Smoking: **{p.get('smoking_status','N/A')}**")
            st.write(f"📅 Year: **{p.get('year_of_initial_diagnosis','N/A')}**")

        with s2:
            st.markdown("**Tumor Info**")
            st.write(f"📍 Site: **{p.get('primary_tumor_site','N/A')}**")
            st.write(f"🔬 pT Stage: **{p.get('pT_stage','N/A')}**")
            st.write(f"🔗 pN Stage: **{p.get('pN_stage','N/A')}**")
            st.write(f"⭐ Grading: **{p.get('grading','N/A')}**")
            st.write(f"🧫 HPV/p16: **{p.get('hpv_association_p16','N/A')}**")

        with s3:
            st.markdown("**Actual Outcomes (Ground Truth)**")
            surv_a = p.get('survival_status','N/A')
            rec_a  = p.get('recurrence','N/A')
            st.write(f"{'✅' if surv_a=='living' else '⚠️'} Status: **{surv_a}**")
            st.write(f"{'⚠️' if rec_a=='yes' else '✅'} Recurrence: **{rec_a}**")
            st.write(f"🔪 Resection: **{p.get('resection_status','N/A')}**")
            st.write(f"🔴 Metastasis: **{p.get('primarily_metastasis','N/A')}**")

        st.divider()

        # ── Blood values ──────────────────────────────────────────────────
        with st.expander("🩸 Blood Test Values", expanded=False):
            bdata = {NICE_NAMES.get(f,f): p.get(f, np.nan) for f in BLOOD}
            bdf = pd.DataFrame(list(bdata.items()), columns=["Marker","Value"])
            bdf["Value"] = bdf["Value"].apply(
                lambda x: round(float(x),3) if pd.notna(x) else "N/A")
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        with st.expander("📄 Full Patient Record", expanded=False):
            st.dataframe(patient, use_container_width=True)

else:
    st.info("👈 Enter a **Patient ID** in the sidebar (e.g. 1, 42, 150) to see predictions.")

    tab1, tab2 = st.tabs(["📋 Dataset Preview", "📊 Distribution"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            sv = df['survival_status'].value_counts().reset_index()
            sv.columns = ['Status','Count']
            st.markdown("**Survival Status**")
            st.dataframe(sv, use_container_width=True, hide_index=True)
        with col_b:
            rv = df['recurrence'].value_counts().reset_index()
            rv.columns = ['Recurrence','Count']
            st.markdown("**Recurrence**")
            st.dataframe(rv, use_container_width=True, hide_index=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧬 FYP2025-HealthTech · Head & Neck Cancer Outcome Prediction · UET Peshawar · HANCOCK Dataset · Gradient Boosting Classifier")
