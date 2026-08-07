import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SmartFee Recommendation System",
    page_icon="🎓",
    layout="wide",
)


def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 28%),
                linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
        }
        .hero-card {
            background: linear-gradient(135deg, #071a3d 0%, #123a95 35%, #2563eb 100%);
            border-radius: 28px;
            padding: 2.2rem 2.2rem 1.7rem;
            box-shadow: 0 20px 45px rgba(37, 99, 235, 0.22);
            margin-bottom: 1rem;
            color: white;
            position: relative;
            overflow: hidden;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -20px -35px auto;
            width: 180px;
            height: 180px;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            filter: blur(2px);
        }
        .hero-card h1 {
            color: white !important;
            font-size: 2.35rem;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .hero-card p {
            color: #dbeafe;
            font-size: 1.02rem;
            margin-bottom: 0;
            max-width: 700px;
        }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            color: #eff6ff;
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.7rem;
            letter-spacing: 0.03em;
        }
        .section-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }
        .section-card h3, .section-card h4 {
            margin-top: 0;
            color: #0f172a;
        }
        .feature-card {
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        .result-card {
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-top: 0.8rem;
        }
        .result-eligible {
            background: linear-gradient(135deg, #ecfdf3 0%, #f0fdf4 100%);
            border: 1px solid #86efac;
        }
        .result-full {
            background: linear-gradient(135deg, #fef2f2 0%, #fff7ed 100%);
            border: 1px solid #fecaca;
        }
        .stButton > button {
            border-radius: 999px;
            padding: 0.7rem 1.3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            border: none;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.24);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()

st.markdown(
    """
    <div class="hero-card">
        <div class="badge">AI-Powered Decision Support</div>
        <h1>🎓 SmartFee</h1>
        <p>Deliver confident, modern fee concession decisions with a smarter view of student financial need.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="feature-card">
            <h4>⚡ Instant Evaluation</h4>
            <p>Get a recommendation quickly from a trained model with a polished, guided workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📊 Insight-Driven</h4>
            <p>Balance income, household burden, academic progress, and locality in one view.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🧠 Professional Interface</h4>
            <p>Designed to feel more like a decision dashboard than a basic data form.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-card">
        <h3>📌 Decision Support Overview</h3>
        <p>Use this dashboard to evaluate a student’s profile and determine whether a fee concession is appropriate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "xgboost_model.pkl")
    model_columns = joblib.load(base_dir / "model_columns.pkl")
    ohe_encoder = joblib.load(base_dir / "ohe_encoder.pkl")
    return model, model_columns, ohe_encoder


try:
    model, model_columns, ohe_encoder = load_artifacts()
    st.sidebar.success("✅ Model and preprocessors loaded successfully")
    st.sidebar.markdown("### How it works")
    st.sidebar.info(
        "The system studies household income, family structure, academic performance, and location to recommend whether a fee concession is suitable."
    )
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model files: {e}")
    st.stop()


st.markdown("<div class='section-card'><h3>📋 Student Profile Form</h3><p>Fill in the details below to generate a recommendation.</p></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='section-card'><h4>💵 Financial Metrics</h4></div>", unsafe_allow_html=True)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, value=22000.0, step=1000.0)
    savings = st.number_input("Monthly Savings (₹)", min_value=0.0, value=2300.0, step=500.0)
    medical_expenses = st.number_input("Medical Expenses (₹)", min_value=0.0, value=2000.0, step=500.0)
    debt_status = st.selectbox("Debt Status", ["No", "Yes"])

with col2:
    st.markdown("<div class='section-card'><h4>👨‍👩‍👧‍👦 Household Structure</h4></div>", unsafe_allow_html=True)
    household_size = st.number_input("Household Size", min_value=1, value=6, step=1)
    earning_members = st.number_input("Earning Members", min_value=1, value=1, step=1)
    school_children = st.number_input("School-Going Children", min_value=0, value=4, step=1)
    income_stability = st.selectbox("Income Stability", ["Stable", "Fluctuating"])
    house_type = st.selectbox("House Type", ["Rented", "Owned"])

with col3:
    st.markdown("<div class='section-card'><h4>🎓 Academic & Profile</h4></div>", unsafe_allow_html=True)
    attendance = st.slider("Attendance (%)", min_value=0.0, max_value=100.0, value=90.0)
    academic_performance = st.slider("Academic Performance Score", min_value=0.0, max_value=100.0, value=92.0)
    teacher_eval = st.slider("Teacher Evaluation (1 to 5)", min_value=1, max_value=5, value=5)
    parent_education = st.selectbox("Parent Education", ["Graduate", "Primary", "Secondary", "Post Graduate"])
    location = st.selectbox("Location", ["Semi-Urban", "Rural", "Urban"])

st.markdown("""
<div class="section-card">
    <h4>💡 What the model considers</h4>
    <p>Financial strain, household dependents, academic commitment, and locality all influence the final recommendation.</p>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Evaluate Fee Concession", type="primary", use_container_width=True):
    inc_stab_encoded = 1 if income_stability == "Stable" else 0
    house_type_encoded = 1 if house_type == "Owned" else 0
    debt_status_encoded = 1 if debt_status == "Yes" else 0

    parent_edu_map = {"Graduate": 0, "Primary": 1, "Secondary": 2, "Post Graduate": 3}
    parent_edu_encoded = parent_edu_map.get(parent_education, 0)

    input_df = pd.DataFrame(
        [{
            "Monthly_Income": monthly_income,
            "Household_Size": household_size,
            "Number_of_Earning_Members": earning_members,
            "Number_of_School_Going_Children": school_children,
            "Savings": savings,
            "Medical_Expenses": medical_expenses,
            "Attendance": attendance,
            "Academic_Performance": academic_performance,
            "Teacher_Evaluation": teacher_eval,
            "Income_Stability": inc_stab_encoded,
            "House_Type": house_type_encoded,
            "Parent_Education": parent_edu_encoded,
            "Debt_Status": debt_status_encoded,
        }]
    )

    loc_encoded = ohe_encoder.transform([[location]])
    loc_cols = ohe_encoder.get_feature_names_out(["Location"])
    loc_df = pd.DataFrame(loc_encoded, columns=loc_cols, index=input_df.index)

    final_input = pd.concat([input_df, loc_df], axis=1)
    final_input = final_input.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(final_input)[0]

    st.markdown("<h3>🎯 Recommendation Result</h3>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            """
            <div class="result-card result-full">
                <h4>❌ Recommendation: Pay Full Fee</h4>
                <p>The family profile suggests sufficient financial capacity to meet standard tuition fees.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card result-eligible">
                <h4>✅ Recommendation: Eligible for Fee Concession</h4>
                <p>The family profile indicates financial burden, so a fee concession is recommended.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("This recommendation is generated from the trained model and the information you provided.")