import streamlit as st
import joblib
import pandas as pd


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("credit_card_approval_model.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef4ff 0%,
        #f7f3ff 45%,
        #edfaff 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #182848;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #5b6475;
    margin-bottom: 25px;
}

/* Section titles */

.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #243b6b;
    margin-top: 10px;
    margin-bottom: 15px;
}

/* Labels */

label {
    font-weight: 600 !important;
    color: #26354d !important;
}

/* Text inputs */

.stTextInput input {
    border-radius: 10px !important;
    border: 1px solid #d5dbea !important;
    background-color: white !important;
}

/* Number inputs */

.stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid #d5dbea !important;
    background-color: white !important;
}

/* Select boxes */

.stSelectbox > div > div {
    border-radius: 10px !important;
    background-color: white !important;
}

/* Buttons */

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    font-size: 19px;
    font-weight: 700;
    border: none;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    box-shadow: 0 7px 20px rgba(79, 70, 229, 0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.35);
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 14px;
    font-size: 17px;
    font-weight: 600;
}

/* Metrics */

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.07);
    border: 1px solid #e3e7f0;
}

/* Expander */

details {
    background: white;
    border-radius: 14px;
    border: 1px solid #e1e5ef;
    padding: 5px;
}

/* Footer */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 40px;
    padding-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">💳 Credit Card Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Credit Application Classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("💳 Project Information")

    st.write(
        "This application uses a Machine Learning model "
        "to classify credit card applications."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Logistic Regression")

    st.subheader("📊 Performance")

    st.write("Accuracy: **89.13%**")
    st.write("ROC-AUC: **0.96**")

    st.divider()

    st.caption("Dataset: UCI Credit Approval Dataset")

    st.caption(
        "Built with Python, Scikit-learn & Streamlit"
    )


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the applicant details below to generate a prediction."
)


# =========================================================
# APPLICANT NAME
# =========================================================

user_name = st.text_input(
    "👤 Applicant Name",
    placeholder="Enter applicant name"
)


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["a", "b"],
        help="Anonymized categorical value from the UCI dataset."
    )


with col2:

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )


with col3:

    debt = st.number_input(
        "Debt",
        min_value=0.0,
        value=0.0,
        step=0.1
    )


# =========================================================
# ROW 2
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    married = st.selectbox(
        "Married",
        ["u", "y", "l"],
        help="Anonymized categorical value."
    )


with col2:

    bank_customer = st.selectbox(
        "Bank Customer",
        ["g", "p", "gg"],
        help="Anonymized categorical value."
    )


with col3:

    education = st.text_input(
        "Education Level",
        value="c",
        help="Enter a categorical code from the dataset."
    )


# =========================================================
# ROW 3
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    ethnicity = st.text_input(
        "Ethnicity",
        value="v",
        help="Enter a categorical code from the dataset."
    )


with col2:

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        value=2.0,
        step=0.1
    )


with col3:

    prior_default = st.selectbox(
        "Prior Default",
        ["t", "f"],
        help="Anonymized categorical value."
    )


# =========================================================
# ROW 4
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    employed = st.selectbox(
        "Employed",
        ["t", "f"],
        help="Anonymized categorical value."
    )


with col2:

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        value=2,
        step=1
    )


with col3:

    drivers_license = st.selectbox(
        "Driver's License",
        ["t", "f"],
        help="Anonymized categorical value."
    )


# =========================================================
# ROW 5
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    citizen = st.selectbox(
        "Citizen",
        ["g", "s", "p"],
        help="Anonymized categorical value."
    )


with col2:

    zip_code = st.text_input(
        "Zip Code",
        value="00000"
    )


with col3:

    income = st.number_input(
        "Income",
        min_value=0,
        value=1000,
        step=100
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍 Predict Credit Approval",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Check applicant name

    if user_name.strip() == "":

        st.warning(
            "⚠️ Please enter the applicant name."
        )

        st.stop()


    # Create applicant DataFrame

    applicant = pd.DataFrame([{

        "Gender": gender,

        "Age": age,

        "Debt": debt,

        "Married": married,

        "BankCustomer": bank_customer,

        "EducationLevel": education,

        "Ethnicity": ethnicity,

        "YearsEmployed": years_employed,

        "PriorDefault": prior_default,

        "Employed": employed,

        "CreditScore": credit_score,

        "DriversLicense": drivers_license,

        "Citizen": citizen,

        "ZipCode": zip_code,

        "Income": income

    }])


    # Model prediction

    prediction = model.predict(applicant)[0]


    # Probability of Class 1

    probability = model.predict_proba(applicant)[0][1]


    st.divider()


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"### 👋 Applicant: **{user_name}**"
    )


    if prediction == 1:

        st.success(
            "✅ Prediction: Class 1 (+)"
        )

    else:

        st.error(
            "❌ Prediction: Class 0 (-)"
        )


    # Probability

    st.metric(
        "Class 1 Probability",
        f"{probability * 100:.2f}%"
    )


    st.write("Probability Visualization")

    st.progress(float(probability))


    # Explanation

    if probability >= 0.75:

        st.info(
            "The model shows a high probability of Class 1."
        )

    elif probability >= 0.50:

        st.info(
            "The model shows a moderate probability of Class 1."
        )

    else:

        st.info(
            "The model shows a low probability of Class 1."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        "Logistic Regression"
    )


with col2:

    st.metric(
        "Accuracy",
        "89.13%"
    )


with col3:

    st.metric(
        "ROC-AUC",
        "0.96"
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.divider()

with st.expander("📚 About the Dataset"):

    st.write(
        """
        **UCI Credit Approval Dataset**

        The project uses the UCI Credit Approval dataset
        for credit application classification.

        • Instances: 690

        • Input attributes: 15

        • Target classes: + and -

        • Dataset contains categorical and numerical attributes.

        • Missing values were handled during preprocessing.

        • Categorical features were encoded using One-Hot Encoding.

        • Numerical missing values were handled using median imputation.

        • Categorical missing values were handled using most-frequent imputation.
        """
    )


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("⚙️ How This Application Works"):

    st.write(
        """
        **Step 1:** User enters applicant information.

        **Step 2:** The trained preprocessing pipeline
        processes the input data.

        **Step 3:** Logistic Regression predicts the target class.

        **Step 4:** The application displays the predicted
        class and Class 1 probability.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '💳 Credit Card Approval Prediction | '
    'Built with Python + Scikit-learn + Streamlit 🤖'
    '</div>',
    unsafe_allow_html=True
)



