import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Main App Configuration (MUST BE FIRST) ===
st.set_page_config(
    page_title="AI Thyroid Cancer Predictor", 
    layout="centered",  # Changed from "wide" to "centered"
    page_icon="🧬"
)

# === Custom CSS for compact styling ===
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .form-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(240, 147, 251, 0.3);
    }
    
    .form-header {
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(79, 172, 254, 0.3);
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Compact form styling */
    .stSelectbox label, .stNumberInput label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.2rem !important;
    }
    
    .stSelectbox > div, .stNumberInput > div {
        margin-bottom: 0.8rem !important;
    }
    
    /* Reduce spacing */
    .stForm {
        padding: 0.5rem 0 !important;
    }
    
    /* Compact metrics */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# === Load and preprocess the dataset ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("differentiated+thyroid+cancer+recurrence/Thyroid_Diff.csv")
        df = df.dropna()

        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        return df, label_encoders
    except FileNotFoundError:
        st.error("🚨 Dataset file not found. Please check the file path.")
        return None, None

# === Train the model ===
def train_model(df):
    possible_targets = ['Outcome', 'Recurred', 'Class', 'Target', 'Label', 'Result', 'Recurrence']
    
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        if len(df) < 10:
            st.error("Dataset too small for training")
            return None, 0, {}, [], None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            report = classification_report(y_test, y_pred, output_dict=True)
        except Exception:
            report = {"accuracy": acc}

        return model, acc, report, list(X.columns), target_col
    
    except Exception as e:
        st.error(f"❌ Error in model training: {e}")
        return None, 0, {}, [], None

# === Main App ===

# Compact Header
st.markdown("""
<div class="main-header">
    <h2>🧬 AI Thyroid Cancer Predictor</h2>
    <p style="font-size: 1em; margin-top: 0.5rem;">
        Machine Learning for Thyroid Cancer Outcome Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# Load dataset and model
df, label_encoders = load_data()

if df is not None:
    result = train_model(df)
    if len(result) == 5:
        model, acc, report, feature_names, target_col = result

        # Compact Patient Input Form
        st.markdown("""
        <div class="form-container">
            <div class="form-header">
                👩‍⚕️ Patient Information
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form", clear_on_submit=False):
            # Organize form fields in 3 columns for better space usage
            cols = st.columns(3)
            
            input_data = {}
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                
                with cols[col_idx]:
                    if feature in label_encoders:
                        options = list(label_encoders[feature].classes_)
                        value = st.selectbox(
                            f"{feature.replace('_', ' ').title()}", 
                            options, 
                            key=f"select_{feature}"
                        )
                        encoded = label_encoders[feature].transform([value])[0]
                        input_data[feature] = encoded
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            step=0.1, 
                            key=f"num_{feature}"
                        )

            # Center the submit button
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Predict Outcome", use_container_width=True)

            if submitted:
                try:
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]
                    confidence = max(probability)
                    
                    # Determine label and styling
                    unique_values = df[target_col].unique()
                    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                        label = "Recurrence Likely" if prediction == 1 else "No Recurrence Expected"
                        result_class = "prediction-positive" if prediction == 1 else "prediction-negative"
                        emoji = "⚠️" if prediction == 1 else "✅"
                    else:
                        label = str(prediction)
                        result_class = "prediction-result"
                        emoji = "🔍"
                    
                    st.markdown(f"""
                    <div class="prediction-result {result_class}">
                        <h2>{emoji} {label}</h2>
                        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                            <div>
                                <h4>Confidence</h4>
                                <h3>{confidence:.1%}</h3>
                            </div>
                            <div>
                                <h4>Risk Level</h4>
                                <h3>{'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}</h3>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability distribution in compact format
                    if len(probability) == 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("No Recurrence", f"{probability[0]:.1%}")
                        with col2:
                            st.metric("Recurrence", f"{probability[1]:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    else:
        st.error("❌ Model training failed. Please check your dataset.")
else:
    st.error("❌ Failed to load dataset. Please check the file path and format.")

# Compact Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 0.5rem; font-size: 0.9rem;">
    <p>🧬 AI Thyroid Cancer Predictor | <em>For educational purposes only</em></p>
</div>
""", unsafe_allow_html=True)
