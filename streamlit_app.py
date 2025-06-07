import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Custom CSS for attractive styling ===
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .form-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 12px 40px rgba(240, 147, 251, 0.3);
    }
    
    .form-header {
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.3);
        animation: pulse 2s infinite;
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
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
st.set_page_config(
    page_title="AI Thyroid Cancer Predictor", 
    layout="wide",
    page_icon="🧬"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 AI-Powered Thyroid Cancer Predictor</h1>
    <p style="font-size: 1.2em; margin-top: 1rem;">
        Advanced Machine Learning for Differentiated Thyroid Cancer Outcome Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# Load dataset and model
df, label_encoders = load_data()

if df is not None:
    result = train_model(df)
    if len(result) == 5:
        model, acc, report, feature_names, target_col = result
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>📊 Model Accuracy</h2>
                <h1>{acc * 100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>📁 Dataset Size</h2>
                <h1>{len(df):,}</h1>
                <p>samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🎯 Features</h2>
                <h1>{len(feature_names)}</h1>
                <p>predictors</p>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance display (text-based)
        if model is not None:
            st.markdown("### 🎯 Most Important Features")
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            for i, row in feature_df.iterrows():
                st.write(f"**{row['Feature']}**: {row['Importance']:.3f}")

        # Patient Input Form
        st.markdown("""
        <div class="form-container">
            <div class="form-header">
                👩‍⚕️ Patient Information Form
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form", clear_on_submit=False):
            # Organize form fields in columns
            num_cols = 3
            cols = st.columns(num_cols)
            
            input_data = {}
            for i, feature in enumerate(feature_names):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    if feature in label_encoders:
                        options = list(label_encoders[feature].classes_)
                        value = st.selectbox(
                            f"🔹 {feature.replace('_', ' ').title()}", 
                            options, 
                            key=f"select_{feature}",
                            help=f"Select the appropriate {feature.lower()} value"
                        )
                        encoded = label_encoders[feature].transform([value])[0]
                        input_data[feature] = encoded
                    else:
                        input_data[feature] = st.number_input(
                            f"📊 {feature.replace('_', ' ').title()}", 
                            step=0.1, 
                            key=f"num_{feature}",
                            help=f"Enter the {feature.lower()} value"
                        )

            # Center the submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
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
                        <h1>{emoji} Prediction Result</h1>
                        <h2 style="margin: 1rem 0;">{label}</h2>
                        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-top: 2rem;">
                            <div>
                                <h3>Confidence Level</h3>
                                <h1>{confidence:.1%}</h1>
                            </div>
                            <div>
                                <h3>Risk Category</h3>
                                <h1>{'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}</h1>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability distribution (text-based)
                    if len(probability) == 2:
                        st.markdown("### 📈 Prediction Probabilities")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("No Recurrence", f"{probability[0]:.1%}")
                        with col2:
                            st.metric("Recurrence", f"{probability[1]:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

        # Dataset Preview
        with st.expander("📋 View Dataset Sample", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
        # Model Performance
        if report:
            with st.expander("📊 Model Performance Details", expanded=False):
                st.json(report)
            
        # Additional Information
        with st.expander("ℹ️ How This AI Model Works", expanded=False):
            st.markdown("""
            **🧠 Machine Learning Algorithm:** Random Forest Classifier
            
            **🎯 Purpose:** Predicts thyroid cancer recurrence risk based on patient clinical data
            
            **📊 Training Process:**
            1. Data preprocessing and cleaning
            2. Feature encoding for categorical variables
            3. Train/test split (80/20)
            4. Random Forest model training with 100 trees
            5. Model validation and performance evaluation
            
            **⚠️ Important Note:** This tool is for educational purposes only and should not replace professional medical consultation.
            """)
    
    else:
        st.error("❌ Model training failed. Please check your dataset.")
else:
    st.error("❌ Failed to load dataset. Please check the file path and format.")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧬 AI-Powered Thyroid Cancer Predictor | Built with Streamlit & Machine Learning</p>
    <p><em>For educational and research purposes only</em></p>
</div>
""", unsafe_allow_html=True)
