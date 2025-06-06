import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Load and preprocess the dataset ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("differentiated+thyroid+cancer+recurrence/Thyroid_Diff.csv")
        st.write("**Dataset columns:**", list(df.columns))  # Debug: Show column names
        df = df.dropna()

        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        return df, label_encoders
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return None, None

# === Train the model ===
def train_model(df):
    # Common target column names in thyroid cancer datasets
    possible_targets = ['Outcome', 'Recurred', 'Class', 'Target', 'Label', 'Result', 'Recurrence']
    
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    # If none found, use the last column (common convention)
    if target_col is None:
        target_col = df.columns[-1]
        st.warning(f"Target column not found. Using last column: '{target_col}'")
    else:
        st.info(f"Using target column: '{target_col}'")
    
    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Check if we have enough data
        if len(df) < 10:
            st.error("Dataset too small for training")
            return None, 0, {}, []
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Handle potential errors in classification report
        try:
            report = classification_report(y_test, y_pred, output_dict=True)
        except Exception as e:
            st.warning(f"Classification report error: {e}")
            report = {"accuracy": acc}

        return model, acc, report, list(X.columns), target_col
    
    except Exception as e:
        st.error(f"Error in model training: {e}")
        return None, 0, {}, [], None

# === Streamlit App UI ===
st.set_page_config(page_title="Thyroid Cancer Outcome Predictor", layout="centered")
st.title("🧠 Thyroid Cancer Outcome Predictor")
st.markdown("This app predicts the **outcome** of differentiated thyroid cancer based on patient information.")

# Load dataset and model
df, label_encoders = load_data()

if df is not None:
    result = train_model(df)
    if len(result) == 5:  # Successful training
        model, acc, report, feature_names, target_col = result
        
        st.subheader("📁 Dataset Sample")
        st.dataframe(df.head())

        st.subheader("📊 Model Performance")
        st.success(f"Model Accuracy: {acc * 100:.2f}%")

        st.subheader("📝 Patient Data Entry Form")

        # === Form for user input ===
        with st.form("prediction_form"):
            input_data = {}
            for feature in feature_names:
                if feature in label_encoders:
                    options = list(label_encoders[feature].classes_)
                    value = st.selectbox(f"{feature}", options, key=f"select_{feature}")
                    encoded = label_encoders[feature].transform([value])[0]
                    input_data[feature] = encoded
                else:
                    input_data[feature] = st.number_input(f"{feature}", step=0.1, key=f"num_{feature}")

            submitted = st.form_submit_button("Predict Outcome")

            if submitted:
                try:
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]
                    
                    # Determine label based on target column values
                    unique_values = df[target_col].unique()
                    if len(unique_values) == 2:
                        if set(unique_values) == {0, 1}:
                            label = "Positive" if prediction == 1 else "Negative"
                        else:
                            label = str(prediction)
                    else:
                        label = str(prediction)
                    
                    st.subheader("🔍 Prediction Result")
                    st.info(f"The predicted outcome is: **{label}**")
                    st.write(f"Confidence: {max(probability):.2%}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        st.subheader("📄 Classification Report")
        if report:
            st.json(report)
    else:
        st.error("Model training failed. Please check your dataset.")
else:
    st.error("Failed to load dataset. Please check the file path and format.")

# === Additional debugging section ===
st.subheader("🔧 Debug Information")
if df is not None:
    st.write("**Dataset shape:**", df.shape)
    st.write("**Data types:**")
    st.write(df.dtypes)
    st.write("**Missing values:**")
    st.write(df.isnull().sum())
else:
    st.write("No dataset loaded for debugging.")