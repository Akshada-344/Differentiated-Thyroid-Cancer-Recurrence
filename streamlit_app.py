import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Set Streamlit config
st.set_page_config(page_title="Thyroid Cancer Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("differentiated+thyroid+cancer+recurrence/Thyroid_Diff.csv")
    df.dropna(inplace=True)
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

df, encoders = load_data()

# Train model
def train_model(df):
    y = df['Recurred'] if 'Recurred' in df.columns else df.iloc[:, -1]
    X = df.drop(columns=['Recurred']) if 'Recurred' in df.columns else df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, list(X.columns), y_test, y_pred

model, acc, report, features, y_test, y_pred = train_model(df)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Dataset Summary", "Graphs", "Prediction", "Prediction Analysis"])

# Home
if page == "Home":
    st.title("🔬 Predict Thyroid Cancer Recurrence with AI")
    st.markdown(
        """
        Welcome to the **Thyroid Cancer Recurrence Predictor**! 👋

        This application leverages machine learning to help predict the likelihood
        of recurrence in patients diagnosed with Differentiated Thyroid Cancer.
        Our goal is to provide valuable insights for informed decision-making.

        ---

        Navigate through the sections using the sidebar on the left:

        * **Dataset Summary:** Get an overview of the data used for training.
        * **Graphs:** Explore various visualizations to understand feature relationships.
        * **Prediction:** Input patient data to receive a recurrence prediction.
        * **Prediction Analysis:** Review the model's performance metrics and insights.

        Let's get started!
        """
    )


# Dataset Summary
elif page == "Dataset Summary":
    st.header("📊 Dataset Summary")
    st.write(df.head())
    st.write("Shape of dataset:", df.shape)
    st.write("Summary statistics:")
    st.write(df.describe())

# Graphs
elif page == "Graphs":
    st.header("📈 Graphical Analysis")

    # Pairplot
    st.subheader("Pair Plot")
    pairplot_features = ['Age', 'Gender', 'Smoking', 'Risk', 'Stage', 'Recurred']
    df_encoded = df.copy()
    for col in ['Gender', 'Smoking', 'Risk', 'Stage', 'Recurred']:
        # Ensure 'Recurred' is encoded if it's an object type, as it's the hue.
        # This part was already in your original code, but explicitly noting for clarity.
        if df_encoded[col].dtype == 'object':
             df_encoded[col] = pd.factorize(df_encoded[col])[0]
    sns.pairplot(df_encoded[pairplot_features], hue='Recurred', palette='Set2')
    st.pyplot(plt.gcf())
    plt.clf()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    df_corr = df.copy()
    for col in df_corr.columns:
        if df_corr[col].dtype == 'object':
            df_corr[col] = pd.factorize(df_corr[col])[0]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6)) # Create a figure and an axes object
    disp.plot(cmap=plt.cm.Blues, ax=ax) # Pass the axes object to disp.plot
    ax.set_title("Confusion Matrix")
    st.pyplot(fig) # Pass the figure object to st.pyplot
    plt.clf()


# Prediction
elif page == "Prediction":
    st.header("🔮 Predict Thyroid Cancer Recurrence")
    with st.form("predict_form"):
        user_input = {}
        cols = st.columns(3)
        for i, feature in enumerate(features):
            col = cols[i % 3]
            if feature in encoders:
                options = encoders[feature].classes_
                val = col.selectbox(f"{feature}", options)
                user_input[feature] = encoders[feature].transform([val])[0]
            else:
                user_input[feature] = col.number_input(f"{feature}", value=0.0)

        submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([user_input])
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            st.success(f"Prediction: {'Recurrence Likely' if pred else 'No Recurrence'}")
            st.info(f"Confidence: {max(prob) * 100:.2f}%")

# Prediction Analysis
elif page == "Prediction Analysis":
    st.header("📈 Prediction Analysis")
    st.subheader("Model Performance Metrics")

    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision (Recurred):** {report['1']['precision']:.4f}")
    st.write(f"**Recall (Recurred):** {report['1']['recall']:.4f}")
    st.write(f"**F1-Score (Recurred):** {report['1']['f1-score']:.4f}")

    st.subheader("Performance Metrics Bar Chart")
    # Prepare data for plotting
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            report['0']['precision'], report['0']['recall'], report['0']['f1-score'],
            report['1']['precision'], report['1']['recall'], report['1']['f1-score']
        ],
        'Class': ['No Recurrence', 'No Recurrence', 'No Recurrence', 'Recurred', 'Recurred', 'Recurred']
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Class', data=metrics_df, palette='viridis', ax=ax)
    ax.set_title("Precision, Recall, and F1-Score by Class")
    ax.set_ylim(0, 1) # Metrics are between 0 and 1
    st.pyplot(fig)
    plt.clf()

    st.subheader("Classification Report")
    # Convert the dictionary report to a DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
