import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("Bank Marketing Classification App")

# Load scaler and encoders
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

# Model selection
model_option = st.selectbox(
    "Select Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model = pickle.load(open(f"model/{model_option}.pkl", "rb"))

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)


    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("Dataset must contain target column 'y'")
    else:
        X = data.drop("y", axis=1)
        y = data["y"]

        if "y" in encoders:
            y = encoders["y"].transform(y)

        # Apply same encoding
        for col in X.select_dtypes(include='object').columns:
            if col in encoders:
                X[col] = encoders[col].transform(X[col])

        # Scaling
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        mcc = matthews_corrcoef(y, predictions)
        auc = roc_auc_score(y, y_prob)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", round(accuracy,4))
        st.write("AUC:", round(auc,4))
        st.write("Precision:", round(precision,4))
        st.write("Recall:", round(recall,4))
        st.write("F1 Score:", round(f1,4))
        st.write("MCC:", round(mcc,4))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, predictions))

        st.subheader("Classification Report")
        st.text(classification_report(y, predictions))
