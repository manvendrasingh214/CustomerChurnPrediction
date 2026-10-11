
import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "notebooks", "customer_churn_model.pkl"
)
SCHEMA_PATH = os.path.join(
    BASE_DIR, "notebooks", "customer_churn_schema.pkl"
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Customer ki details bharo aur predict karo ki customer "
    "service chhod sakta hai ya nahi."
)

@st.cache_resource
def load_artifacts():
    model_data = joblib.load(MODEL_PATH)
    schema_data = joblib.load(SCHEMA_PATH)
    return model_data, schema_data

try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file nahi mili: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(SCHEMA_PATH):
        st.error(f"Schema file nahi mili: {SCHEMA_PATH}")
        st.stop()

    model_data, schema = load_artifacts()
    model = model_data["model"]
    threshold = float(model_data["threshold"])

except Exception as e:
    st.error(f"Model load karne mein problem: {e}")
    st.stop()

st.subheader("Customer Details")

with st.form("customer_form"):
    customer_data = {}
    columns = st.columns(2)

    for i, (feature, info) in enumerate(schema.items()):
        with columns[i % 2]:
            if info["type"] == "numeric":
                minimum = float(info["min"])
                maximum = float(info["max"])
                default = float(info["default"])

                if minimum == maximum:
                    customer_data[feature] = minimum
                else:
                    customer_data[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=1.0 if info.get("integer", False) else 0.1
                    )
            else:
                options = info.get("options", [])
                default = info.get("default")

                if not options:
                    customer_data[feature] = st.text_input(feature)
                else:
                    index = options.index(default) if default in options else 0
                    customer_data[feature] = st.selectbox(
                        feature,
                        options,
                        index=index
                    )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary"
    )

if submitted:
    customer_df = pd.DataFrame(
        [customer_data],
        columns=list(schema.keys())
    )

    try:
        probability = float(model.predict_proba(customer_df)[0, 1])
        prediction = int(probability >= threshold)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{probability:.2%}")
        col2.metric("Decision Threshold", f"{threshold:.2f}")

        if prediction == 1:
            st.error("⚠️ Customer may churn (leave the service).")
        else:
            st.success("✅ Customer is predicted not to churn.")

        st.caption(
            "Prediction is based on the trained machine-learning model. "
            "It is an estimate, not a guarantee."
        )

    except Exception as e:
        st.error(f"Prediction mein problem: {e}")
