
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
from pathlib import Path

st.set_page_config(
    page_title="NetGuard ML",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_OPTIONS = {
    "Original Random Forest (Multiclass)":
        MODEL_DIR / "netguard_multiclass_model.joblib",
    "NetGuard ML v3":
        MODEL_DIR / "netguard_multiclass_model_v3.joblib",
    "NetGuard ML v4":
        MODEL_DIR / "netguard_multiclass_model_v4.joblib",
}

ISOLATION_MODEL_PATH = (
    MODEL_DIR / "netguard_isolation_forest_experimental.joblib"
)
ISOLATION_CONFIG_PATH = (
    MODEL_DIR / "netguard_isolation_forest_config.json"
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("🛡️ NetGuard ML")
st.subheader("Machine Learning-Based Network Intrusion Detection System")

mode = st.radio(
    "Choose detection mode",
    [
        "Multiclass Detection",
        "Experimental Anomaly Detection"
    ],
    horizontal=True
)

st.write(
    "Upload a CSV containing network-flow features to analyze "
    "network traffic. This dashboard performs batch analysis; "
    "it does not capture live network traffic."
)

if mode == "Multiclass Detection":
    selected_model_name = st.selectbox(
        "Select intrusion detection model",
        list(MODEL_OPTIONS.keys())
    )
    model_path = MODEL_OPTIONS[selected_model_name]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}")
        st.stop()

    try:
        model = load_model(str(model_path))
        st.success(f"{selected_model_name} loaded successfully.")
        expected_features = list(
            model.named_steps["imputer"].feature_names_in_
        )
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        key="multiclass_upload"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            st.write("### Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Uploaded records: {len(df):,}")

            missing = [
                col for col in expected_features
                if col not in df.columns
            ]

            if missing:
                st.error(
                    f"Missing {len(missing)} required feature(s)."
                )
                st.write("Missing features:", missing[:30])
            else:
                X = df[expected_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                X = X.replace([np.inf, -np.inf], np.nan)

                if X.isna().all(axis=1).any():
                    st.warning(
                        "Some rows have no usable numeric feature values."
                    )

                if st.button(
                    "Analyze Network Traffic",
                    type="primary",
                    key="multiclass_analyze"
                ):
                    with st.spinner("Analyzing traffic..."):
                        predictions = model.predict(X)

                    results = df.copy()
                    results["Prediction"] = predictions
                    results["Status"] = np.where(
                        results["Prediction"] == "BENIGN",
                        "BENIGN",
                        "POTENTIAL ATTACK"
                    )

                    benign_count = int(
                        (predictions == "BENIGN").sum()
                    )
                    attack_count = len(predictions) - benign_count

                    st.write("### Detection Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", f"{len(results):,}")
                    c2.metric("Benign Records", f"{benign_count:,}")
                    c3.metric("Potential Attacks", f"{attack_count:,}")

                    st.write("### Predicted Class Distribution")
                    distribution = (
                        results["Prediction"]
                        .value_counts()
                        .rename_axis("Class")
                        .reset_index(name="Records")
                    )
                    st.dataframe(
                        distribution, use_container_width=True
                    )

                    st.write("### Detection Results")
                    selected_class = st.selectbox(
                        "Filter predictions",
                        ["All"] + sorted(
                            results["Prediction"].unique().tolist()
                        ),
                        key="multiclass_filter"
                    )

                    filtered = results
                    if selected_class != "All":
                        filtered = results[
                            results["Prediction"] == selected_class
                        ]

                    st.dataframe(
                        filtered, use_container_width=True
                    )
                    st.download_button(
                        "Download Results CSV",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="netguard_multiclass_results.csv",
                        mime="text/csv"
                    )

                    st.warning(
                        "Predictions are model-based indicators, not "
                        "proof of malicious activity. Performance on "
                        "unseen traffic may differ from development "
                        "results. Review alerts before taking action."
                    )

        except Exception as e:
            st.error(f"Unable to process the uploaded CSV: {e}")


else:
    st.info(
        "Experimental mode: Isolation Forest learns patterns from "
        "benign reference traffic and flags unusual records. "
        "An anomaly is not necessarily an attack."
    )

    if not ISOLATION_MODEL_PATH.exists():
        st.error(
            "Isolation Forest model not found: "
            f"{ISOLATION_MODEL_PATH}"
        )
        st.stop()

    if not ISOLATION_CONFIG_PATH.exists():
        st.error(
            "Isolation Forest configuration not found: "
            f"{ISOLATION_CONFIG_PATH}"
        )
        st.stop()

    try:
        anomaly_model = load_model(str(ISOLATION_MODEL_PATH))

        with open(
            ISOLATION_CONFIG_PATH, "r", encoding="utf-8"
        ) as f:
            config = json.load(f)

        expected_features = config.get("features", [])
        default_threshold = float(
            config.get("score_threshold", -0.02999)
        )

        st.success("Experimental Isolation Forest model loaded.")

    except Exception as e:
        st.error(f"Unable to load anomaly detector: {e}")
        st.stop()

    threshold = st.number_input(
        "Anomaly score threshold",
        value=default_threshold,
        format="%.5f",
        help=(
            "Uses the saved experimental threshold by default. "
            "Lower scores are more anomalous. Changing this value "
            "changes the alert sensitivity."
        )
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        key="anomaly_upload"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            st.write("### Uploaded Data Preview")
            st.dataframe(
    df.head(10),
    use_container_width=True,
    height=300
)
            st.caption(f"Uploaded records: {len(df):,}")

            missing = [
                col for col in expected_features
                if col not in df.columns
            ]

            if not expected_features:
                st.error(
                    "No feature list found in the configuration file."
                )
            elif missing:
                st.error(
                    f"Missing {len(missing)} required feature(s)."
                )
                st.write("Missing features:", missing[:30])
            else:
                X = df[expected_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.mask(X.abs() > 1e15)

                if X.isna().all(axis=1).any():
                    st.warning(
                        "Some rows have no usable numeric feature values. "
                        "Review these rows before relying on their scores."
                    )

                if st.button(
                    "Detect Anomalies",
                    type="primary",
                    key="anomaly_analyze"
                ):
                    with st.spinner("Scoring network traffic..."):
                        scores = anomaly_model.decision_function(X)
                        predictions = np.where(
                            scores < threshold,
                            "POTENTIAL ANOMALY",
                            "NORMAL-LIKE"
                        )

                    results = df.copy()
                    results["Anomaly Score"] = scores
                    results["Prediction"] = predictions
                    results["Status"] = predictions

                    anomaly_count = int(
                        (predictions == "POTENTIAL ANOMALY").sum()
                    )
                    normal_count = len(results) - anomaly_count

                    st.write("### Anomaly Detection Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", f"{len(results):,}")
                    c2.metric("Potential Anomalies", f"{anomaly_count:,}")
                    c3.metric("Normal-Like Records", f"{normal_count:,}")

                    st.write("### Prediction Distribution")
                    distribution = (
                        results["Prediction"]
                        .value_counts()
                        .rename_axis("Class")
                        .reset_index(name="Records")
                    )
                    st.dataframe(
                        distribution, use_container_width=True
                    )

                    st.write("### Detection Results")
                    selected_class = st.selectbox(
                        "Filter results",
                        ["All"] + sorted(
                            results["Prediction"].unique().tolist()
                        ),
                        key="anomaly_filter"
                    )

                    filtered = results
                    if selected_class != "All":
                        filtered = results[
                            results["Prediction"] == selected_class
                        ]

                    # Display only a limited number of rows
                    DISPLAY_LIMIT = 500

                    st.caption(
                        f"Showing up to {DISPLAY_LIMIT:,} rows "
                        f"out of {len(filtered):,} matching records."
                    )

                    st.dataframe(
                        filtered.head(DISPLAY_LIMIT),
                        use_container_width=True,
                        height=450
                    )

                    



                    st.download_button(
                        "Download Anomaly Results CSV",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="netguard_anomaly_results.csv",
                        mime="text/csv"
                    )

                    st.warning(
                        "This is an experimental anomaly detector. "
                        "Its threshold was calibrated on a small benign "
                        "sample, and unusual traffic is not necessarily "
                        "malicious. Validate it on independent, labelled "
                        "traffic before operational use."
                    )

        except Exception as e:
            st.error(f"Unable to analyze the uploaded CSV: {e}")
