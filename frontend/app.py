import base64
from io import BytesIO
import requests
import streamlit as st
import pandas as pd
import os
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #e94560;
    }
    .feature-card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e94560;
        margin-bottom: 16px;
    }
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #e94560;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4);
    }
    .stTextInput input, .stSelectbox select {
        background-color: #1a1a2e !important;
        border: 1px solid #e94560 !important;
        color: white !important;
        border-radius: 8px;
    }
    h1, h2, h3 { color: #e94560 !important; }
    .answer-box {
        background-color: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        white-space: pre-wrap;
        color: #a8dadc;
    }
    .success-badge {
        background-color: #1a472a;
        color: #52b788;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

def decode_base64_image(base64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def call_backend(method: str, endpoint: str, **kwargs):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method == "get":
            response = requests.get(url, timeout=60)
        else:
            response = requests.post(url, timeout=120, **kwargs)

        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_detail = f"HTTP {response.status_code}: {response.text}"
                
            st.error(f"❌ Error: {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as error:
        st.error(f"⚠️ Unexpected error: {str(error)}")
        return None

st.sidebar.markdown("## 🤖 AI Data Analyst")
st.sidebar.markdown("---")

selected_section = st.sidebar.radio(
    "Navigate to:",
    options=["📂 Upload Data", "📊 Data Profile", "🧹 Clean Data", "📈 EDA Charts", "💬 Ask a Question", "🔮 Predict"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by FastAPI + Streamlit + sentence-transformers + scikit-learn")

if selected_section == "📂 Upload Data":
    st.title("📂 Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.button("🚀 Upload & Load Dataset"):
            with st.spinner("Uploading and reading..."):
                result = call_backend(
                    "post",
                    "/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                )

            if result:
                st.success(f"✅ {result['message']}")
                col1, col2 = st.columns(2)
                col1.metric("📋 Total Rows", result["shape"]["rows"])
                col2.metric("📌 Total Columns", result["shape"]["columns"])

                st.subheader("👀 Preview (First 5 Rows)")
                preview_df = pd.DataFrame(result["preview"])
                st.table(preview_df)

                st.session_state["columns"] = list(preview_df.columns)
                st.session_state["uploaded"] = True

elif selected_section == "📊 Data Profile":
    st.title("📊 Data Profile")
    
    if st.button("🔍 Generate Profile"):
        with st.spinner("Profiling dataset..."):
            result = call_backend("get", "/profile")

        if result:
            col1, col2 = st.columns(2)
            col1.metric("📋 Rows", result["shape"]["rows"])
            col2.metric("📌 Columns", result["shape"]["columns"])

            st.subheader("📌 Column Data Types")
            dtypes_df = pd.DataFrame(list(result["dtypes"].items()), columns=["Column", "Data Type"])
            st.table(dtypes_df)

            st.subheader("❓ Missing Values")
            missing_df = pd.DataFrame(list(result["missing_values"].items()), columns=["Column", "Missing Count"])
            missing_df["Status"] = missing_df["Missing Count"].apply(lambda x: "⚠️ Has missing" if x > 0 else "✅ Complete")
            st.table(missing_df)

            st.subheader("📈 Basic Statistics")
            stats_df = pd.DataFrame(result["statistics"])
            st.dataframe(stats_df.astype(object), use_container_width=True)

elif selected_section == "🧹 Clean Data":
    st.title("🧹 Clean Your Data")
    
    cleaning_method = st.selectbox(
        "Select cleaning method:",
        options=["drop", "mean", "median"],
    )

    if st.button("🧹 Clean Dataset"):
        with st.spinner("Cleaning dataset..."):
            result = call_backend("post", "/clean", json={"method": cleaning_method})

        if result:
            st.success(f"✅ {result['message']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("📋 Original Rows", result["original_rows"])
            col2.metric("✅ Cleaned Rows", result["cleaned_rows"])
            col3.metric("🗑️ Rows Removed", result["rows_removed"])

            st.subheader("👀 Preview of Cleaned Data")
            preview_df = pd.DataFrame(result["preview"])
            st.table(preview_df)

elif selected_section == "📈 EDA Charts":
    st.title("📈 Exploratory Data Analysis")
    
    if st.button("📊 Generate Charts"):
        with st.spinner("Creating charts..."):
            result = call_backend("get", "/eda")

        if result:
            st.subheader("📊 Histograms (Numeric Columns)")
            histograms = result.get("histograms", [])

            if histograms:
                for i in range(0, len(histograms), 2):
                    cols = st.columns(2)
                    for j, histogram in enumerate(histograms[i:i+2]):
                        with cols[j]:
                            image = decode_base64_image(histogram["image"])
                            st.image(image, caption=f"Distribution: {histogram['column']}", use_container_width=True)
            else:
                st.warning("No numeric columns found for histograms.")

            st.subheader("🔗 Correlation Matrix")
            correlation_image_b64 = result.get("correlation_matrix")
            if correlation_image_b64:
                correlation_image = decode_base64_image(correlation_image_b64)
                st.image(correlation_image, caption="Correlation Matrix Heatmap", use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns to generate a correlation matrix.")

elif selected_section == "💬 Ask a Question":
    st.title("💬 Ask a Question")
    
    user_query = st.text_input("Your question:")

    if st.button("🔍 Search") and user_query:
        with st.spinner("Searching..."):
            result = call_backend("post", "/query", json={"query": user_query})

        if result:
            st.subheader("💡 Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

            st.subheader("🔍 Most Relevant Rows Found")
            for i, row_text in enumerate(result["relevant_rows"], 1):
                with st.expander(f"Row {i}"):
                    st.text(row_text)

elif selected_section == "🔮 Predict":
    st.title("🔮 ML Prediction")

    target_column = st.text_input("Enter target column name:")

    if st.button("🚀 Train & Predict") and target_column:
        with st.spinner("Training model..."):
            result = call_backend("post", "/predict", json={"target_column": target_column})

        if result:
            st.success("✅ Model trained successfully!")

            st.subheader("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Target Column", result["target_column"])
            col2.metric("📋 Training Rows", result["training_rows"])
            col3.metric("📐 MSE", result["mean_squared_error"])
            col4.metric("📈 R² Score", result["r2_score"])

            st.subheader("🔧 Features Used")
            features_df = pd.DataFrame({"Feature Columns": result["feature_columns"]})
            st.table(features_df)

            st.subheader("🔮 Sample Predictions (First 10)")
            predictions_df = pd.DataFrame(result["sample_predictions"])
            predictions_df.columns = ["Actual Value", "Predicted Value"]
            predictions_df["Difference"] = (predictions_df["Actual Value"] - predictions_df["Predicted Value"]).round(4)
            st.table(predictions_df)

            r2 = result["r2_score"]
            if r2 > 0.8:
                st.success(f"🎉 Great model! R² = {r2}")
            elif r2 > 0.5:
                st.warning(f"📊 Decent model. R² = {r2}")
            else:
                st.error(f"⚠️ Weak model. R² = {r2}")
