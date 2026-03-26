import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sentence_transformers import SentenceTransformer
import faiss

def profile_data(dataframe: pd.DataFrame) -> dict:
    return {
        "shape": {"rows": dataframe.shape[0], "columns": dataframe.shape[1]},
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isnull().sum().to_dict(),
        "statistics": dataframe.describe(include="all").fillna("").to_dict(),
    }

def clean_data(dataframe: pd.DataFrame, method: str) -> pd.DataFrame:
    cleaned_df = dataframe.copy()

    if method == "drop":
        cleaned_df = cleaned_df.dropna()
    elif method == "mean":
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
    elif method == "median":
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
    else:
        raise ValueError(f"Unknown cleaning method: '{method}'")

    return cleaned_df

def plot_to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(figure)
    return encoded_image

def generate_histograms(dataframe: pd.DataFrame) -> list:
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

    for column_name in numeric_columns:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(dataframe[column_name].dropna(), bins=20, color="#6C63FF", edgecolor="white")
        axis.set_title(f"Distribution of {column_name}", fontsize=13)
        axis.set_xlabel(column_name)
        axis.set_ylabel("Frequency")
        axis.grid(axis="y", alpha=0.3)

        histograms.append({
            "column": column_name,
            "image": plot_to_base64(figure)
        })

    return histograms

def generate_correlation_matrix(dataframe: pd.DataFrame) -> str:
    numeric_df = dataframe.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr()
    figure, axis = plt.subplots(figsize=(8, 6))
    heatmap = axis.imshow(correlation, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    figure.colorbar(heatmap, ax=axis)

    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(correlation.columns, fontsize=9)
    axis.set_title("Correlation Matrix", fontsize=14)

    for row_idx in range(len(correlation.columns)):
        for col_idx in range(len(correlation.columns)):
            value = correlation.iloc[row_idx, col_idx]
            axis.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)

    return plot_to_base64(figure)

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def convert_rows_to_text(dataframe: pd.DataFrame) -> list:
    text_rows = []
    for _, row in dataframe.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        text_rows.append(row_text)
    return text_rows

def build_faiss_index(text_rows: list):
    model = get_embedding_model()
    embeddings = model.encode(text_rows, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    vector_dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(vector_dimension)
    faiss_index.add(embeddings)

    return faiss_index, embeddings

def retrieve_top_rows(query: str, faiss_index, text_rows: list, top_k: int = 5) -> list:
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    _, indices = faiss_index.search(query_embedding, top_k)
    return [text_rows[idx] for idx in indices[0] if idx < len(text_rows)]

def answer_query(query: str, relevant_rows: list) -> str:
    context = "\n".join([f"  • {row}" for row in relevant_rows])
    return (
        f"📌 Query: {query}\n\n"
        f"🔍 Most Relevant Data Rows Found:\n{context}\n\n"
        f"💡 These rows are the most semantically similar to your question. "
        f"Review them to find your answer."
    )

def train_linear_regression(dataframe: pd.DataFrame, target_column: str) -> dict:
    numeric_df = dataframe.select_dtypes(include=[np.number]).dropna()

    if target_column not in numeric_df.columns:
        return {"error": f"Target column '{target_column}' must be numeric for Linear Regression."}
    if numeric_df.shape[0] < 10:
        return {"error": "Not enough rows (need at least 10) to train the model."}

    feature_columns = [col for col in numeric_df.columns if col != target_column]

    if len(feature_columns) == 0:
        return {"error": "No feature columns available. Need at least one numeric column besides the target."}

    X = numeric_df[feature_columns].values
    y = numeric_df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predictions)
    r2 = r2_score(y_test, y_predictions)

    sample_predictions = [
        {"actual": round(float(actual), 4), "predicted": round(float(predicted), 4)}
        for actual, predicted in zip(y_test[:10], y_predictions[:10])
    ]

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "mean_squared_error": round(mse, 4),
        "r2_score": round(r2, 4),
        "sample_predictions": sample_predictions,
    }
