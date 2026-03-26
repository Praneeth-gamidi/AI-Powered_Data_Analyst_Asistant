import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import services

app = FastAPI(
    title="AI Data Analyst Assistant",
    description="Upload a CSV and explore it with data profiling, cleaning, EDA, NLP queries, and ML prediction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset = None
clean_dataset = None
faiss_index = None
text_rows = None

class CleanRequest(BaseModel):
    method: str

class QueryRequest(BaseModel):
    query: str

class PredictRequest(BaseModel):
    target_column: str

@app.get("/")
def root():
    return {"message": "AI Data Analyst Assistant API is running!"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global dataset

    file_content = await file.read()
    try:
        dataset = pd.read_csv(io.BytesIO(file_content))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(error)}")

    preview_df = dataset.head(5).fillna("")
    return {
        "message": f"File '{file.filename}' uploaded successfully.",
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "preview": preview_df.to_dict(orient="records"),
    }

@app.get("/profile")
def get_profile():
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please use /upload first.")
    return services.profile_data(dataset)

@app.post("/clean")
def clean_dataset_endpoint(request: CleanRequest):
    global clean_dataset

    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please use /upload first.")

    clean_dataset = services.clean_data(dataset, request.method)
    preview_df = clean_dataset.head(5).fillna("")
    
    return {
        "message": f"Data cleaned using method: '{request.method}'",
        "original_rows": dataset.shape[0],
        "cleaned_rows": clean_dataset.shape[0],
        "rows_removed": dataset.shape[0] - clean_dataset.shape[0],
        "preview": preview_df.to_dict(orient="records"),
    }

@app.get("/eda")
def run_eda():
    active_dataset = clean_dataset if clean_dataset is not None else dataset

    if active_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please use /upload first.")

    histograms = services.generate_histograms(active_dataset)
    correlation_image = services.generate_correlation_matrix(active_dataset)

    return {
        "histograms": histograms,
        "correlation_matrix": correlation_image,
    }

@app.post("/query")
def query_dataset(request: QueryRequest):
    global faiss_index, text_rows

    active_dataset = clean_dataset if clean_dataset is not None else dataset

    if active_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please use /upload first.")

    if faiss_index is None or text_rows is None:
        text_rows = services.convert_rows_to_text(active_dataset)
        faiss_index, _ = services.build_faiss_index(text_rows)

    relevant_rows = services.retrieve_top_rows(request.query, faiss_index, text_rows, top_k=5)
    answer = services.answer_query(request.query, relevant_rows)

    return {
        "query": request.query,
        "answer": answer,
        "relevant_rows": relevant_rows,
    }

@app.post("/predict")
def predict(request: PredictRequest):
    active_dataset = clean_dataset if clean_dataset is not None else dataset

    if active_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please use /upload first.")

    result = services.train_linear_regression(active_dataset, request.target_column)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
