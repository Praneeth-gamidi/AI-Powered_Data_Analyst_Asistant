# 🤖 AI-Powered Data Analyst Assistant

A full-stack AI application that lets you upload a CSV dataset and:
- 📊 **Profile** your data (types, missing values, statistics)
- 🧹 **Clean** missing values (drop, mean, or median fill)
- 📈 **Explore** with EDA charts (histograms + correlation matrix)
- 💬 **Ask questions** in natural language using RAG (sentence-transformers + FAISS)
- 🔮 **Predict** with Linear Regression (scikit-learn)

---

## Project Structure

```
AI-Powered_Data_Analyst_Assistant/
├── backend/
│   ├── main.py        # FastAPI app and all endpoints
│   └── services.py    # All business logic
├── frontend/
│   └── app.py         # Streamlit UI
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Setup & Run

### Step 1: Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install may take a few minutes as it downloads the sentence-transformers model (~90MB).

---

### Step 3: Start the FastAPI Backend

Open a terminal and run:

```bash
uvicorn backend.main:app --reload
```

The API will be available at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

---

### Step 4: Start the Streamlit Frontend

Open a **second terminal** and run:

```bash
streamlit run frontend/app.py
```

The app will open in your browser at: `http://localhost:8501`

---

## How to Use

1. **📂 Upload Data** — Upload any CSV file
2. **📊 Data Profile** — View column types, missing values, and statistics
3. **🧹 Clean Data** — Handle missing values (drop rows, or fill with mean/median)
4. **📈 EDA Charts** — View histograms and the correlation matrix
5. **💬 Ask a Question** — Type a natural language question about your data
6. **🔮 Predict** — Select a target column and train a Linear Regression model

---

## Sample CSV Datasets to Try

- [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
- [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- [Tips Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Semantic Search | sentence-transformers, FAISS |
| Visualization | matplotlib |

---

## Notes

- The dataset is stored **in-memory** on the backend (simple, beginner-friendly approach)
- The RAG system uses **`all-MiniLM-L6-v2`** — a lightweight but powerful embedding model
- No external LLM API key required — the query system uses semantic search + formatted output
- Clean data → then run EDA/Query/Predict for best results
