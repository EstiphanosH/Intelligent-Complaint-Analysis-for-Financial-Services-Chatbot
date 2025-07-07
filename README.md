
# 🧠 Intelligent Complaint Analysis System for Financial Services

## Transforming Customer Feedback into Strategic Insights

This repository contains a **RAG-powered chatbot solution** developed for **CrediTrust Financial** to analyze customer complaints and generate actionable insights. The system empowers internal teams—such as product managers, compliance officers, and customer support—to understand real pain points across financial products in real-time.

---

## 🎯 Business Objective

CrediTrust, a mobile-first digital finance provider in East Africa, receives thousands of customer complaints each month across five product lines:

- Credit Cards  
- Personal Loans  
- Buy Now, Pay Later (BNPL)  
- Savings Accounts  
- Money Transfers  

This solution addresses the following key goals:

- ⏱ **Reduce complaint trend identification time** from days to minutes  
- 🙋‍♀️ **Enable non-technical teams** to self-serve insights  
- 📈 **Proactively detect and resolve** systemic issues  

---

## 🚀 Key Features

- 🔎 **Semantic Search Engine** using ChromaDB  
- 🧠 **LLM-Powered Insight Generation** with Mistral-7B  
- 🗂 **Multi-Product Analysis** with category-level filtering  
- 🧾 **Source Attribution** with complaint ID metadata  
- 📡 **Real-Time Feedback Loop** for proactive monitoring  
- ✅ **Regulatory Traceability** via audit-ready outputs  

---

## 🧠 Technical Stack

| Component        | Technology         | Purpose                          |
|------------------|--------------------|----------------------------------|
| Vector Store     | ChromaDB           | Efficient semantic retrieval     |
| Embeddings       | all-MiniLM-L6-v2   | Text representation              |
| LLM              | Mistral-7B         | Contextual answer generation     |
| Text Processing  | spaCy, LangChain   | Cleaning and chunking            |
| UI Framework     | Gradio             | Interactive user interface       |
| Data Processing  | Pandas, NumPy      | EDA and transformations          |
| Visualization    | Matplotlib, Seaborn| Complaint insights               |
| Environment      | Poetry             | Dependency management            |

---

## 🏗 Solution Architecture

1. **Data Ingestion**: Load and clean CFPB complaint dataset  
2. **Preprocessing & EDA**: Filter, tokenize, and normalize narratives  
3. **Chunking & Embedding**: Slice long texts and embed with MiniLM  
4. **Vector Indexing**: Store embeddings in ChromaDB  
5. **Retrieval & Generation**: Use RAG to synthesize user queries  
6. **Interactive Interface**: Gradio app for internal users  

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites

- Python 3.10+
- Poetry (recommended) or pip

### 📦 Setup with Poetry

```bash
git clone https://github.com/creditrust/ai-complaint-analysis.git
cd ai-complaint-analysis
poetry install
poetry shell
```

### 📦 Alternative: pip

```bash
pip install -r requirements.txt
```

### 🚦 Execution Pipeline


Step 1: Preprocess and clean complaint data
```bash
python src/data_preprocessing.py
```

Step 2: Create vector index from complaint embeddings

```bash
python src/vector_store_creation.py
```
Step 3: Run the chatbot application
```bash
python src/app.py
```

Access the app locally at: [http://localhost:7860](http://localhost:7860)

---

## 📂 Project Structure

```
Intelligent-Complaint-Analysis-for-Financial-Services-Chatbot/
├── data/
│   ├── raw/                       # Original datasets
│   └── processed/                 # Filtered and cleaned complaints
├── notebooks/
│   └── eda_preprocessing.ipynb    # EDA and visual insights
├── src/
│   ├── data_preprocessing.py      # Text cleaning logic
│   ├── vector_store_creation.py   # Chunking and indexing
│   ├── rag_pipeline.py            # Core RAG system
│   └── app.py                     # Gradio interface
├── vector_store/                  # Persistent vector DB
├── reports/
│   ├── interim_report.md          # Task 1–2 findings
│   └── final_report.md            # Full documentation
├── tests/
│   ├── test_preprocessing.py
│   └── test_retrieval.py
├── pyproject.toml
└── README.md
```

---

## 💻 Usage Example

```python
from src.rag_pipeline import ComplaintAnalyzer

analyzer = ComplaintAnalyzer()
response = analyzer.ask("Why are customers unhappy with BNPL services?")

print("Insight:", response.answer)
print("Sources:", response.sources)  # List of complaint IDs
```

---

## 🧪 Sample Queries

- "Top 3 issues with credit cards in Kenya"
- "Compare complaint trends for BNPL vs personal loans"
- "Recent fraud patterns in money transfers"
- "Customer pain points for savings accounts"

---

## 📈 Expected Business Impact

| Metric                  | Current     | Target    | Improvement        |
|-------------------------|-------------|-----------|--------------------|
| Trend Identification    | 2–3 days    | < 2 mins  | 2000× faster       |
| Analyst Dependence      | 100%        | 15%       | 85% reduction      |
| Proactive Fix Rate      | 22%         | 68%       | 3× improvement     |
| Compliance Risk Level   | High        | Medium    | 40% reduction      |

---

## 👥 Team

- **Project Lead**:  
- **Product Owner**:  
- **Engineering Contact**: 

---
## 📜 License

This project is licensed under the **MIT License**.

> “This solution transforms complaint analysis from a reactive chore to a strategic advantage, helping us build better financial products for East Africa.”  
> — *Asha M., BNPL Product Lead*
