from .rag_logic import RAGSystem
import pandas as pd
from .utils import configure_logging
import os

logger = configure_logging()

# Updated evaluation questions
EVAL_QUESTIONS = [
    ("What are common complaints about BNPL late fees?", "BNPL"),
    ("Why are customers unhappy with money transfers?", "Money Transfer"),
    ("What credit card billing issues do customers report?", "Credit Card"),
    ("How are savings account interest rates causing problems?", "Savings Account"),
    ("What personal loan application issues occur?", "Personal Loan")
]

def evaluate_rag_system(rag: RAGSystem) -> pd.DataFrame:
    """Evaluate RAG system with predefined questions"""
    results = []
    
    for question, category in EVAL_QUESTIONS:
        logger.info(f"Evaluating: {question}")
        try:
            response = rag.generate_response(question)
            answer = response['answer']
            sources = response['sources']['documents'][0][:2] if response['sources']['documents'] else []
            
            # Simple relevance scoring
            score = 3  # Neutral score
            if category.lower() in answer.lower():
                score = 4
            if "don't know" in answer.lower() or "not enough" in answer.lower():
                score = 2
                
            results.append({
                "Question": question,
                "Generated Answer": answer,
                "Retrieved Sources": [src[:100] + "..." for src in sources],
                "Quality Score": score,
                "Analysis": "Relevant" if score > 3 else "Needs improvement"
            })
        except Exception as e:
            logger.error(f"Evaluation failed for: {question} - {str(e)}")
            results.append({
                "Question": question,
                "Generated Answer": "EVALUATION ERROR",
                "Retrieved Sources": [],
                "Quality Score": 1,
                "Analysis": "Evaluation failed"
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        logger.info("Starting RAG evaluation...")
        rag = RAGSystem()
        eval_results = evaluate_rag_system(rag)
        
        # Save results
        output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "evaluation_report.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        eval_results.to_csv(output_path, index=False)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Evaluation process failed: {str(e)}")