import sys
import os

# Add the project root (parent of src) to sys.path
sys.path.insert(0, os.path.abspath('..'))
from src.utils import configure_logging
logger = configure_logging()

import pandas as pd
import re
from tqdm import tqdm

PRODUCT_MAP = {
    'Credit card': 'Credit Card',
    'Credit Card': 'Credit Card',
    'Personal loan': 'Personal Loan',
    'Personal Loan': 'Personal Loan',
    'Payday loan': 'Personal Loan',
    'Vehicle loan': 'Personal Loan',
    'Buy Now Pay Later': 'BNPL',
    'BNPL': 'BNPL',
    'Savings account': 'Savings Account',
    'Savings Account': 'Savings Account',
    'Money transfers': 'Money Transfer',
    'Money Transfer': 'Money Transfer',
}

def load_data(file_path: str) -> pd.DataFrame:
    """Load raw complaint data from CSV"""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def filter_products(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to target products"""
    logger.info("Filtering target products")
    valid_products = list(PRODUCT_MAP.values())
    df['Product'] = df['Product'].map(PRODUCT_MAP).fillna('Other')
    return df[df['Product'].isin(valid_products)]

def clean_narrative(text: str) -> str:
    """Clean complaint narrative text"""
    if not isinstance(text, str):
        return ""
    
    # Remove common boilerplate phrases
    boilerplates = [
        r"i am writing to file a complaint",
        r"dear (sir|madam|consumer financial protection bureau)",
        r"regarding my account [x\d]+",
        r"please see below for details"
    ]
    
    for pattern in boilerplates:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove special characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,;:!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main preprocessing pipeline"""
    logger.info("Starting preprocessing")
    
    # Filter products
    df = filter_products(df)
    
    # Handle missing narratives
    initial_count = len(df)
    df = df[df['Consumer complaint narrative'].notna()]
    df = df[df['Consumer complaint narrative'].str.strip() != ""]
    logger.info(f"Removed {initial_count - len(df)} records with empty narratives")
    
    # Clean text
    tqdm.pandas(desc="Cleaning narratives")
    df['clean_narrative'] = df['Consumer complaint narrative'].progress_apply(clean_narrative)
    
    # Add word count metadata
    df['word_count'] = df['clean_narrative'].apply(lambda x: len(x.split()))
    
    logger.info(f"Final processed records: {len(df)}")
    return df[['Complaint ID', 'Product', 'clean_narrative', 'word_count']]

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to CSV"""
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    df = load_data("../data/raw/complaints.csv")
    processed_df = preprocess_data(df)
    save_processed_data(processed_df, "../data/processed/filtered_complaints.csv")