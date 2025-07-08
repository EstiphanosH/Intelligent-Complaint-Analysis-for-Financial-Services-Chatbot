import pytest
import pandas as pd
import numpy as np
from src.data_processing import filter_products, clean_narrative

def test_filter_products():
    test_data = pd.DataFrame({
        'Product': ['Credit card', 'Mortgage', 'Personal loan', 'Debt collection', 
                   'BNPL', 'Savings account', 'Money transfers', 'Credit reporting'],
        'Complaint ID': range(8)
    })
    
    filtered = filter_products(test_data)
    expected_products = ['Credit Card', 'Personal Loan', 'BNPL', 'Savings Account', 'Money Transfer']
    assert set(filtered['Product'].unique()) == set(expected_products)
    assert len(filtered) == 5

def test_clean_narrative():
    # Test basic cleaning
    text = "I am writing to FILE a complaint! About my ACCOUNT @XYZ Bank. #frustrated"
    cleaned = clean_narrative(text)
    assert cleaned == "about my account xyz bank frustrated"
    
    # Test boilerplate removal
    text = "Dear Consumer Financial Protection Bureau, I am writing to file a complaint regarding..."
    cleaned = clean_narrative(text)
    assert "dear" not in cleaned
    assert "writing to file a complaint" not in cleaned
    
    # Test empty handling
    assert clean_narrative("") == ""
    assert clean_narrative(None) == ""
    assert clean_narrative("   ") == ""
    
    # Test special characters
    text = "Account# 12345 - Balance $100.50 was charged incorrectly!"
    cleaned = clean_narrative(text)
    assert "#" not in cleaned
    assert "$" not in cleaned
    assert "." not in cleaned
    assert cleaned == "account 12345 balance 100.50 was charged incorrectly"

def test_preprocess_data():
    # This would require mocking file loading in actual implementation
    pass