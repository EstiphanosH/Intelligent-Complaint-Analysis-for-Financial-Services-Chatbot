import os
import logging
import sys
from typing import List, Generator  # ✅ Add this line

# Add this new function
def get_project_root():
    """Get absolute path to project root"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add this new function
def get_data_path(*subpaths):
    """Get absolute path to data directory"""
    root = get_project_root()
    return os.path.join(root, "data", *subpaths)

# Update configure_logging to use project root
def configure_logging(log_file: str = "complaint_analysis.log") -> logging.Logger:
    """Configure application logging"""
    log_path = os.path.join(get_project_root(), "logs", log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def batch_generator(data: List, batch_size: int) -> Generator:
    """Generate batches from list data"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
