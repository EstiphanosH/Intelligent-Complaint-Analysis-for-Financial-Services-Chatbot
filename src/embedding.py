import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from tqdm import tqdm

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("embedding_errors.log")
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_project_root():
    """Get absolute path to project root"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def get_vector_dir():
    """Get absolute path to vector store directory"""
    vector_dir = os.path.join(get_project_root(), "vectorstore")
    os.makedirs(vector_dir, exist_ok=True)
    return vector_dir

def get_chunks_path():
    """Get absolute path to chunks file"""
    return os.path.join(get_project_root(), "data", "processed", "chunks.csv")

def initialize_vector_db() -> chromadb.Collection:
    """Initialize ChromaDB vector database"""
    vector_dir = get_vector_dir()
    try:
        client = chromadb.PersistentClient(
            path=vector_dir,
            settings=Settings(allow_reset=True)
        )

        return client.get_or_create_collection(
            name="creditrust_complaints",
            metadata={"hnsw:space": "cosine"}
)
    except Exception as e:
        logger.error(f"Failed to initialize vector DB: {str(e)}")
        raise

def batch_generator(data: list, batch_size: int):
    """Generate batches from list data (replaces utils import)"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def load_chunk_data() -> pd.DataFrame:
    """Load chunk data with validation and error handling"""
    chunks_path = get_chunks_path()
    
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found at {chunks_path}")
        logger.info("Please run the chunking process first: python -m src.chunking")
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
    
    try:
        logger.info(f"Loading chunks from {chunks_path}")
        return pd.read_csv(chunks_path)
    except Exception as e:
        logger.error(f"Failed to load chunks: {str(e)}")
        raise

def generate_embeddings():
    """Generate embeddings and persist to vector DB"""
    try:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        collection = initialize_vector_db()
        
        # Load data
        chunk_df = load_chunk_data()
        
        # Check required columns
        required_columns = ['text', 'complaint_id', 'product', 'chunk_id']
        missing = [col for col in required_columns if col not in chunk_df.columns]
        if missing:
            raise ValueError(f"Missing columns in chunks data: {', '.join(missing)}")
        
        # Delete and recreate the collection to clear all data
        client = collection._client  # get the client from the collection
        client.delete_collection(collection.name)
        collection = client.get_or_create_collection(
            name="creditrust_complaints",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Existing vector data cleared")
        
        logger.info("Generating embeddings...")
        batch_size = 256
        texts = chunk_df['text'].tolist()
        metadatas = chunk_df[['complaint_id', 'product', 'chunk_id']].to_dict('records')
        ids = chunk_df['chunk_id'].tolist()
        
        total_chunks = len(texts)
        logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
        
        with tqdm(total=total_chunks, desc="Embedding chunks") as pbar:
            for i, (batch_texts, batch_metadatas, batch_ids) in enumerate(
                zip(
                    batch_generator(texts, batch_size),
                    batch_generator(metadatas, batch_size),
                    batch_generator(ids, batch_size)
                )
            ):
                batch_embeddings = model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                ).tolist()
                
                collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                pbar.update(len(batch_ids))
                logger.debug(f"Processed batch {i+1}: {len(batch_ids)} embeddings")
        
        logger.info(f"✅ Persisted {total_chunks} embeddings to vector store")
        return True
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting embedding process...")
    
    if not os.path.exists(get_chunks_path()):
        logger.error("Chunks file not found. Please run chunking process first.")
        logger.info("Execute: python -m src.chunking")
        sys.exit(1)
    
    success = generate_embeddings()
    if success:
        logger.info("Embedding process completed successfully!")
    else:
        logger.error("Embedding process failed")
        sys.exit(1)