import re
import os
import pandas as pd
from src.utils import configure_logging, get_data_path

logger = configure_logging()

class TextSplitter:
    """Custom text splitter to avoid LangChain dependencies"""
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    
    def split_text(self, text: str) -> list:
        """Recursively split text into chunks preserving sentence boundaries"""
        # Base case: text is short enough
        if len(text.split()) <= self.chunk_size:
            return [text]
        
        # First try splitting by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 1:
            result = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk.split()) + len(para.split()) <= self.chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        result.append(current_chunk.strip())
                    current_chunk = para
                    
                    # Add remaining text recursively
                    if len(para.split()) > self.chunk_size:
                        result.extend(self.split_text(para))
            if current_chunk:
                result.append(current_chunk.strip())
            return result
        
        # Then try splitting by sentences
        sentences = re.split(self.sentence_endings, text)
        if len(sentences) > 1:
            result = []
            current_chunk = ""
            
            for sent in sentences:
                if len(current_chunk.split()) + len(sent.split()) <= self.chunk_size:
                    current_chunk += " " + sent if current_chunk else sent
                else:
                    if current_chunk:
                        result.append(current_chunk.strip())
                    current_chunk = sent[-self.chunk_overlap:] + " " + sent if current_chunk else sent
            if current_chunk:
                result.append(current_chunk.strip())
            return result
        
        # Finally split by words
        words = text.split()
        return [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size - self.chunk_overlap)]

def create_chunks(df: pd.DataFrame, chunk_size: int = 512, chunk_overlap: int = 64) -> pd.DataFrame:
    """Split narratives into chunks with metadata"""
    logger.info(f"Creating text chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    splitter = TextSplitter(chunk_size, chunk_overlap)
    chunks = []
    metadata = []
    
    for _, row in df.iterrows():
        text = row['clean_narrative']
        if pd.isna(text) or not text.strip():
            continue
            
        # Skip splitting very short texts
        word_count = len(text.split())
        if word_count < chunk_size // 2:
            chunks.append(text)
            metadata.append({
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'start_index': 0,
                'chunk_id': f"{row['Complaint ID']}_0"
            })
        else:
            text_chunks = splitter.split_text(text)
            for i, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                metadata.append({
                    'complaint_id': row['Complaint ID'],
                    'product': row['Product'],
                    'start_index': i * (chunk_size - chunk_overlap),
                    'chunk_id': f"{row['Complaint ID']}_{i}"
                })
    
    logger.info(f"Created {len(chunks)} chunks from {len(df)} narratives")
    return pd.DataFrame({'text': chunks, **pd.DataFrame(metadata)})

# Main execution guard with import protection
if __name__ == "__main__":
    # Use path helper
    input_path = get_data_path("processed", "filtered_complaints.csv")
    output_path = get_data_path("processed", "chunks.csv")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run data processing first")
    else:
        df = pd.read_csv(input_path)
        chunk_df = create_chunks(df)
        chunk_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(chunk_df)} chunks to {output_path}")