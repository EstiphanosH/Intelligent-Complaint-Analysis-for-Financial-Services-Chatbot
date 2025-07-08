import os
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from src.utils import configure_logging

logger = configure_logging()

# Use open-access models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"  # Open-access alternative

class RAGSystem:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing RAG system on {self.device.upper()}")
        
        # Initialize components
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.vector_db = self._init_vector_db()
        
        # Initialize text generation model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise
        
        # Prompt template optimized for FLAN-T5
        self.prompt_template = """Answer the question based on the context below. If you don't know the answer, say "I don't have enough information".

Context: {context}

Question: {question}

Answer: """

    def _init_vector_db(self) -> chromadb.Collection:
        try:
            client = chromadb.PersistentClient(
                # Correct path to vector store
                path=os.path.join(os.path.dirname(__file__), "..", "vectorstore"),
                settings=Settings(allow_reset=True)
            )
            # Use correct collection name
            return client.get_collection("complaint_vectors")
        except Exception as e:
            logger.error(f"Vector DB initialization failed: {str(e)}")
            # Try to create collection if it doesn't exist
            try:
                logger.warning("Attempting to create collection")
                collection = client.create_collection("complaint_vectors")
                logger.info("Created new collection 'complaint_vectors'")
                return collection
            except Exception as create_error:
                logger.critical(f"Failed to create collection: {str(create_error)}")
                raise


    def generate_response(self, query: str, max_length=512) -> dict:
        """Generate answer using RAG pipeline"""
        try:
            # Retrieve context
            retrieval_results = self.retrieve(query)
            context = "\n".join([
                f"[Source {i+1}] {text[:200]}..." 
                for i, text in enumerate(retrieval_results['documents'][0])
            ])
            
            # Format prompt
            prompt = self.prompt_template.format(context=context, question=query)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            if self.device == "cuda":
                inputs = inputs.to("cuda")
            
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                num_return_sequences=1
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "answer": answer,
                "sources": retrieval_results
            }
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "answer": "Error processing your request",
                "sources": {'documents': [], 'metadatas': []}
            }