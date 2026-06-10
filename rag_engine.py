import os
import json
import re
import math
import requests
from typing import List, Dict, Any, Optional

# File path for index storage
INDEX_FILE = "data/index.json"
os.makedirs("data", exist_ok=True)

class TFIDFRetriever:
    """A pure Python TF-IDF retriever that requires no external dependencies."""
    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.doc_count = 0
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and alphanumeric words
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, chunks: List[Dict[str, Any]]):
        self.corpus = chunks
        self.doc_count = len(chunks)
        
        # Calculate Doc Frequency (DF) for each term
        df: Dict[str, int] = {}
        for chunk in chunks:
            tokens = set(self._tokenize(chunk["text"]))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
                
        # Calculate IDF (using smoothing)
        self.idf = {}
        for token, count in df.items():
            self.idf[token] = math.log((self.doc_count + 1) / (count + 1)) + 1.0

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.corpus:
            return []
            
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.corpus[:top_k]
            
        # Compute query vector
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
            
        query_vector: Dict[str, float] = {}
        for token, count in query_tf.items():
            if token in self.idf:
                query_vector[token] = count * self.idf[token]
                
        query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        if query_magnitude == 0:
            return self.corpus[:top_k]
            
        scores = []
        for chunk in self.corpus:
            chunk_tokens = self._tokenize(chunk["text"])
            chunk_tf: Dict[str, int] = {}
            for token in chunk_tokens:
                chunk_tf[token] = chunk_tf.get(token, 0) + 1
                
            # Cosine similarity calculation (dot product / (mag_q * mag_d))
            dot_product = 0.0
            chunk_mag_squared = 0.0
            
            for token, count in chunk_tf.items():
                if token in self.idf:
                    val = count * self.idf[token]
                    chunk_mag_squared += val ** 2
                    if token in query_vector:
                        dot_product += query_vector[token] * val
                        
            chunk_magnitude = math.sqrt(chunk_mag_squared)
            
            similarity = 0.0
            if query_magnitude > 0 and chunk_magnitude > 0:
                similarity = dot_product / (query_magnitude * chunk_magnitude)
                
            scores.append((chunk, similarity))
            
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, sim in scores[:top_k]:
            result = chunk.copy()
            result["score"] = round(sim, 4)
            results.append(result)
            
        return results

class EmbeddingsEngine:
    """Manages vector generation via external APIs or falls back to local TF-IDF."""
    def __init__(self, mode: str = "tfidf", api_key: Optional[str] = None):
        self.mode = mode.lower()
        self.api_key = api_key

    def get_embedding(self, text: str) -> Optional[List[float]]:
        if self.mode == "openai":
            if not self.api_key:
                return None
            try:
                url = "https://api.openai.com/v1/embeddings"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                data = {
                    "input": text,
                    "model": "text-embedding-3-small"
                }
                response = requests.post(url, headers=headers, json=data, timeout=10)
                if response.status_code == 200:
                    return response.json()["data"][0]["embedding"]
            except Exception as e:
                print(f"OpenAI embedding error: {e}")
                
        elif self.mode == "huggingface":
            try:
                # Using free Hugging Face API inference
                url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                response = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
                if response.status_code == 200:
                    res = response.json()
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], float):
                        return res
                    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                        return res[0]
            except Exception as e:
                print(f"Hugging Face embedding error: {e}")
                
        return None

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x ** 2 for x in v1))
    mag2 = math.sqrt(sum(y ** 2 for y in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

class RAGEngine:
    def __init__(self):
        self.index_file = INDEX_FILE
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = {"documents": {}, "chunks": []}
        else:
            self.index = {"documents": {}, "chunks": []}
            self.save_index()
            
        self.tfidf_retriever = TFIDFRetriever()
        if self.index["chunks"]:
            self.tfidf_retriever.fit(self.index["chunks"])

    def save_index(self):
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving index: {e}")

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        # Custom recursive splitter
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the end of the text, try to find a natural boundary
            if end < text_len:
                # Look for paragraph or sentence boundaries within the last 20% of the chunk size
                search_start = max(start, end - int(chunk_size * 0.25))
                sub_text = text[search_start:end]
                
                # Check for paragraph break, then sentence ending, then word boundary
                boundary = -1
                for pattern in [r'\n\n', r'\n', r'\. ', r'\? ', r'\! ', r' ']:
                    matches = list(re.finditer(pattern, sub_text))
                    if matches:
                        boundary = search_start + matches[-1].end()
                        break
                
                if boundary != -1:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = max(start + 1, end - overlap)
            
        return chunks

    def add_document(self, filename: str, content: str, doc_id: str, embedding_mode: str = "tfidf", api_key: Optional[str] = None) -> Dict[str, Any]:
        # Split text into chunks
        raw_chunks = self.chunk_text(content)
        
        # Initialize embedding engine
        engine = EmbeddingsEngine(mode=embedding_mode, api_key=api_key)
        
        doc_chunks = []
        for idx, text in enumerate(raw_chunks):
            chunk_data = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_idx": idx,
                "text": text,
                "embedding": None
            }
            
            # Generate embedding if using API-based retrieval
            if embedding_mode in ["openai", "huggingface"]:
                embedding = engine.get_embedding(text)
                if embedding:
                    chunk_data["embedding"] = embedding
                    
            doc_chunks.append(chunk_data)

        # Store document metadata
        self.index["documents"][doc_id] = {
            "filename": filename,
            "chunk_count": len(doc_chunks),
            "char_count": len(content),
            "embedding_mode": embedding_mode
        }
        
        # Append new chunks and save
        self.index["chunks"].extend(doc_chunks)
        self.save_index()
        
        # Refit TF-IDF retriever
        self.tfidf_retriever.fit(self.index["chunks"])
        
        return self.index["documents"][doc_id]

    def delete_document(self, doc_id: str):
        if doc_id in self.index["documents"]:
            del self.index["documents"][doc_id]
            # Filter out chunks for this document
            self.index["chunks"] = [c for c in self.index["chunks"] if c["doc_id"] != doc_id]
            self.save_index()
            # Refit TF-IDF retriever
            self.tfidf_retriever.fit(self.index["chunks"])
            return True
        return False

    def retrieve(self, query: str, top_k: int = 4, embedding_mode: str = "tfidf", api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.index["chunks"]:
            return []
            
        # 1. Fallback to TF-IDF if requested or if no embedding floats exist
        has_embeddings = all(c.get("embedding") is not None for c in self.index["chunks"])
        
        if embedding_mode == "tfidf" or not has_embeddings:
            return self.tfidf_retriever.retrieve(query, top_k=top_k)
            
        # 2. Vector search using API embeddings
        engine = EmbeddingsEngine(mode=embedding_mode, api_key=api_key)
        query_embedding = engine.get_embedding(query)
        
        if not query_embedding:
            # Fallback if API fails
            print(f"Embedding API failed for query. Falling back to local TF-IDF.")
            return self.tfidf_retriever.retrieve(query, top_k=top_k)
            
        # Calculate cosine similarity
        scores = []
        for chunk in self.index["chunks"]:
            if chunk.get("embedding") is not None:
                sim = cosine_similarity(query_embedding, chunk["embedding"])
                scores.append((chunk, sim))
            else:
                scores.append((chunk, 0.0))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, sim in scores[:top_k]:
            res = chunk.copy()
            # Remove embedding from output to keep payload light
            if "embedding" in res:
                del res["embedding"]
            res["score"] = round(sim, 4)
            results.append(res)
            
        return results

    def get_documents(self) -> Dict[str, Any]:
        return self.index["documents"]
