import os
import json
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import streamlit as st

# Data Loader Class
class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_data(self) -> pd.DataFrame:
        records = []
        print(f"Scanning directory: {self.base_path}")
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Get clinical note sections
                        content_parts = []
                        for i in range(1, 7):
                            key = f"input{i}"
                            if key in data and isinstance(data[key], str):
                                content_parts.append(data[key].strip())
                        
                        full_content = "\n\n".join(content_parts)
                        disease = os.path.basename(root)
                        file_id = file.replace(".json", "")
                        
                        if full_content:
                            records.append({
                                "file_id": file_id,
                                "disease": disease,
                                "content": full_content
                            })
                    except Exception as e:
                        pass 
                        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} records.")
        return df

# Retriever Class
class ClinicalRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # This model is small (~80MB) and fits easily in memory
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, df: pd.DataFrame):
        print("Encoding documents...")
        texts = df['content'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = df.to_dict('records')
        print("Index built successfully.")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'content': doc['content'],
                    'disease': doc['disease'],
                    'score': float(distances[0][i])
                })
        return results

# Custom RAG Pipeline Class (Gemini API Version)
class CustomRAGPipeline:
    """
    Custom RAG Pipeline that uses Google Gemini API for generation.
    This is lightweight and suitable for Streamlit Cloud free tier.
    """
    
    def __init__(self, dataset_path: str):
        print("="*60)
        print("INITIALIZING GEMINI RAG PIPELINE")
        print("="*60)
        
        # Step 1: Configure API
        api_key = AIzaSyDm6AG6_hFc3PDmYNIbJF-t2945o6UHEOQ
        
        # Try getting key from Streamlit secrets first (for Cloud)
        try:
            api_key = st.secrets["AIzaSyDm6AG6_hFc3PDmYNIbJF-t2945o6UHEOQ"]
        except:
            pass
            
        # Try getting from environment variable (for local/Kaggle)
        if not api_key:
            api_key = os.environ.get("AIzaSyDm6AG6_hFc3PDmYNIbJF-t2945o6UHEOQ")
            
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Please set it in Streamlit secrets or environment variables.")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Step 2: Load and preprocess data
        print("\n[Step 1/2] Loading dataset...")
        self.data_loader = DataLoader(dataset_path)
        self.documents_df = self.data_loader.load_data()
        
        # Step 3: Build retrieval index
        print("\n[Step 2/2] Building retrieval index...")
        self.retriever = ClinicalRetriever()
        self.retriever.build_index(self.documents_df)
        
        print("Pipeline initialization complete!")
        print("="*60 + "\n")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        return self.retriever.retrieve(query, k=k)
    
    def prepare_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        # Build context from top 3 documents
        context_text = ""
        for i, doc in enumerate(retrieved_docs):
            snippet = doc['content'][:1000] # Can use more context with Gemini
            context_text += f"Document {i+1} ({doc['disease']}):\n{snippet}\n\n"
        
        prompt = f"""You are a clinical assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot answer this based on the provided clinical notes."

Context:
{context_text}

Question: {query}

Answer:"""
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def run(self, query: str) -> Dict:
        retrieved_docs = self.retrieve_context(query)
        prompt = self.prepare_prompt(query, retrieved_docs)
        answer = self.generate_response(prompt)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }
