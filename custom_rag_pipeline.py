# Install required libraries first
# !pip install -q sentence-transformers faiss-cpu transformers torch pandas scikit-learn

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pandas as pd
import numpy as np
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

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

# Custom RAG Pipeline Class (No library pipeline used)
class CustomRAGPipeline:
    """
    Custom RAG Pipeline that manually handles:
    1. Data Loading
    2. Document Retrieval
    3. Context Preparation
    4. LLM Inference
    5. Response Generation
    """
    
    def __init__(self, dataset_path: str, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print("="*60)
        print("INITIALIZING CUSTOM RAG PIPELINE")
        print("="*60)
        
        # Step 1: Load and preprocess data
        print("\n[Step 1/4] Loading dataset...")
        self.data_loader = DataLoader(dataset_path)
        self.documents_df = self.data_loader.load_data()
        
        # Step 2: Build retrieval index
        print("\n[Step 2/4] Building retrieval index...")
        self.retriever = ClinicalRetriever()
        self.retriever.build_index(self.documents_df)
        
        # Step 3: Load LLM manually (no pipeline)
        print("\n[Step 3/4] Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded successfully.")
        
        # Step 4: Set generation parameters
        print("\n[Step 4/4] Configuring generation parameters...")
        self.generation_config = {
            "max_new_tokens": 200,
            "temperature": 0.1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": False  # Disable cache to avoid compatibility issues
        }
        print("Pipeline initialization complete!")
        print("="*60 + "\n")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Step 1 of pipeline: Retrieve relevant documents
        """
        return self.retriever.retrieve(query, k=k)
    
    def prepare_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Step 2 of pipeline: Prepare prompt with context
        """
        # Build context from top 2 documents
        context_text = ""
        for i, doc in enumerate(retrieved_docs[:2]):
            snippet = doc['content'][:500]
            context_text += f"Document {i+1} ({doc['disease']}):\n{snippet}\n\n"
        
        # Create structured prompt
        prompt = f"""You are a clinical assistant. Answer the question based ONLY on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Step 3 of pipeline: Generate response using LLM (manual inference)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate tokens manually
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # Decode output
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
            answer = answer.split("\n\nDocument")[0].split("\n\n\n")[0].strip()
            return answer
        return full_text
    
    def run(self, query: str) -> Dict:
        """
        Complete RAG pipeline execution
        Returns: Dictionary with query, retrieved docs, and generated answer
        """
        # Pipeline Step 1: Retrieve
        retrieved_docs = self.retrieve_context(query)
        
        # Pipeline Step 2: Prepare prompt
        prompt = self.prepare_prompt(query, retrieved_docs)
        
        # Pipeline Step 3: Generate
        answer = self.generate_response(prompt)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }

# Evaluation Class
class RAGEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_retrieval_relevance(self, query: str, retrieved_docs: List[Dict], 
                                     expected_disease: str = None) -> Dict:
        if not retrieved_docs:
            return {"precision": 0.0, "avg_score": 0.0, "num_retrieved": 0}
        
        avg_score = np.mean([doc['score'] for doc in retrieved_docs])
        precision = None
        if expected_disease:
            relevant_count = sum(1 for doc in retrieved_docs if expected_disease.lower() in doc['disease'].lower())
            precision = relevant_count / len(retrieved_docs)
        
        return {
            "precision": precision,
            "avg_score": avg_score,
            "num_retrieved": len(retrieved_docs),
            "diseases_found": list(set([doc['disease'] for doc in retrieved_docs]))
        }
    
    def calculate_retrieval_diversity(self, retrieved_docs: List[Dict]) -> float:
        if not retrieved_docs:
            return 0.0
        unique_diseases = len(set([doc['disease'] for doc in retrieved_docs]))
        return unique_diseases / len(retrieved_docs)
    
    def evaluate_answer_relevance(self, query: str, answer: str, context_docs: List[Dict]) -> Dict:
        query_embedding = self.embedding_model.encode([query])
        answer_embedding = self.embedding_model.encode([answer])
        query_answer_sim = cosine_similarity(query_embedding, answer_embedding)[0][0]
        
        context_text = " ".join([doc['content'][:200] for doc in context_docs[:2]])
        context_embedding = self.embedding_model.encode([context_text])
        answer_context_sim = cosine_similarity(answer_embedding, context_embedding)[0][0]
        
        return {
            "query_answer_similarity": float(query_answer_sim),
            "answer_context_similarity": float(answer_context_sim),
            "is_grounded": answer_context_sim > 0.3
        }
    
    def evaluate_pipeline_response(self, result: Dict, expected_disease: str = None) -> Dict:
        retrieval_metrics = self.evaluate_retrieval_relevance(
            result['query'], 
            result['retrieved_documents'], 
            expected_disease
        )
        diversity = self.calculate_retrieval_diversity(result['retrieved_documents'])
        relevance_metrics = self.evaluate_answer_relevance(
            result['query'], 
            result['answer'], 
            result['retrieved_documents']
        )
        
        return {
            "retrieval": {**retrieval_metrics, "diversity": diversity},
            "generation": relevance_metrics
        }
    
    def run_test_suite(self, test_queries: List[Dict], pipeline) -> pd.DataFrame:
        results = []
        for i, test in enumerate(test_queries):
            query = test['query']
            expected_disease = test.get('expected_disease', None)
            
            print(f"Evaluating {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Run custom pipeline
            result = pipeline.run(query)
            metrics = self.evaluate_pipeline_response(result, expected_disease)
            
            results.append({
                "query": query,
                "expected_disease": expected_disease,
                "retrieval_precision": metrics['retrieval']['precision'],
                "retrieval_avg_score": metrics['retrieval']['avg_score'],
                "retrieval_diversity": metrics['retrieval']['diversity'],
                "query_answer_similarity": metrics['generation']['query_answer_similarity'],
                "answer_grounded": metrics['generation']['is_grounded']
            })
        
        return pd.DataFrame(results)

# Configuration
DATASET_PATH = "/kaggle/input/rag-dataset/mimic-iv-ext-direct-1.0.0/Finished" 

# Test queries
TEST_QUERIES = [
    {"query": "What are the symptoms of heart failure?", "expected_disease": "Heart Failure"},
    {"query": "How is diabetes diagnosed?", "expected_disease": "Diabetes"},
    {"query": "What causes stroke?", "expected_disease": "Stroke"},
    {"query": "Treatment options for hypertension", "expected_disease": "Hypertension"},
    {"query": "Signs of pulmonary embolism", "expected_disease": "Pulmonary Embolism"}
]

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path not found: {DATASET_PATH}")
        print("Searching for dataset...")
        for root, dirs, files in os.walk("/kaggle/input"):
            if "Finished" in dirs:
                DATASET_PATH = os.path.join(root, "Finished")
                print(f"Found dataset at: {DATASET_PATH}")
                break
    
    if os.path.exists(DATASET_PATH):
        # Initialize custom RAG pipeline
        rag_pipeline = CustomRAGPipeline(DATASET_PATH)
        
        # Run evaluation first
        print("\n" + "="*60)
        print("RUNNING EVALUATION ON TEST QUERIES")
        print("="*60)
        
        evaluator = RAGEvaluator()
        results_df = evaluator.run_test_suite(TEST_QUERIES, rag_pipeline)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Average Retrieval Precision: {results_df['retrieval_precision'].mean():.2f}")
        print(f"Average Query-Answer Similarity: {results_df['query_answer_similarity'].mean():.2f}")
        print(f"Percentage of Grounded Answers: {results_df['answer_grounded'].mean()*100:.1f}%")
        print(f"Average Retrieval Diversity: {results_df['retrieval_diversity'].mean():.2f}")
        
        # Now start interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("System ready. Type 'exit' to stop.")
        
        while True:
            query = input("\nEnter clinical query: ")
            if query.lower() == 'exit': break
            
            print("Running RAG pipeline...")
            result = rag_pipeline.run(query)
            
            print("\n" + "="*50)
            print(f"ANSWER: {result['answer']}")
            print("="*50)
            print(f"\nRetrieved from: {[doc['disease'] for doc in result['retrieved_documents']]}")
