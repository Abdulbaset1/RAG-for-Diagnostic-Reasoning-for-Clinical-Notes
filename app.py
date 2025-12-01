# app.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import torch
import os
import zipfile
import requests
from io import BytesIO
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'documents_df' not in st.session_state:
    st.session_state.documents_df = None

# Data Loader Class
class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_data(self) -> pd.DataFrame:
        records = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
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
                        continue
                        
        df = pd.DataFrame(records)
        return df

# Retriever Class with scikit-learn alternative to FAISS
class ClinicalRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = []
        self.df = None

    def build_index(self, df: pd.DataFrame):
        texts = df['content'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(embeddings).astype('float32')
        self.documents = df.to_dict('records')
        self.df = df

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        if self.embeddings is None:
            return []
        
        query_vector = self.model.encode([query]).astype('float32')
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Get top k indices
        if k > len(similarities):
            k = len(similarities)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'content': doc['content'],
                'disease': doc['disease'],
                'score': float(similarities[idx])  # Using similarity score instead of distance
            })
        return results

# Custom RAG Pipeline Class
class CustomRAGPipeline:
    def __init__(self, dataset_path: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        # Step 1: Load and preprocess data
        self.data_loader = DataLoader(dataset_path)
        self.documents_df = self.data_loader.load_data()
        
        # Step 2: Build retrieval index
        self.retriever = ClinicalRetriever()
        self.retriever.build_index(self.documents_df)
        
        # Step 3: Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use CPU if CUDA is not available
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        self.model.eval()
        
        # Step 4: Set generation parameters
        self.generation_config = {
            "max_new_tokens": 200,
            "temperature": 0.1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        return self.retriever.retrieve(query, k=k)
    
    def prepare_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        context_text = ""
        for i, doc in enumerate(retrieved_docs[:2]):
            snippet = doc['content'][:500]
            context_text += f"Document {i+1} ({doc['disease']}):\n{snippet}\n\n"
        
        prompt = f"""You are a clinical assistant. Answer the question based ONLY on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **self.generation_config
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
            answer = answer.split("\n\nDocument")[0].split("\n\n\n")[0].strip()
            return answer
        return full_text
    
    def run(self, query: str) -> Dict:
        retrieved_docs = self.retrieve_context(query)
        prompt = self.prepare_prompt(query, retrieved_docs)
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

# Function to download dataset from GitHub
def download_dataset_from_github():
    github_url = "https://github.com/Abdulbaset1/RAG-for-Diagnostic-Reasoning-for-Clinical-Notes/raw/main/mimic-iv-ext-direct-1.0.0.zip"
    
    try:
        progress_bar = st.progress(0, text="Downloading dataset from GitHub...")
        
        response = requests.get(github_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        zip_file = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress, text=f"Downloading dataset: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
        
        progress_bar.progress(1.0, text="Extracting dataset...")
        
        with zipfile.ZipFile(zip_file) as z:
            extract_path = Path("./data")
            extract_path.mkdir(exist_ok=True)
            z.extractall(extract_path)
        
        # Find the Finished folder
        dataset_path = None
        for root, dirs, files in os.walk("./data"):
            if "Finished" in dirs:
                dataset_path = os.path.join(root, "Finished")
                break
        
        progress_bar.empty()
        
        if dataset_path and os.path.exists(dataset_path):
            return dataset_path
        else:
            st.error("Could not find 'Finished' folder in extracted dataset")
            return None
            
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        st.error(f"Error downloading dataset: {str(e)}")
        return None

# Function to check for local dataset
def find_local_dataset():
    possible_paths = [
        "./mimic-iv-ext-direct-1.0.0/Finished",
        "./data/mimic-iv-ext-direct-1.0.0/Finished",
        "../input/mimic-iv-ext-direct-1.0.0/Finished",
        "/kaggle/input/mimic-iv-ext-direct-1.0.0/Finished",
        "mimic-iv-ext-direct-1.0.0/Finished"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Initialize the application
def initialize_app():
    if st.session_state.pipeline is None or not st.session_state.dataset_loaded:
        dataset_path = None
        
        # Check for local dataset first
        with st.spinner("Looking for local dataset..."):
            dataset_path = find_local_dataset()
        
        # If not found locally, try to download from GitHub
        if dataset_path is None:
            st.info("Local dataset not found. Downloading from GitHub...")
            dataset_path = download_dataset_from_github()
        
        if dataset_path and os.path.exists(dataset_path):
            with st.spinner("Initializing Clinical RAG System..."):
                try:
                    # Initialize pipeline
                    st.session_state.pipeline = CustomRAGPipeline(dataset_path)
                    st.session_state.evaluator = RAGEvaluator()
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_path = dataset_path
                    return True
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        else:
            st.error("Dataset not found. Please ensure the dataset is available.")
            return False
    return True

# Main application layout
def main():
    st.title("Clinical RAG Assistant for Diagnostic Reasoning")
    st.markdown("---")
    
    # Sidebar for controls and information
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("Initialize/Restart System", type="primary", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.dataset_loaded = False
            st.session_state.history = []
            st.rerun()
        
        st.markdown("---")
        st.header("System Information")
        
        if st.session_state.dataset_loaded:
            st.success("✓ System Initialized")
            if st.session_state.pipeline and hasattr(st.session_state.pipeline, 'documents_df'):
                st.write(f"**Documents loaded:** {len(st.session_state.pipeline.documents_df)}")
            if st.session_state.dataset_path:
                st.write(f"**Dataset path:** {st.session_state.dataset_path}")
        else:
            st.warning("⚠ System not initialized")
        
        st.markdown("---")
        st.header("Sample Queries")
        
        sample_queries = [
            "What are the symptoms of heart failure?",
            "How is diabetes diagnosed?",
            "What causes stroke?",
            "Treatment options for hypertension",
            "Signs of pulmonary embolism"
        ]
        
        for query in sample_queries:
            if st.button(query, use_container_width=True):
                st.session_state.query_input = query
                st.rerun()
        
        st.markdown("---")
        st.header("Query History")
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                st.caption(f"{len(st.session_state.history)-i}. {item['query'][:50]}...")
        else:
            st.caption("No queries yet")
    
    # Main content area
    if not st.session_state.dataset_loaded:
        st.header("System Initialization")
        st.write("The Clinical RAG Assistant requires initialization before use.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Initialize System", type="primary", use_container_width=True, key="init_btn"):
                if initialize_app():
                    st.success("System initialized successfully!")
                    st.rerun()
        
        st.markdown("---")
        st.write("### System Requirements")
        st.write("""
        1. The system will download the MIMIC-IV clinical notes dataset from GitHub
        2. Initial setup may take 3-5 minutes to load models and build indexes
        3. The system uses efficient models suitable for cloud deployment
        4. All processing happens in memory - no disk persistence required
        """)
        
        st.write("### Expected Components")
        with st.expander("Click to view system components"):
            st.write("""
            - **Embedding Model**: all-MiniLM-L6-v2 (80MB)
            - **Language Model**: microsoft/Phi-3-mini-4k-instruct (2GB)
            - **Retrieval Engine**: Cosine similarity-based
            - **Dataset**: MIMIC-IV Clinical Notes (will be downloaded)
            """)
        
    else:
        # Query input section
        st.header("Clinical Query Interface")
        
        # Query input with history recall
        query_input = st.text_area(
            "Enter your clinical query:",
            value=st.session_state.get('query_input', ''),
            height=100,
            placeholder="Example: What are the symptoms of heart failure?",
            key="query_input_area"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            retrieve_k = st.slider("Documents to retrieve:", min_value=1, max_value=5, value=3, key="retrieve_k")
        with col2:
            max_tokens = st.slider("Max response tokens:", min_value=50, max_value=500, value=200, key="max_tokens")
        with col3:
            submit_col1, submit_col2 = st.columns([3, 1])
            with submit_col1:
                if st.button("Submit Query", type="primary", use_container_width=True, key="submit_query"):
                    if query_input.strip():
                        with st.spinner("Processing query..."):
                            try:
                                # Update generation config
                                st.session_state.pipeline.generation_config["max_new_tokens"] = max_tokens
                                
                                # Run pipeline
                                result = st.session_state.pipeline.run(query_input)
                                
                                # Evaluate response
                                metrics = st.session_state.evaluator.evaluate_pipeline_response(result)
                                
                                # Store in history
                                st.session_state.history.append({
                                    "query": query_input,
                                    "result": result,
                                    "metrics": metrics
                                })
                                
                                # Display results
                                display_results(result, metrics)
                                
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.warning("Please enter a query")
            with submit_col2:
                if st.button("Clear", type="secondary", use_container_width=True):
                    st.session_state.query_input = ""
                    st.rerun()
        
        st.markdown("---")
        
        # Display last result if available
        if st.session_state.history:
            display_last_result()
        
        # System metrics section
        st.header("System Performance")
        display_system_metrics()

# Function to display query results
def display_results(result, metrics):
    st.subheader("Generated Answer")
    
    # Create a container for the answer with custom styling
    answer_container = st.container(border=True)
    with answer_container:
        st.markdown(f"""
        <div style='
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
        '>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics display
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precision_value = metrics['retrieval']['precision'] if metrics['retrieval']['precision'] is not None else "N/A"
        precision_display = f"{precision_value:.2f}" if isinstance(precision_value, (int, float)) else precision_value
        st.metric(
            label="Retrieval Precision",
            value=precision_display
        )
    
    with col2:
        st.metric(
            label="Answer Relevance",
            value=f"{metrics['generation']['query_answer_similarity']:.2f}"
        )
    
    with col3:
        grounded_status = "Grounded" if metrics['generation']['is_grounded'] else "Not Grounded"
        color = "green" if metrics['generation']['is_grounded'] else "red"
        st.metric(
            label="Answer Grounding",
            value=grounded_status
        )
    
    with col4:
        st.metric(
            label="Retrieval Diversity",
            value=f"{metrics['retrieval']['diversity']:.2f}"
        )
    
    # Retrieved documents
    st.subheader("Retrieved Documents")
    
    if result['retrieved_documents']:
        for i, doc in enumerate(result['retrieved_documents']):
            with st.expander(f"Document {i+1}: {doc['disease']} (Relevance: {doc['score']:.4f})", expanded=(i==0)):
                st.write(f"**Content Preview:**")
                st.text(doc['content'][:500] + "...")
    else:
        st.warning("No documents retrieved")

# Function to display last result
def display_last_result():
    if st.session_state.history:
        last_result = st.session_state.history[-1]
        
        st.subheader("Last Query Result")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Query:** {last_result['query']}")
        with col2:
            if st.button("Clear History", type="secondary"):
                st.session_state.history = []
                st.rerun()
        
        if len(st.session_state.history) > 1:
            st.write(f"**Previous queries in session:** {len(st.session_state.history)-1}")

# Function to display system metrics
def display_system_metrics():
    if st.session_state.history:
        # Calculate average metrics
        retrieval_precisions = []
        answer_relevance = []
        grounded_answers = 0
        
        for item in st.session_state.history:
            if item['metrics']['retrieval']['precision'] is not None:
                retrieval_precisions.append(item['metrics']['retrieval']['precision'])
            answer_relevance.append(item['metrics']['generation']['query_answer_similarity'])
            if item['metrics']['generation']['is_grounded']:
                grounded_answers += 1
        
        if retrieval_precisions:
            avg_precision = np.mean(retrieval_precisions)
        else:
            avg_precision = 0
            
        if answer_relevance:
            avg_relevance = np.mean(answer_relevance)
        else:
            avg_relevance = 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Average Retrieval Precision",
                value=f"{avg_precision:.2f}" if retrieval_precisions else "N/A"
            )
        
        with col2:
            st.metric(
                label="Average Answer Relevance",
                value=f"{avg_relevance:.2f}"
            )
        
        with col3:
            if st.session_state.history:
                grounded_pct = (grounded_answers / len(st.session_state.history)) * 100
                st.metric(
                    label="Grounded Answers",
                    value=f"{grounded_pct:.1f}%"
                )
            else:
                st.metric(
                    label="Grounded Answers",
                    value="0%"
                )
    
    # System information
    st.subheader("System Configuration")
    
    if st.session_state.pipeline:
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Retrieval Model:**")
            st.write("all-MiniLM-L6-v2")
            
            st.write("**LLM Model:**")
            st.write("microsoft/Phi-3-mini-4k-instruct")
            
            st.write("**Retrieval Method:**")
            st.write("Cosine Similarity")
        
        with info_col2:
            if hasattr(st.session_state.pipeline, 'documents_df'):
                st.write("**Document Count:**")
                st.write(len(st.session_state.pipeline.documents_df))
            
            if hasattr(st.session_state.pipeline.retriever, 'embeddings') and st.session_state.pipeline.retriever.embeddings is not None:
                st.write("**Embedding Dimension:**")
                st.write(st.session_state.pipeline.retriever.embeddings.shape[1])
            
            st.write("**Python Version:**")
            st.write(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("If you encounter dependency issues, please ensure all requirements are properly installed.")
        import traceback
        with st.expander("View detailed error traceback"):
            st.code(traceback.format_exc())
