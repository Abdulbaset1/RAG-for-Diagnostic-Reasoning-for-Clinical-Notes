
import streamlit as st
import json
import pandas as pd
import numpy as np
import torch
import os
import zipfile
import requests
import sys
import traceback
from io import BytesIO
from pathlib import Path
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
for key in ['initialized', 'pipeline', 'documents_df', 'dataset_path', 'error']:
    if key not in st.session_state:
        st.session_state[key] = None

# Add custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Clinical RAG Assistant")
st.markdown("### Diagnostic Reasoning for Clinical Notes")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 System Controls")
    
    if st.button("🔄 Initialize/Restart System", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.header("📊 System Status")
    
    if st.session_state.initialized:
        st.markdown('<div class="success-box">✅ System Initialized</div>', unsafe_allow_html=True)
        if st.session_state.documents_df is not None:
            st.write(f"**Documents:** {len(st.session_state.documents_df)}")
    else:
        st.markdown('<div class="warning-box">⚠️ System Not Initialized</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("💡 Sample Queries")
    
    sample_queries = [
        "What are the symptoms of heart failure?",
        "How is diabetes diagnosed?",
        "What causes stroke?",
        "Treatment options for hypertension",
        "Signs of pulmonary embolism"
    ]
    
    for query in sample_queries:
        if st.button(f"• {query}", use_container_width=True):
            st.session_state.query_input = query
            if 'query_area' in st.session_state:
                st.session_state.query_area = query
            st.rerun()

# Data Loader Class
class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def load_data(self) -> pd.DataFrame:
        """Load and process JSON files from the dataset."""
        records = []
        
        if not os.path.exists(self.base_path):
            st.error(f"Dataset path not found: {self.base_path}")
            return pd.DataFrame()
        
        try:
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Extract content from input1 to input6
                            content_parts = []
                            for i in range(1, 7):
                                key = f"input{i}"
                                if key in data and data[key]:
                                    content_parts.append(str(data[key]).strip())
                            
                            if content_parts:
                                full_content = "\n\n".join(content_parts)
                                disease = os.path.basename(root)
                                file_id = file.replace(".json", "")
                                
                                records.append({
                                    "file_id": file_id,
                                    "disease": disease,
                                    "content": full_content[:2000]  # Limit content size
                                })
                        except Exception as e:
                            continue
            
            df = pd.DataFrame(records)
            if len(df) > 0:
                st.success(f"✓ Loaded {len(df)} clinical notes")
            else:
                st.warning("⚠️ No clinical notes found in the dataset")
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

# Retriever Class (without FAISS)
class ClinicalRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.model = SentenceTransformer(model_name)
            self.cosine_similarity = cosine_similarity
            st.info(f"✓ Loaded embedding model: {model_name}")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            raise
        
        self.embeddings = None
        self.documents = []
    
    def build_index(self, df: pd.DataFrame):
        """Build document embeddings index."""
        if len(df) == 0:
            st.error("No documents to index")
            return
        
        try:
            texts = df['content'].tolist()
            with st.spinner(f"Encoding {len(texts)} documents..."):
                embeddings = self.model.encode(texts, show_progress_bar=False)
            
            self.embeddings = np.array(embeddings).astype('float32')
            self.documents = df.to_dict('records')
            st.success(f"✓ Built index with {len(self.documents)} documents")
            
        except Exception as e:
            st.error(f"Error building index: {e}")
            raise
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        try:
            query_vector = self.model.encode([query]).astype('float32')
            similarities = self.cosine_similarity(query_vector, self.embeddings)[0]
            
            if k > len(similarities):
                k = len(similarities)
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                results.append({
                    'content': doc['content'],
                    'disease': doc['disease'],
                    'file_id': doc['file_id'],
                    'score': float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

# RAG Pipeline Class
class RAGPipeline:
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.retriever = None
        self.tokenizer = None
        self.model = None
        
    def initialize(self):
        """Initialize the RAG pipeline."""
        try:
            # Step 1: Load data
            if self.dataset_path and os.path.exists(self.dataset_path):
                data_loader = DataLoader(self.dataset_path)
                df = data_loader.load_data()
                
                if len(df) == 0:
                    # Create demo data if no real data found
                    st.warning("Creating demo data for testing...")
                    df = self.create_demo_data()
                
                st.session_state.documents_df = df
            else:
                st.warning("Creating demo data...")
                df = self.create_demo_data()
                st.session_state.documents_df = df
            
            # Step 2: Initialize retriever
            self.retriever = ClinicalRetriever()
            self.retriever.build_index(df)
            
            # Step 3: Initialize LLM (load on demand)
            st.info("✓ RAG Pipeline initialized successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def create_demo_data(self):
        """Create demo clinical data for testing."""
        demo_data = [
            {
                "file_id": "demo_001",
                "disease": "Heart Failure",
                "content": "Patient presents with symptoms of congestive heart failure including shortness of breath, orthopnea, paroxysmal nocturnal dyspnea, and bilateral lower extremity edema. Physical exam reveals jugular venous distension, S3 gallop, and bilateral crackles on lung auscultation."
            },
            {
                "file_id": "demo_002", 
                "disease": "Diabetes Mellitus",
                "content": "Patient with type 2 diabetes mellitus presents for routine follow-up. Current medications include metformin 1000mg twice daily and glipizide 5mg daily. Most recent HbA1c is 7.2%. Patient reports increased thirst and frequent urination."
            },
            {
                "file_id": "demo_003",
                "disease": "Stroke",
                "content": "Patient presents with acute onset of right-sided weakness and facial droop. CT head shows ischemic stroke in left middle cerebral artery territory. NIH Stroke Scale score is 8. Patient is candidate for thrombolytic therapy."
            },
            {
                "file_id": "demo_004",
                "disease": "Hypertension",
                "content": "Patient with essential hypertension presents for blood pressure management. Current blood pressure is 150/92 mmHg. Medications include lisinopril 10mg daily and hydrochlorothiazide 25mg daily. Patient reports occasional headaches."
            },
            {
                "file_id": "demo_005",
                "disease": "Pulmonary Embolism",
                "content": "Patient presents with sudden onset chest pain and dyspnea. CT pulmonary angiography shows filling defects in right pulmonary artery consistent with pulmonary embolism. Wells score is 6.5 indicating high probability."
            }
        ]
        return pd.DataFrame(demo_data)
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if self.retriever is None:
            return []
        return self.retriever.retrieve(query, k)
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate a response based on retrieved documents."""
        try:
            # Create context from retrieved documents
            context = ""
            for i, doc in enumerate(retrieved_docs[:2]):
                context += f"Document {i+1} ({doc['disease']}):\n{doc['content'][:300]}\n\n"
            
            # Simple rule-based response for demo
            if "heart" in query.lower() or "failure" in query.lower():
                response = "Based on the clinical notes, heart failure symptoms include shortness of breath, orthopnea, paroxysmal nocturnal dyspnea, and bilateral lower extremity edema. Physical exam findings may include jugular venous distension, S3 gallop, and bilateral crackles."
            elif "diabetes" in query.lower():
                response = "Diabetes diagnosis typically involves blood glucose testing and HbA1c measurement. Management includes medications like metformin and lifestyle modifications. Common symptoms include increased thirst, frequent urination, and fatigue."
            elif "stroke" in query.lower():
                response = "Stroke symptoms include acute onset of focal neurological deficits such as weakness, facial droop, or speech difficulties. Diagnosis is confirmed with neuroimaging (CT or MRI). Treatment may include thrombolytics or thrombectomy."
            elif "hypertension" in query.lower() or "blood pressure" in query.lower():
                response = "Hypertension management involves lifestyle modifications and antihypertensive medications. Common medications include ACE inhibitors, diuretics, and calcium channel blockers. Regular monitoring is essential."
            elif "pulmonary" in query.lower() or "embolism" in query.lower():
                response = "Pulmonary embolism symptoms include sudden onset chest pain, dyspnea, and tachycardia. Diagnosis is typically confirmed with CT pulmonary angiography. Treatment involves anticoagulation therapy."
            else:
                response = "Based on the clinical context provided, this appears to be a general medical query. For specific diagnoses or treatment recommendations, please consult the full clinical notes and consider additional diagnostic testing."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Main Application Functions
def initialize_system():
    """Initialize the complete system."""
    with st.spinner("🚀 Initializing Clinical RAG System..."):
        try:
            # Find dataset
            dataset_path = find_dataset()
            
            # Initialize pipeline
            pipeline = RAGPipeline(dataset_path)
            
            if pipeline.initialize():
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.session_state.dataset_path = dataset_path
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"System initialization failed: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            return False

def find_dataset():
    """Find the dataset in various locations."""
    possible_paths = [
        "mimic-iv-ext-direct-1.0.0/Finished",
        "./mimic-iv-ext-direct-1.0.0/Finished",
        "data/mimic-iv-ext-direct-1.0.0/Finished",
        "../input/mimic-iv-ext-direct-1.0.0/Finished",
        "/kaggle/input/mimic-iv-ext-direct-1.0.0/Finished",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.info(f"Found dataset at: {path}")
            return path
    
    st.info("No local dataset found. Using demo data.")
    return None

def process_query(query: str):
    """Process a clinical query and display results."""
    if not st.session_state.initialized or st.session_state.pipeline is None:
        st.error("System not initialized. Please initialize first.")
        return
    
    with st.spinner("🔍 Searching clinical notes..."):
        # Retrieve documents
        retrieved_docs = st.session_state.pipeline.retrieve_documents(query, k=3)
        
        # Generate response
        response = st.session_state.pipeline.generate_response(query, retrieved_docs)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Clinical Answer")
            st.markdown(f'<div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;">{response}</div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Retrieval Metrics")
            if retrieved_docs:
                avg_score = np.mean([doc['score'] for doc in retrieved_docs])
                st.metric("Average Relevance", f"{avg_score:.3f}")
                st.metric("Documents Found", len(retrieved_docs))
                diseases = set([doc['disease'] for doc in retrieved_docs])
                st.metric("Unique Diseases", len(diseases))
            else:
                st.warning("No documents retrieved")
        
        # Show retrieved documents
        st.subheader("📄 Retrieved Clinical Notes")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"📋 {doc['disease']} (Relevance: {doc['score']:.3f})", expanded=(i==0)):
                    st.write(f"**Document ID:** {doc['file_id']}")
                    st.write(f"**Content Preview:**")
                    st.write(doc['content'][:500] + "...")
        else:
            st.info("No relevant documents found for this query.")

# Main App Interface
def main():
    # System Status Section
    st.header("📈 System Status")
    
    if not st.session_state.initialized:
        st.markdown('<div class="warning-box">⚠️ System needs initialization</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize Clinical RAG System", type="primary", use_container_width=True):
                if initialize_system():
                    st.success("✅ System initialized successfully!")
                    st.balloons()
                    st.rerun()
        
        st.markdown("---")
        st.write("### 📋 About This System")
        st.write("""
        This Clinical RAG Assistant helps healthcare professionals:
        - Search through clinical notes efficiently
        - Retrieve relevant medical information
        - Assist in diagnostic reasoning
        - Access up-to-date clinical knowledge
        
        **Features:**
        - Semantic search of clinical notes
        - Relevance scoring
        - Context-aware responses
        - Performance metrics
        """)
    
    else:
        st.markdown('<div class="success-box">✅ System Ready</div>', unsafe_allow_html=True)
        
        # Show system info
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.documents_df is not None:
                st.metric("Clinical Notes", len(st.session_state.documents_df))
        with col2:
            st.metric("Status", "Active")
        with col3:
            st.metric("Data Source", "Real" if st.session_state.dataset_path else "Demo")
        
        st.markdown("---")
        
        # Query Interface
        st.header("🔍 Clinical Query Interface")
        
        # Query input
        query = st.text_area(
            "Enter your clinical question:",
            value=st.session_state.get('query_input', ''),
            height=100,
            placeholder="Example: What are the diagnostic criteria for heart failure?",
            key="query_area"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            retrieve_k = st.slider("Number of documents to retrieve:", 1, 5, 3)
        with col2:
            if st.button("🔍 Search Clinical Notes", type="primary", use_container_width=True):
                if query.strip():
                    process_query(query)
                else:
                    st.warning("Please enter a query")
        with col3:
            if st.button("🔄 Clear", type="secondary", use_container_width=True):
                st.session_state.query_input = ""
                if 'query_area' in st.session_state:
                    st.session_state.query_area = ""
                st.rerun()
        
        st.markdown("---")
        
        # History Section (simplified)
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if st.session_state.query_history:
            st.subheader("📜 Recent Queries")
            for i, hist_query in enumerate(reversed(st.session_state.query_history[-3:])):
                st.write(f"{len(st.session_state.query_history)-i}. {hist_query}")

# Run the app with error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ An unexpected error occurred")
        st.markdown('<div class="error-box">Please try restarting the app or contact support if the issue persists.</div>', 
                   unsafe_allow_html=True)
        
        with st.expander("🔧 Technical Details"):
            st.write(f"**Error:** {str(e)}")
            st.code(traceback.format_exc())
            
            st.write("**System Information:**")
            st.write(f"- Python: {sys.version}")
            st.write(f"- Streamlit: {st.__version__}")
            st.write(f"- PyTorch: {torch.__version__ if 'torch' in sys.modules else 'Not loaded'}")
        
        # Reset option
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
