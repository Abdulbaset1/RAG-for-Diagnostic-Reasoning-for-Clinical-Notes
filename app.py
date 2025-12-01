import streamlit as st
import os
import torch
from custom_rag_pipeline import CustomRAGPipeline

# Page Config
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .stChatInput {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🏥 Clinical RAG Assistant")
st.markdown("### Powered by MIMIC-IV-Ext & Microsoft Phi-3")
st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset Path Input
    # Check for local zip file from GitHub repo
    zip_path = "mimic-iv-ext-direct-1.0.0.zip"
    extract_path = "mimic-iv-ext-direct-1.0.0"
    
    if os.path.exists(zip_path) and not os.path.exists(extract_path):
        with st.spinner("Unzipping dataset... This may take a while."):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.success("Dataset unzipped!")
            
    # Determine default path
    if os.path.exists(os.path.join(extract_path, "Finished")):
        default_path = os.path.join(extract_path, "Finished")
    else:
        default_path = "/kaggle/input/rag-dataset/mimic-iv-ext-direct-1.0.0/Finished"
    
    dataset_path = st.text_input("Dataset Path", value=default_path)
    
    st.info(f"Using Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if not torch.cuda.is_available():
        st.warning("⚠️ Running on CPU. Model inference will be slow and may crash on free tier hosting due to memory limits.")
    
    if st.button("Reload Pipeline"):
        st.cache_resource.clear()
        st.rerun()

# Initialize Pipeline (Cached)
@st.cache_resource
def get_pipeline(path):
    if not os.path.exists(path):
        return None
    
    with st.spinner("Initializing RAG Pipeline... (This may take a minute)"):
        try:
            pipeline = CustomRAGPipeline(path)
            return pipeline
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            return None

# Main Logic
if not os.path.exists(dataset_path):
    st.error(f"⚠️ Dataset not found at: {dataset_path}")
    st.warning("Please check the path in the sidebar.")
else:
    pipeline = get_pipeline(dataset_path)
    
    if pipeline:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📚 View Retrieved Sources"):
                        for i, doc in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1} ({doc['disease']})**")
                            st.caption(f"Relevance Score: {doc['score']:.4f}")
                            st.text(doc['content'][:300] + "...")
                            st.divider()

        # Chat Input
        if prompt := st.chat_input("Ask a clinical question (e.g., 'Symptoms of heart failure')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Run Pipeline
                        result = pipeline.run(prompt)
                        response = result['answer']
                        sources = result['retrieved_documents']
                        
                        # Display Answer
                        st.markdown(response)
                        
                        # Display Sources
                        with st.expander("📚 View Retrieved Sources"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1} ({doc['disease']})**")
                                st.caption(f"Relevance Score: {doc['score']:.4f}")
                                st.text(doc['content'][:300] + "...")
                                st.divider()
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
