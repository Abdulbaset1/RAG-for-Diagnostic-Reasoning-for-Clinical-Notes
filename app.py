# app.py - Minimal test version
import streamlit as st
import sys

st.title("Clinical RAG Assistant - Debug Mode")
st.write(f"Python version: {sys.version}")
st.write("App is loading...")

try:
    import torch
    st.success(f"✓ PyTorch installed: {torch.__version__}")
except Exception as e:
    st.error(f"✗ PyTorch error: {e}")

try:
    from transformers import __version__ as transformers_version
    st.success(f"✓ Transformers installed: {transformers_version}")
except Exception as e:
    st.error(f"✗ Transformers error: {e}")

try:
    from sentence_transformers import __version__ as st_version
    st.success(f"✓ Sentence Transformers installed: {st_version}")
except Exception as e:
    st.error(f"✗ Sentence Transformers error: {e}")

# Test basic functionality
if st.button("Run Simple Test"):
    with st.spinner("Testing..."):
        try:
            # Test numpy
            import numpy as np
            arr = np.array([1, 2, 3])
            st.write(f"Numpy test: {arr.sum()}")
            
            # Test pandas
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            st.write(f"Pandas test shape: {df.shape}")
            
            st.success("All tests passed!")
        except Exception as e:
            st.error(f"Test failed: {e}")
