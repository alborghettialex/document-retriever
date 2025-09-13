import streamlit as st
import requests

st.title("Document Retriever")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file:
    st.write("Loading...")
    files = [("files", (uploaded_file.name, uploaded_file.getvalue()))]
    
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files=files
    )
    
    if response.status_code == 200:
        st.success("Completed!")
    else:
        st.error(f"Error: {response.text}")
