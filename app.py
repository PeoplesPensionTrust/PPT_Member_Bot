import streamlit as st
import faiss
import fitz
import json
import os
import numpy as np
from openai import OpenAI
import openai
from bot_functions import ResponseHandler

# ---- STREAMLIT APP CONFIG ----
st.set_page_config(page_title="AI Chatbot", layout="wide")

st.title("PPT Conversational Engine")
st.write("Ask me anything about People's Pension Trust")

# ---- LOAD ASSETS ----
api_file_path = "api_key/api_key.txt"
embeddings_file_path = "generated_assets/embeddings.json"
index_file_path = "generated_assets/faiss_index.index"
contextual_documents_folder = "context_documents"


# ---- API KEY INPUT FROM FILE ----
api_key = open(api_file_path, "r").readline().strip()
openai.api_key = api_key


# commenter
# @st.cache_data  # Cache to avoid reloading on every interaction


def load_assets():
    """Load FAISS index, responses, and contextual documents."""
    # Load FAISS index
    index = faiss.read_index(index_file_path)
    
    # Load responses
    with open(embeddings_file_path, "r") as file:
        data = json.load(file)
        responses = data["responses"]

    # Extract text from PDF files
    pdf_texts = {}
    for pdf_file in os.listdir(contextual_documents_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(contextual_documents_folder, pdf_file)
            pdf_texts[pdf_file] = extract_text_from_pdf(pdf_path)
    
    # Convert extracted text into a list for processing
    context_docs = list(pdf_texts.values())

    return index, responses, context_docs

# ---- FUNCTION TO EXTRACT TEXT FROM PDF ----
def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Load FAISS index, responses, and context documents
index, responses, context_docs = load_assets()

# ---- INITIALIZE RESPONSE HANDLER ----
if api_key:
    response_handler = ResponseHandler(
        faiss_index=index, responses=responses, documents=context_docs, api_key=api_key
    )
else:
    response_handler = None

# ---- USER INPUT ----
if "query" not in st.session_state:
    st.session_state.query = ""

# --- FORM BOX --- 
with st.form("query_form"):
    query = st.text_input("How may I help you?:", value=st.session_state.query, key="query_input")
    submit_button = st.form_submit_button("Ask")  


# ---- RESPONSE HANDLING ----
if submit_button:
    if not api_key:
        st.warning("API Connection Error.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            best_response = response_handler.get_best_response(query)
        st.subheader("Assistant:")
        st.write(best_response)


        st.session_state.query = ""
        



       

st.markdown("---")
st.info("Type your question above and press 'Ask' to interact with the AI.")

