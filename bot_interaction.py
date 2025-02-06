from bot_functions import ResponseHandler
# from getpass import getpass
import faiss
from openai import OpenAI
import numpy as np
import json
import os  
import fitz
import openai



# -- LOAD INDEX & EMBEDDING FILES -- 
embeddings_file_path = "generated_assets/embeddings.json"
index_file_path = "generated_assets/faiss_index.index"
api_file_path = "api_key/api_key.txt"


# ---LOAD THE API KEY ---
api_key = open(api_file_path, "r").readline().strip()
openai.api_key = api_key

# # --- IF YOU'D RATHER INPUT THE API KEY ---
# ---- Don't forget to uncomment getpass / switch to input ---
# api_key = getpass("Enter your OpenAI API key: ").strip()
# openai.api_key = api_key


# Load FAISS Index and Embeddings
print("Loading assets...")
index = faiss.read_index(index_file_path)
with open(embeddings_file_path, "r") as file:
    data = json.load(file)
    responses = data["responses"]
print("Assets loaded successfully.")



# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Path to the folder containing PDF files
contextual_documents_folder = "context_documents"

# Extract text from all PDF files in the folder
pdf_texts = {}
for pdf_file in os.listdir(contextual_documents_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(contextual_documents_folder, pdf_file)
        pdf_texts[pdf_file] = extract_text_from_pdf(pdf_path)


# Convert the dictionary of context files to a list
context_docs = list(pdf_texts.values())





# Initialize the FAISS index, responses, documents, and API key
response_handler = ResponseHandler(faiss_index=index, responses=responses, documents=context_docs, api_key=api_key)

# Query loop
while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit", "stop", "no", "no thanks", "it's okay", "it's alright", "no thank you", "bye", "later", "goodbye", "ttyl"]:
        print("Have a lovely day. Conversation ended.")
        break
    best_response = response_handler.get_best_response(query)
    print(f"Assistant: {best_response}")