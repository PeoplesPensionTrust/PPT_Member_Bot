from getpass import getpass
from openai import OpenAI
import json
import numpy as np
import faiss
import os
import openai

# # Step 1: Enter OpenAI API key
# api_key = getpass("Enter your OpenAI API key: ").strip()


api_file_path = "api_key/api_key.txt"

# ---LOAD THE API KEY ---
api_key = open(api_file_path, "r").readline().strip()
openai.api_key = api_key



# Step 2: Load Training Data
file_path = "asset_generation_training_data/chatbot_training_data.txt"

# Read the file
with open (file_path, "r") as file:
    training_data = file.read()

# Optionally print parts of the training data
# Can be commented out based on your discretion
print(training_data[:500])



# Parse intents and responses from the TXT file
entries = training_data.split("# Intent")
intents = []
for entry in entries[1:]:
    parts = entry.split("Assistant:")
    if len(parts) == 2:
        intent_section, response_section = parts
        intent = intent_section.strip()
        response = response_section.strip()
        intents.append({"intent": intent, "response": response})

print(f"Processed {len(intents)} intents.")


# Step 3: Generate Embeddings
embeddings = []
responses = []

client = OpenAI(api_key=api_key)

for intent in intents:
    response = intent["response"]
    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=response
    )
    embedding = embedding_response.data[0].embedding  # Access the embedding
    embeddings.append(embedding)
    responses.append(response)

for response in range(len(responses)):
    responses[response] = responses[response].replace("[STAR]", "*")
    responses[response] = responses[response].replace("[HASH]", "#")


# Ensure the "generated_assets" folder exists
os.makedirs("generated_assets", exist_ok=True)

# Step 3: Save embeddings and responses to a JSON file in the "generated_assets" folder
embeddings_file_path = "generated_assets/embeddings.json"
with open(embeddings_file_path, "w") as file:
    json.dump({"responses": responses, "embeddings": embeddings}, file)
print(f"Embeddings generated and saved to {embeddings_file_path}.")

# Step 4: Create FAISS Index
embedding_matrix = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Save FAISS index to the "generated_assets" folder
faiss_index_file_path = "generated_assets/faiss_index.index"
faiss.write_index(index, faiss_index_file_path)
print(f"FAISS index created and saved to {faiss_index_file_path}.")



