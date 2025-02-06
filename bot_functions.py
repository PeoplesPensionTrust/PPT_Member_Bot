import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from openai import OpenAI

class ResponseHandler:
    def __init__(self, faiss_index, responses, documents, api_key):
        """
        Initialize the ResponseHandler.

        Args:
        - faiss_index: FAISS index containing embeddings.
        - responses: List of predefined responses corresponding to the FAISS index.
        - documents: List of documents for RAG implementation.
        - api_key: OpenAI API key for generating responses.
        """
        self.index = faiss_index
        self.responses = responses
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(documents).toarray()
        self.api_key = api_key
    

    def search_documents(self, query, top_k=2):
        """Search documents using TF-IDF vectors."""
        query_vector = self.vectorizer.transform([query]).toarray()
        distances, indices = faiss.IndexFlatL2(self.document_vectors.shape[1]).search(query_vector, top_k)
        results = [(self.documents[i], distances[0][i]) for i in indices[0]]
        return results

    def use_embeddings_response(self, query_embedding):
        """Fetch the best response from embeddings."""
        distances, indices = self.index.search(np.array([query_embedding], dtype="float32"), k=1)
        best_match_index = indices[0][0]
        print(f"Selected from embeddings, Distance: {distances[0][0]}")
        return self.responses[best_match_index]

    def use_rag_response(self, query):
        """Generate a response using RAG."""
        search_results = self.search_documents(query)
        context = "\n".join([result[0] for result in search_results])

        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        client = OpenAI(api_key = self.api_key)
        # openai.api_key = self.api_key
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant of People's Pension Trust."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096
        )
        print("Selected from RAG implementation")
        return response.choices[0].message.content#.strip()

    def use_openai_response(self, query):
        """Fetch a response directly from OpenAI API."""
        client = OpenAI(api_key = self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant of People's Pension Trust."},
                {"role": "user", "content": query}
            ],
            max_tokens=4096
        )
        print("Selected from OpenAI API")
        return response.choices[0].message.content#.strip()

    def get_best_response(self, query):
        """Determine the response based on distances[0][0]."""
        query_embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = query_embedding_response.data[0].embedding 

        # Search the FAISS index
        distances, _ = self.index.search(np.array([query_embedding], dtype="float32"), k=1)
        distance = distances[0][0]

        # Decision logic based on Euclidean distance from FAISS
        # Closer the distance is to 0, the better the query matches to the embeddings
        if 0 <= distance <= 0.50:
            return self.use_embeddings_response(query_embedding)
        elif 0.51 <= distance <= 0.7:
            return self.use_rag_response(query)
        else:
            return self.use_openai_response(query)
