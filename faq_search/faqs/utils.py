# faqs/utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# Load the model and precomputed data
model = SentenceTransformer('fine_tuned_sbert_model')
faq_embeddings = np.load('faq_data/faq_embeddings.npy')
with open('faq_data/faq_questions.json', 'r') as f:
    faq_questions = json.load(f)
with open('faq_data/faq_answers.json', 'r') as f:
    faq_answers = json.load(f)

# Load FAISS index
index = faiss.read_index("faiss_index/faiss_index.index")

def get_top_k_faqs(query, k=5):
    query_embedding = model.encode(query).astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k)
    
    results = [
        {   
            "Id": int(idx),
            "Question": faq_questions[idx],
            "Answer": faq_answers[idx],
            "Distance": distances[0][i]
        }
        for i, idx in enumerate(indices[0])
    ]
    return results