#4.

from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch

# Ensure that all tensors are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load precomputed data
# faq_embeddings = np.load('faq_embeddings.npy')
# Load precomputed FAQ embeddings (ensure they are on the same device)
faq_embeddings = torch.tensor(np.load('faq_embeddings.npy')).to(device)
with open('faq_questions.json', 'r') as f:
    faq_questions = json.load(f)
with open('faq_answers.json', 'r') as f:
    faq_answers = json.load(f)

model = SentenceTransformer('fine_tuned_sbert_model')

# Function to handle an incoming query
def get_top_k_faqs(query, k=5):
    # Generate the query embedding
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    
    # Compute cosine similarity
    similarities = cos_sim(query_embedding, faq_embeddings).squeeze(0).cpu().numpy()
    
    # Get top-K indices based on similarity scores
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Retrieve top-K questions and answers
    results = [
        {
            "Question": faq_questions[idx],
            "Answer": faq_answers[idx],
            "Similarity Score": similarities[idx]
        }
        for idx in top_k_indices
    ]
    return results

# Example query
query = "Can this TV be wall mounted?"
top_k_results = get_top_k_faqs(query, k=5)

# Print the top-K results
print("Top-K Similar FAQs:")
for i, result in enumerate(top_k_results, 1):
    print(f"Rank {i}:")
    print(f"  Question: {result['Question']}")
    print(f"  Answer: {result['Answer']}")
    print(f"  Similarity Score: {result['Similarity Score']:.4f}")