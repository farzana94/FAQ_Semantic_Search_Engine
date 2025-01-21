# 5.
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the fine-tuned SBERT model
model = SentenceTransformer('fine_tuned_sbert_model')

# Load the FAQ dataset
# faq_questions = [...]  # Replace with a list of FAQ questions
# faq_answers = [...]    # Replace with a list of FAQ answers

with open('faq_questions.json', 'r') as file:
    faq_questions = json.load(file)

with open('faq_answers.json', 'r') as file:
    faq_answers = json.load(file)

# Generate FAQ embeddings
faq_embeddings = model.encode(faq_questions,convert_to_tensor=True)

# Convert embeddings to float32 (FAISS requires float32 format)
# faq_embeddings = np.array(faq_embeddings, dtype='float32')
# faq_embeddings = np.array(faq_embeddings)
faq_embeddings = faq_embeddings.cpu().numpy().astype('float32')
# Build a FAISS index
dimension = faq_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)

# Add embeddings to the index
index.add(faq_embeddings)

# Save the index for future use
faiss.write_index(index, "faiss_index.index")

# Save the questions and answers
# np.save('faq_questions.npy', faq_questions)
# np.save('faq_answers.npy', faq_answers)
# Convert the FAQ questions and answers into NumPy arrays
faq_questions_np = np.array(faq_questions, dtype=object)  # Object dtype to store strings
faq_answers_np = np.array(faq_answers, dtype=object)

# Save FAQ questions and answers as .npy files
np.save('faq_questions.npy', faq_questions_np)
np.save('faq_answers.npy', faq_answers_np)


print("FAISS index built and saved!")
#--------------------------------------
# Load the FAISS index and other data
index = faiss.read_index("faiss_index.index")
faq_questions = np.load('faq_questions.npy', allow_pickle=True)
faq_answers = np.load('faq_answers.npy', allow_pickle=True)

# Function to retrieve top-K FAQs using FAISS
def get_top_k_faqs(query, k=5):
    # Generate the query embedding
    query_embedding = model.encode(query).astype('float32')
    
    # Search the index for the top-K nearest neighbors
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Retrieve the top-K questions and answers
    results = [
        {
            "Question": faq_questions[idx],
            "Answer": faq_answers[idx],
            "Distance": distances[0][i]
        }
        for i, idx in enumerate(indices[0])
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
    print(f"  Distance: {result['Distance']:.4f}")