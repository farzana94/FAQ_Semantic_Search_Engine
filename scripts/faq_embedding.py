#3.

from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load the fine-tuned model
model = SentenceTransformer('fine_tuned_sbert_model')

# Load the FAQ dataset
with open('qa_dataset.json', 'r') as file:
    qa_pairs = json.load(file)

# Precompute embeddings for FAQ questions
faq_questions = [pair['Question'] for pair in qa_pairs]
faq_answers = [pair['Answer'] for pair in qa_pairs]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Save the embeddings and questions
np.save('faq_embeddings.npy', faq_embeddings.cpu().numpy())
with open('faq_questions.json', 'w') as f:
    json.dump(faq_questions, f)
with open('faq_answers.json', 'w') as f:
    json.dump(faq_answers, f)

print("Precomputed embeddings saved!")