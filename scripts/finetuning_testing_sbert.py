#2. 


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from prepare_faq_dataset import train_data

# Load pre-trained Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create training examples
train_examples = [
    InputExample(texts=[pair['input_text'], pair['label_text']], label=1.0)
    for pair in train_data
]

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function (Cosine Similarity)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path='fine_tuned_sbert_model'
)

#Testing model
# Load the fine-tuned model
fine_tuned_model = SentenceTransformer('fine_tuned_sbert_model')

# Example: Get embeddings for a new question
query = "Can this TV be wall mounted?"
embedding = fine_tuned_model.encode(query)
print("Embedding Shape:", embedding.shape)