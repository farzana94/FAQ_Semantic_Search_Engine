#1.

import json
import pandas as pd

# Load the dataset
with open('qa_dataset.json', 'r') as file:
    qa_pairs = json.load(file)

# Prepare the data for fine-tuning (question-answer pairs)
train_data = [{'input_text': pair['Question'], 'label_text': pair['Answer']} for pair in qa_pairs]

# Convert to a pandas DataFrame for convenience
df = pd.DataFrame(train_data)
df.to_csv("faq_dataset.csv", index=False)  # Save as CSV for reference
print(df.head())