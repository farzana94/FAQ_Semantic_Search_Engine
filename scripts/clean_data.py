#to remove duplicates

# import json

# # Load the JSON file
# with open('/Users/WORK/Desktop/3.semantic_search_engine/Amazon_QA_extracted.json', 'r') as file:
#     data = json.load(file)

# # Function to remove duplicates and count them
# def remove_duplicates(qna_list):
#     unique_questions = set()
#     cleaned_qna = []
#     duplicates_count = 0

#     for qna in qna_list:
#         question = qna['Question']
#         if question not in unique_questions:
#             unique_questions.add(question)
#             cleaned_qna.append(qna)
#         else:
#             duplicates_count += 1

#     return cleaned_qna, duplicates_count

# # Iterate over categories and clean the data
# for category, items in data.items():
#     for item in items:
#         item['qna'], duplicates_count = remove_duplicates(item['qna'])
#         print(f"Category '{category}' Item ID '{item['id']}': Removed {duplicates_count} duplicates.")

# # Save the cleaned JSON to a file
# with open('cleaned_qna.json', 'w') as file:
#     json.dump(data, file, indent=4)

#to make dataset with only q and a

import json

# Load the JSON file
with open('cleaned_qna.json', 'r') as file:
    data = json.load(file)

# Extract Q&A pairs into a flat list
qa_pairs = []

for category, items in data.items():
    for item in items:
        for qna in item['qna']:
            qa_pairs.append({
                "Question": qna['Question'],
                "Answer": qna['Answer']
            })

# Save the Q&A dataset to a new JSON file
with open('qa_dataset.json', 'w') as file:
    json.dump(qa_pairs, file, indent=4)

# Optional: Print the first few pairs to verify
print("Sample Q&A Pairs:", qa_pairs[:5])