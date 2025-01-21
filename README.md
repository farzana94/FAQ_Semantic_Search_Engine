# FAQ_Semantic_Search_Engine
Here’s a sample README.md file for your project to push on GitHub. You can customize it as needed.

Semantic Search Engine

This project implements a semantic search engine for Frequently Asked Questions (FAQs) using Django, Sentence-BERT (SBERT), and FAISS or Elasticsearch. The engine retrieves contextually similar results based on user queries and evaluates performance using metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Features
	•	Semantic similarity search using SBERT.
	•	Indexing and retrieval using FAISS or Elasticsearch.
	•	Evaluation of search results with MRR and NDCG.
	•	Web interface for querying and viewing results.
	•	JSON-based dataset for FAQs.
	•	Support for hard negative mining to improve model accuracy.



Getting Started

Prerequisites
	•	Python 3.11
	•	FAISS 
	•	Django 4.2+
	•	Sentence-BERT (sentence-transformers library)

Model was fine tuned using Amazon QA.

Setup
	1.	Clone the Repository:

git clone https://github.com/your-username/semantic-search-engine.git
cd semantic-search-engine


	2.	Set Up Virtual Environment:

python3.11 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


	3.	Install Dependencies:

pip install -r requirements.txt


	4.	Run Migrations:

python3 manage.py migrate


	5.	Download or Add the SBERT Model:
Place your fine-tuned SBERT model in the fine_tuned_sbert_model/ directory.
	6.	Start the Development Server:

python3 manage.py runserver


	7.	Access the Web Interface:
Open your browser and navigate to http://127.0.0.1:8000.

How to Use
	1.	Navigate to the search page and enter a query in the search bar.
	2.	View retrieved documents ranked by semantic similarity.
	3.	Evaluate search quality using metrics like MRR and NDCG.


Evaluation

Run evaluation metrics on labeled test queries in evaluation_data.txt.

Example Command:

from search_engine.services.evaluation_service import calculate_mrr, calculate_ndcg

queries = ["What is your return policy?", "How do I reset my password?"]
retrieved_docs = [["doc1", "doc3", "doc2"], ["doc5", "doc4", "doc6"]]
ground_truth = [{"doc1", "doc2"}, {"doc5"}]

mrr = calculate_mrr(ground_truth, retrieved_docs)
ndcg = calculate_ndcg(ground_truth, retrieved_docs)

print(f"MRR: {mrr}")
print(f"NDCG: {ndcg}")

Future Enhancements
	•	Add user authentication for personalized search history.
	•	Implement query auto-completion and suggestion.
	•	Integrate Elasticsearch for scalable indexing.
	•	Add advanced analytics for query performance.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
	•	SentenceTransformers for semantic embeddings.
	•	FAISS for efficient similarity search.
	•	Django for the web framework.

Feel free to replace your-username in the repository link with your GitHub username and add a LICENSE file if applicable.
