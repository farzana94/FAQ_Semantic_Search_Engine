# FAQ_Semantic_Search_Engine
Here’s a sample README.md file for your project to push on GitHub. You can customize it as needed.

Semantic Search Engine:

This project implements a semantic search engine for Frequently Asked Questions (FAQs) using Django, Sentence-BERT (SBERT), and FAISS or Elasticsearch. The engine retrieves contextually similar results based on user queries and evaluates performance using metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Features:

•	Semantic similarity search using SBERT.

•	Indexing and retrieval using FAISS or Elasticsearch.

•	Evaluation of search results with MRR and NDCG.

•	Web interface for querying and viewing results.

•	JSON-based dataset for FAQs.

•	Support for hard negative mining to improve model accuracy.



Getting Started:

Prerequisites:

•	Python 3.11

•	FAISS 

•	Django 4.2+

•	Sentence-BERT (sentence-transformers library)

Model was fine tuned using Amazon QA dataset.

Setup:
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
	Place your fine-tuned SBERT model in the working directory.
 
  6.	Start the Development Server:

	python3 manage.py runserver


  7.	Access the Web Interface:
        Open your browser and navigate to http://127.0.0.1:8000.

How to Use
	1.	Navigate to the search page and enter a query in the search bar.
	2.	View retrieved documents ranked by semantic similarity.
	3.	Evaluate search quality using metrics like MRR and NDCG.


  8.	Evaluation:

Run evaluation metrics on labeled test queries in evaluation_data.txt.

Future Enhancements:

•	Add user authentication for personalized search history.

•	Implement query auto-completion and suggestion.

•	Integrate Elasticsearch for scalable indexing.

•	Add advanced analytics for query performance.

Acknowledgments:

•	SentenceTransformers for semantic embeddings.

•	FAISS for efficient similarity search.

•	Django for the web framework.

