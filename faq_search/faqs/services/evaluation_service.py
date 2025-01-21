import numpy as np

queries = ["What should I do if it is not working Remote Control?", "What is the process to reset Samsung TV?",
"Where can I find information about Customer Support?", "Can I check Remote Control?", "How do I enable Wi-Fi?","How do I check Warranty?", "Is this tv of good quality for use as a computer monitor?"]
retrieved_docs = [
    [894, 507, 2946, 3129, 2407],  
    [2407, 1193, 2417, 2471, 865],   
    [733, 1841, 661, 3167, 1578],
    [2497, 1934, 2388, 1799, 975],
    [2407, 2935, 474, 2472, 939],
    [1971, 2764, 1670, 2744, 163],
    [3285, 9, 1954, 20, 1881]
]
ground_truth = [{},
    {2417},
    {},
    {2497, 1934, 2388, 1799},
    {2472, 939},  
    {3285, 9, 1954, 20, 1881},
    {3285, 9, 1954, 20, 1881}          
]

def calculate_mrr(retrieved_docs, ground_truth):
    reciprocal_ranks = []
    for docs, relevant_docs in zip(retrieved_docs, ground_truth):
        rank = next((i + 1 for i, doc in enumerate(docs) if doc in relevant_docs), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

mrr = calculate_mrr(retrieved_docs, ground_truth)
print("Mean Reciprocal Rank (MRR):", mrr)



def calculate_dcg(scores):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))

def calculate_ndcg(retrieved_docs, ground_truth):
    ndcg_scores = []
    for docs, relevant_docs in zip(retrieved_docs, ground_truth):
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in docs]
        dcg = calculate_dcg(relevance_scores)
        idcg = calculate_dcg(sorted(relevance_scores, reverse=True))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    return sum(ndcg_scores) / len(ndcg_scores)

ndcg = calculate_ndcg(retrieved_docs, ground_truth)
print("Normalized Discounted Cumulative Gain (NDCG):", ndcg)


