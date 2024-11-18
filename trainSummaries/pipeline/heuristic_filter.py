import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from embedding_model import get_embedding

def filter_summaries(summaries, example_embeddings):
    gap_threshold = -0.035
    with ThreadPoolExecutor() as executor:
        summary_embeddings = list(executor.map(get_embedding, summaries))

    similarities = [np.mean(cosine_similarity([emb], example_embeddings).flatten()) for emb in summary_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]

    selected_summaries = []
    for idx in sorted_indices[:20]:  # Consider the top 20 summaries
        if idx + 1  == len(sorted_indices):
            selected_summaries.append(summaries[idx])
            break
        if len(selected_summaries) <= 5 or similarities[idx] - similarities[idx + 1] < gap_threshold: # Take at least 5 articles 
            selected_summaries.append(summaries[idx])
        else:
            break 
            
    
    return "$END_OF_ARTICLE$ ".join(selected_summaries)
