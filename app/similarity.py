from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

def compute_cosine_similarity(card_info):
    card_info['Details_summary_cleaned'] = card_info['Details_summary'].astype(str).str.replace('[\[\]\']', '', regex=True)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(card_info['Details_summary_cleaned'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    similarity_scores = []
    for idx, row in enumerate(cosine_sim):
        row[idx] = -1  # 자기 자신 제외
        if row.max() <= 0:
            most_similar_idx = None
            similarity_score = 0
        else:
            most_similar_idx = np.argmax(row)
            similarity_score = row[most_similar_idx]
        most_similar_card = (
            card_info.loc[most_similar_idx, 'cardId'] if most_similar_idx is not None and most_similar_idx < len(card_info) else None
        )
        similarity_scores.append({
            "cardId": card_info.loc[idx, 'cardId'],
            "MostSimilarCard": most_similar_card,
            "SimilarityScore": round(similarity_score, 3)
        })

    return cosine_sim, similarity_scores

def compute_category_cosine_similarity(card_merge):
    card_merge['mainCtgId_list'] = card_merge['mainCtgId'].apply(lambda x: list(map(int, str(x).split())))
    mlb = MultiLabelBinarizer()
    category_one_hot = mlb.fit_transform(card_merge['mainCtgId_list'])
    return cosine_similarity(category_one_hot)
