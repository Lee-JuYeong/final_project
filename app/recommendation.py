import pandas as pd
import numpy as np

def calculate_match_score(card_merge, interest_data):
    user_card_scores = []
    interest_data['user_id'] = interest_data['user_id'].astype(str)

    for user_id in interest_data['user_id'].unique():
        # 사용자 관심 데이터 필터링
        user_interest_df = interest_data[interest_data['user_id'] == user_id]
        category_count = len(user_interest_df['category'].unique())

        # 가중치
        weight = 1 / category_count if category_count > 0 else 1

        for _, card_row in card_merge.iterrows():
            # Match Score
            match_score = 0
            for _, interest_row in user_interest_df.iterrows():
                if interest_row['category'] in card_row['mainCtgId_list']:
                    match_score += (
                        interest_row['암묵적관심도'] * weight +
                        interest_row['명시적관심도']
                    )
   
            user_card_scores.append({
                "User_id": user_id,
                "Card ID": card_row['cardId'],
                "Match Score": round(match_score, 3),
                "Total Fee": card_row['Total Fee']
            })

    return pd.DataFrame(user_card_scores)
def recommend_top_cards(user_id, card_merge, interest_data, details_cosine_sim, category_cosine_sim):
    recommendations = []

    # 사용자 관심 데이터 필터링
    user_interest_data = interest_data[interest_data['user_id'] == user_id]
    if user_interest_data.empty:
        return pd.DataFrame({"error": f"No interest data found for user_id: {user_id}"})

    for idx, card_row in card_merge.iterrows():
        if idx >= details_cosine_sim.shape[0] or idx >= category_cosine_sim.shape[0]:
            continue  # 매트릭스 크기 초과 방지

        # 상세설명 코사인 유사도 계산
        details_cosine_row = details_cosine_sim[idx]
        details_cosine_row[idx] = -1  # 자기 자신 제외

        if details_cosine_row.max() <= 0:
            most_similar_idx = None
            similarity_score = 0
        else:
            most_similar_idx = np.argmax(details_cosine_row)
            similarity_score = details_cosine_row[most_similar_idx]

        most_similar_card = (
            card_merge.iloc[most_similar_idx]['cardId'] if most_similar_idx is not None and most_similar_idx < len(card_merge) else None
        )

        # 카테고리 코사인 유사도 계산
        category_cosine_row = category_cosine_sim[idx]
        category_cosine_row[idx] = -1  # 자기 자신 제외

        if category_cosine_row.max() <= 0:
            most_similar_idx_category = None
            category_similarity_score = 0
        else:
            most_similar_idx_category = np.argmax(category_cosine_row)
            category_similarity_score = category_cosine_row[most_similar_idx_category]

        most_similar_card_category = (
            card_merge.iloc[most_similar_idx_category]['cardId'] if most_similar_idx_category is not None and most_similar_idx_category < len(card_merge) else None
        )

        # Match Score 계산
        match_score = 0
        for _, interest_row in user_interest_data.iterrows():
            if interest_row['category'] in card_row['mainCtgId_list']:
                match_score += (
                    interest_row['암묵적관심도'] * 1 / len(user_interest_data['category'].unique()) + interest_row['명시적관심도']
                )

        recommendations.append({
            "User ID": user_id,
            "Card ID": card_row['cardId'],
            "Match Score": round(match_score, 3),
            "Total Fee": card_row['Total Fee'],
            "MostSimilarCard": most_similar_card,
            "SimilarityScore": round(similarity_score, 3),
            "MostSimilarCard_category": most_similar_card_category,
            "CategorySimilarityScore": round(category_similarity_score, 3),
        })

    recommendations_df = pd.DataFrame(recommendations)
    if not recommendations_df.empty:
        return recommendations_df.sort_values(by=['Match Score', 'Total Fee'], ascending=[False, True]).iloc[:1]
    return pd.DataFrame({"error": "No recommendations generated."})
