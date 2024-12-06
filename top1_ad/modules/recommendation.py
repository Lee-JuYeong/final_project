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

    # user_id와 interest_data의 데이터 타입 일치
    user_id = str(user_id)
    interest_data['user_id'] = interest_data['user_id'].astype(str)

    # 사용자 관심 데이터 필터링
    user_interest_data = interest_data[interest_data['user_id'] == user_id]
    print("Filtered Interest Data for user_id=12:")
    print(user_interest_data)

    if user_interest_data.empty:
        return pd.DataFrame({"error": [f"No interest data found for user_id: {user_id}"]})

    # 나머지 로직 유지
    for idx, card_row in card_merge.iterrows():
        # Match Score 계산 등
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
            "Total Fee": card_row['Total Fee']
        })

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return pd.DataFrame({"error": ["No recommendations generated"]})

    return recommendations_df.sort_values(by=['Match Score', 'Total Fee'], ascending=[False, True]).iloc[:1]
