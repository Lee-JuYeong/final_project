import pandas as pd
import numpy as np

# 카드와 사용자 관심 데이터를 기반으로 매칭 점수 계산
def calculate_match_score(card_merge, interest_data):
    user_card_scores = []  # 최종 결과를 저장할 리스트
    interest_data['user_id'] = interest_data['user_id'].astype(str)  # user_id의 데이터 타입을 문자열로 변환

    # 모든 사용자에 대해 반복
    for user_id in interest_data['user_id'].unique():
        # 특정 사용자의 관심 데이터 필터링
        user_interest_df = interest_data[interest_data['user_id'] == user_id]
        category_count = len(user_interest_df['category'].unique())  # 사용자가 관심 있는 카테고리 개수 계산

        # 카테고리 수에 따른 가중치 설정
        weight = 1 / category_count if category_count > 0 else 1

        # 각 카드에 대해 매칭 점수 계산
        for _, card_row in card_merge.iterrows():
            match_score = 0  # 초기 매칭 점수

            # 사용자 관심 데이터를 카드 데이터와 비교
            for _, interest_row in user_interest_df.iterrows():
                if interest_row['category'] in card_row['mainCtgId_list']:  # 카드의 주요 카테고리가 사용자의 관심 카테고리와 일치하는지 확인
                    match_score += (
                        interest_row['암묵적관심도'] * weight +  # 암묵적 관심도에 가중치 적용
                        interest_row['명시적관심도']  # 명시적 관심도는 그대로 사용
                    )

            # 결과를 리스트에 추가
            user_card_scores.append({
                "User_id": user_id,  # 사용자 ID
                "Card ID": card_row['cardId'],  # 카드 ID
                "Match Score": round(match_score, 3),  # 매칭 점수 (소수점 3자리)
                "Total Fee": card_row['Total Fee']  # 카드의 총 연회비
            })

    # 리스트를 데이터프레임으로 변환하여 반환
    return pd.DataFrame(user_card_scores)

# 특정 사용자에 대한 추천 카드 상위 1개를 반환
def recommend_top_cards(user_id, card_merge, interest_data, details_cosine_sim, category_cosine_sim):
    recommendations = []  # 추천 결과를 저장할 리스트

    # user_id를 문자열로 변환하여 interest_data의 데이터 타입과 일치시킴
    user_id = str(user_id)
    interest_data['user_id'] = interest_data['user_id'].astype(str)

    # 사용자 관심 데이터 필터링
    user_interest_data = interest_data[interest_data['user_id'] == user_id]

    # 필터링된 관심 데이터 출력 (디버깅용)
    print("Filtered Interest Data for user_id=12:")
    print(user_interest_data)

    # 사용자 관심 데이터가 없는 경우 에러 메시지를 포함한 빈 데이터프레임 반환
    if user_interest_data.empty:
        return pd.DataFrame({"error": [f"No interest data found for user_id: {user_id}"]})

    # 각 카드에 대해 매칭 점수를 계산
    for idx, card_row in card_merge.iterrows():
        match_score = 0  # 초기 매칭 점수

        # 사용자 관심 데이터를 카드 데이터와 비교
        for _, interest_row in user_interest_data.iterrows():
            if interest_row['category'] in card_row['mainCtgId_list']:  # 카드의 주요 카테고리가 사용자의 관심 카테고리와 일치하는지 확인
                match_score += (
                    interest_row['암묵적관심도'] * 1 / len(user_interest_data['category'].unique()) +  # 암묵적 관심도 가중치 적용
                    interest_row['명시적관심도']  # 명시적 관심도 추가
                )

        # 추천 결과를 리스트에 추가
        recommendations.append({
            "User ID": user_id,  # 사용자 ID
            "Card ID": card_row['cardId'],  # 카드 ID
            "Match Score": round(match_score, 3),  # 매칭 점수 (소수점 3자리)
            "Total Fee": card_row['Total Fee']  # 카드의 총 연회비
        })

    # 리스트를 데이터프레임으로 변환
    recommendations_df = pd.DataFrame(recommendations)

    # 추천 결과가 없는 경우 에러 메시지를 포함한 데이터프레임 반환
    if recommendations_df.empty:
        return pd.DataFrame({"error": ["No recommendations generated"]})

    # 매칭 점수 기준 내림차순, 총 연회비 기준 오름차순으로 정렬하여 상위 1개만 반환
    return recommendations_df.sort_values(by=['Match Score', 'Total Fee'], ascending=[False, True]).iloc[:1]
