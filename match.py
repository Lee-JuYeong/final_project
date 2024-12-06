from flask import Flask, request, Response
import json
from match_score.app.data_loader import initialize_data
from match_score.app.similarity import compute_cosine_similarity, compute_category_cosine_similarity
from match_score.app.recommendation import recommend_top_cards

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        # 사용자 ID 가져오기
        user_id = request.args.get('user_id', type=int)
        if user_id is None:
            return Response(
                json.dumps({"error": "user_id is required"}, ensure_ascii=False, indent=2),
                status=400,
                mimetype='application/json'
            )

        # 데이터 초기화
        card_merge, interest, card_info = initialize_data()

        # 상세설명 코사인 유사도 계산
        details_cosine_sim, similarity_scores = compute_cosine_similarity(card_info)

        # 카테고리 코사인 유사도 계산
        category_cosine_sim = compute_category_cosine_similarity(card_merge)

        # 추천 결과 생성
        recommended_card = recommend_top_cards(
            user_id=user_id,
            card_merge=card_merge,
            interest_data=interest,
            details_cosine_sim=details_cosine_sim,
            category_cosine_sim=category_cosine_sim
        )

        # 추천 결과 디버깅
        print("Recommended Card DataFrame:")
        print(recommended_card)

        if not recommended_card.empty:
            result = recommended_card.iloc[0].to_dict()
            print("First Recommended Card:")
            print(result)

            # 원하는 순서로 출력
            response = json.dumps(
                {
                    "User ID": int(result.get("User ID", 0)),
                    "Card ID": int(result.get("Card ID", 0)),
                    "Match Score": result.get("Match Score", 0),
                    "Total Fee": int(result.get("Total Fee", 0)),
                    "MostSimilarCard": int(result.get("MostSimilarCard", 0)) if result.get("MostSimilarCard") is not None else None,
                    "SimilarityScore": result.get("SimilarityScore", 0),
                    "MostSimilarCard_category": int(result.get("MostSimilarCard_category", 0)) if result.get("MostSimilarCard_category") is not None else None,
                    "CategorySimilarityScore": result.get("CategorySimilarityScore", 0)
                },
                ensure_ascii=False,
                indent=2
            )
        else:
            response = json.dumps(
                {"error": "No recommendations found"},
                ensure_ascii=False,
                indent=2
            )

        return Response(response, status=200, mimetype='application/json')

    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}, ensure_ascii=False, indent=2),
            status=500,
            mimetype='application/json'
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5060)
