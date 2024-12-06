from flask import Blueprint, request, jsonify
from modules.utils import generate_ad_copy_by_card_id
from modules.recommendation import recommend_top_cards
from modules.similarity import compute_cosine_similarity, compute_category_cosine_similarity
from modules.data_loader import initialize_data
import numpy as np

# 데이터 초기화
card_merge, interest_data, card_info = initialize_data()
details_cosine_sim = compute_cosine_similarity(card_info)
category_cosine_sim = compute_category_cosine_similarity(card_merge)

ad_routes = Blueprint("ad_routes", __name__)

@ad_routes.route("/generate_top1_ad", methods=["GET"])
def generate_ad():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        # 추천 카드 계산
        recommendations = recommend_top_cards(
            user_id=user_id,
            card_merge=card_merge,
            interest_data=interest_data,
            details_cosine_sim=details_cosine_sim,
            category_cosine_sim=category_cosine_sim
        )

        if recommendations.empty or "Card ID" not in recommendations.columns:
            return jsonify({"error": "No recommendations found or Card ID missing"}), 404

        # 추천된 카드 ID 가져오기
        top_card_id = recommendations.iloc[0]["Card ID"]

        # JSON 직렬화 가능하도록 Python 기본 타입으로 변환
        if isinstance(top_card_id, (np.integer, np.int64, np.int32)):
            top_card_id = int(top_card_id)
        elif isinstance(top_card_id, (np.floating, np.float64, np.float32)):
            top_card_id = float(top_card_id)

        # 광고 문구 생성
        ad_copy = generate_ad_copy_by_card_id(top_card_id)
        if "error" in ad_copy:
            return jsonify(ad_copy), 500

        # 광고 문구 반환
        ad_sentences = ad_copy["adCopy"].split("\n")
        return jsonify({
            "cardId": top_card_id,
            "adCopy1": ad_sentences[0],
            "adCopy2": ad_sentences[1]
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Unexpected error occurred: {str(e)}",
            "stackTrace": traceback.format_exc()
        }), 500
