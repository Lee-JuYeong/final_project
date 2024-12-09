from flask import Blueprint, request, jsonify
from modules.utils import generate_ad_copy_by_card_id
from modules.recommendation import recommend_top_cards
from modules.similarity import compute_cosine_similarity, compute_category_cosine_similarity
from modules.data_loader import initialize_data
import numpy as np

# 데이터 초기화
# initialize_data 함수를 사용하여 카드 데이터, 관심도 데이터, 상세 정보 데이터를 로드
card_merge, interest_data, card_info = initialize_data()

# 카드 상세 설명에 대한 코사인 유사도 계산
details_cosine_sim = compute_cosine_similarity(card_info)

# 카드 카테고리에 대한 코사인 유사도 계산
category_cosine_sim = compute_category_cosine_similarity(card_merge)

# Flask Blueprint 생성
# "ad_routes"는 이 모듈의 API 경로를 관리하기 위한 블루프린트
ad_routes = Blueprint("ad_routes", __name__)

# 광고 생성 API 엔드포인트
@ad_routes.route("/generate_top1_ad", methods=["GET"])
def generate_ad():
    # 사용자 ID를 쿼리 매개변수로부터 가져옴
    user_id = request.args.get("user_id")
    if not user_id:
        # user_id가 제공되지 않은 경우 400 에러 반환
        return jsonify({"error": "user_id is required"}), 400

    try:
        # 추천 카드 계산
        # 사용자의 관심도와 카드 데이터를 기반으로 추천 카드 목록 생성
        recommendations = recommend_top_cards(
            user_id=user_id,
            card_merge=card_merge,
            interest_data=interest_data,
            details_cosine_sim=details_cosine_sim,
            category_cosine_sim=category_cosine_sim
        )

        # 추천 결과가 비어 있거나 "Card ID" 열이 없는 경우 처리
        if recommendations.empty or "Card ID" not in recommendations.columns:
            return jsonify({"error": "No recommendations found or Card ID missing"}), 404

        # 추천된 카드 중 첫 번째 카드의 ID 가져오기
        top_card_id = recommendations.iloc[0]["Card ID"]

        # NumPy 데이터 타입을 Python 기본 타입으로 변환
        if isinstance(top_card_id, (np.integer, np.int64, np.int32)):
            top_card_id = int(top_card_id)  # 정수형 변환
        elif isinstance(top_card_id, (np.floating, np.float64, np.float32)):
            top_card_id = float(top_card_id)  # 부동소수점형 변환

        # 광고 문구 생성
        # 추천된 카드 ID를 사용하여 광고 문구 생성
        ad_copy = generate_ad_copy_by_card_id(top_card_id)

        # 광고 문구 생성 중 오류가 발생한 경우 처리
        if "error" in ad_copy:
            return jsonify(ad_copy), 500

        # 광고 문구를 두 문장으로 분리
        ad_sentences = ad_copy["adCopy"].split("\n")

        # 성공적인 응답으로 카드 ID와 광고 문구 반환
        return jsonify({
            "cardId": top_card_id,
            "adCopy1": ad_sentences[0],  # 첫 번째 문장
            "adCopy2": ad_sentences[1]   # 두 번째 문장
        })

    except Exception as e:
        # 예외 발생 시 에러 메시지와 스택 트레이스를 반환
        import traceback
        return jsonify({
            "error": f"Unexpected error occurred: {str(e)}",
            "stackTrace": traceback.format_exc()
        }), 500
