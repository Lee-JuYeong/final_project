from flask import Blueprint, request, jsonify
from .utils import generate_ad_copy_by_card_id

ad_routes = Blueprint("ad_routes", __name__)

@ad_routes.route("/generate_ad", methods=["GET"])
def generate_ad():
    # Query parameter로부터 cardId 가져오기
    card_id = request.args.get("cardId")

    if not card_id:
        return jsonify({"error": "Card ID가 제공되지 않았습니다."}), 400

    try:
        # 광고 생성 함수 호출
        result = generate_ad_copy_by_card_id(int(card_id))

        # 에러 처리
        if "error" in result:
            return jsonify(result), 400

        # 광고 문구 분리 및 검증
        ad_copy = result.get("adCopy")
        if not ad_copy:
            return jsonify({"error": "광고 문구 생성 실패: 응답이 비어 있습니다."}), 500

        # 광고 문구 나누기
        ad_sentences = ad_copy.split("\n")
        if len(ad_sentences) < 2:
            return jsonify({"error": "광고 문구가 충분하지 않습니다."}), 500

        # 결과 반환
        formatted_result = {
            "adCopy1": ad_sentences[0].strip("\""),
            "adCopy2": ad_sentences[1].strip("\""),
            "cardId": result.get("cardId")
        }
        return jsonify(formatted_result)

    except Exception as e:
        # 예외 처리 및 디버그 정보 반환
        import traceback
        return jsonify({
            "error": f"Unexpected error occurred: {str(e)}",
            "stackTrace": traceback.format_exc()
        }), 500
