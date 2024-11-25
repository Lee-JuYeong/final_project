from flask import Flask, request, jsonify
import pandas as pd
from modules.processing import preprocess_data, calculate_similarity,clean_text
import re
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

@app.route("/")
def index():
    return {"message": "Welcome to the Text Processing API"}

@app.route("/upload", methods=["POST"])
def upload_file():
    """CSV 파일 업로드 및 요약 정보 반환"""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # 파일 읽기
        data = pd.read_csv(file)
        summary = {
            "rows": data.shape[0],
            "columns": data.shape[1],
            "column_names": data.columns.tolist(),
        }
        return jsonify({"status": "success", "summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process_data():
    """텍스트 데이터 전처리 및 유사도 계산"""
    try:
        json_data = request.json
        text_column = json_data.get("text_column")
        texts = json_data.get("texts")

        if not texts:
            return jsonify({"error": "Texts not provided"}), 400

        # 텍스트 전처리 및 유사도 계산
        cleaned_texts = [clean_text(text) for text in texts]
        similarity_matrix = calculate_similarity(cleaned_texts)

        return jsonify({"status": "success", "similarity_matrix": similarity_matrix}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)