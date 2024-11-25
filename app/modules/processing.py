import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text: str) -> str:
    """텍스트 정규화"""
    return re.sub(r'[^a-zA-Z0-9가-힣 ]', '', str(text))

def preprocess_data(data: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """DataFrame의 특정 열을 정규화"""
    data[text_column] = data[text_column].apply(clean_text)
    return data

def calculate_similarity(texts: list) -> list:
    """텍스트 간 코사인 유사도 계산"""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(matrix).tolist()
