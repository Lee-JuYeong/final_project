import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # OpenAI API 키 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 데이터 파일 경로
    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "./data/card_information.csv")  # 기본 경로 설정

    # 기본 확인
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
