import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from .config import Config
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

# 데이터 로드 및 전처리
file_path = Config.DATA_FILE_PATH  # 데이터 파일 경로 설정
card_data = pd.read_csv(file_path)  # CSV 파일에서 데이터 로드

# 카드 정보 콘텐츠 생성
# 각 카드의 정보를 'content' 컬럼에 텍스트 형식으로 변환
card_data["content"] = card_data.apply(
    lambda row: (
        f"Card Name: {row['cardName']}\n"  # 카드 이름
        f"Benefits: {', '.join(eval(row['Detail_summ_name'])) if pd.notna(row['Detail_summ_name']) else '혜택 정보 없음'}\n"  # 혜택 요약
        f"Details: {' '.join(eval(row['Details_summary'])) if pd.notna(row['Details_summary']) else '상세 정보 없음'}"  # 혜택 상세 내용
    ),
    axis=1
)

# 벡터 스토어 생성
# DataFrameLoader를 사용해 데이터를 로드하고, 벡터화된 검색을 위해 FAISS를 초기화
loader = DataFrameLoader(card_data, page_content_column="content")  # 'content' 컬럼을 기준으로 데이터 로드
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # OpenAI 임베딩 모델 사용
docsearch = FAISS.from_documents(loader.load(), embeddings)  # FAISS 벡터 스토어 생성
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})  # 가장 유사한 문서 1개 검색

# 프롬프트 템플릿 정의
# 광고 문구를 작성하기 위한 기본 틀 제공
prompt_template = PromptTemplate(
    input_variables=["context", "card_name", "benefits"],  # 템플릿에서 사용하는 변수 정의
    template="""\
    당신은 최고의 광고 카피라이터입니다. 아래 카드 정보를 바탕으로 감동적이고 창의적인 광고 문구를 작성하세요.
    - 광고 문구는 반드시 **두 개의 독립된 문장**으로 작성되어야 합니다.
    - 각 문장은 줄바꿈으로 구분됩니다.
    - 소비자가 매력을 느낄 수 있도록 혜택과 감성적인 메시지를 포함하세요.
    - 이모티콘을 활용하여 생동감을 더하세요.
    - 광고문구에 benefits의 내용을 넣는다면 최대 3개까지만 표현될 수 있게 작성하세요.
    - 문장은 잘리지 않고 완성된 형태여야 합니다.
    - 각 문장의 길이는 50~60자로 작성하세요.
    - 뻔하지 않은 문장으로 작성하세요.
    - cardId의 cardName을 꼭 언급해주세요.

    카드 정보:
    {context}

    카드 이름: {card_name}
    카드 혜택: {benefits}

    작성된 두 문장으로 구성된 광고 문구:
    """
)

# 광고 문구 생성 함수
def generate_ad_copy_by_card_id(card_id):
    try:
        # 카드 ID로 검색
        card_row = card_data[card_data['cardId'] == card_id]  # cardId로 해당 카드 정보 검색
        if card_row.empty:  # 카드 ID가 없는 경우 처리
            return {"error": f"Card ID '{card_id}'에 대한 정보를 찾을 수 없습니다."}

        # 카드 이름 및 문맥 추출
        card_name = card_row.iloc[0]['cardName']  # 카드 이름 추출
        context = card_row.iloc[0]['content']  # 카드 정보 텍스트 추출

        # 혜택 추출
        if "Benefits: " in context:
            # 'Benefits:' 섹션에서 혜택 내용만 추출
            benefits_start = context.find("Benefits: ")
            benefits_end = context.find("\nDetails: ")
            benefits = context[benefits_start + len("Benefits: "):benefits_end].strip()
        else:
            benefits = "혜택 정보 없음"  # 혜택 정보가 없는 경우 기본값 설정

        # 프롬프트 생성
        prompt = prompt_template.format(context=context, card_name=card_name, benefits=benefits)  # 프롬프트 텍스트 생성

        # LLM 호출
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=110)  # OpenAI 모델 초기화
        response = llm.predict(prompt)  # LLM 호출 및 응답 생성

        # 모델 응답 검증
        if not response:  # 모델이 빈 응답을 반환한 경우 처리
            raise ValueError("모델 응답이 비어 있습니다.")

        # 광고 문구를 두 문장으로 분리
        ad_sentences = response.strip().split("\n")  # 응답을 줄바꿈 기준으로 분리
        ad_sentences = [sentence.replace("\\", "").strip() for sentence in ad_sentences if sentence.strip()]  # 깨끗한 문장 리스트로 정리

        # 두 문장이 부족한 경우 처리
        while len(ad_sentences) < 2:
            ad_sentences.append("더 많은 혜택을 지금 바로 만나보세요!")  # 부족한 문장 채우기

        # 문장이 끊기지 않도록 보정
        for i, sentence in enumerate(ad_sentences):
            if not sentence.endswith((".", "!", "?")):  # 문장 끝이 문장부호로 종료되지 않으면 보정
                ad_sentences[i] += ""

        # 결과 반환
        return {
            "cardId": card_id,
            "cardName": card_name,
            "benefits": benefits,
            "adCopy": "\n".join(ad_sentences[:2])  # 첫 두 문장을 반환
        }

    except Exception as e:
        # 디버깅 정보 출력
        import traceback
        print(f"오류 발생: {e}")  # 예외 메시지 출력
        print(traceback.format_exc())  # 예외 스택 트레이스 출력
        return {"error": f"광고 문구 생성 실패: {e}"}  # 예외 메시지 반환
