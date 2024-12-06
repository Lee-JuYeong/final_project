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
file_path = Config.DATA_FILE_PATH
card_data = pd.read_csv(file_path)
card_data["content"] = card_data.apply(
    lambda row: (
        f"Card Name: {row['cardName']}\n"
        f"Benefits: {', '.join(eval(row['Detail_summ_name'])) if pd.notna(row['Detail_summ_name']) else '혜택 정보 없음'}\n"
        f"Details: {' '.join(eval(row['Details_summary'])) if pd.notna(row['Details_summary']) else '상세 정보 없음'}"
    ),
    axis=1
)

# 벡터 스토어 생성
loader = DataFrameLoader(card_data, page_content_column="content")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
docsearch = FAISS.from_documents(loader.load(), embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["context", "card_name", "benefits"],
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
        card_row = card_data[card_data['cardId'] == card_id]
        if card_row.empty:
            return {"error": f"Card ID '{card_id}'에 대한 정보를 찾을 수 없습니다."}

        # 카드 이름 및 문맥 추출
        card_name = card_row.iloc[0]['cardName']
        context = card_row.iloc[0]['content']

        # Benefits 추출
        if "Benefits: " in context:
            benefits_start = context.find("Benefits: ")
            benefits_end = context.find("\nDetails: ")
            benefits = context[benefits_start + len("Benefits: "):benefits_end].strip()
        else:
            benefits = "혜택 정보 없음"

        # 프롬프트 생성
        prompt = prompt_template.format(context=context, card_name=card_name, benefits=benefits)

        # LLM 호출
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=110)
        response = llm.predict(prompt)

        # 모델 응답 검증
        if not response:
            raise ValueError("모델 응답이 비어 있습니다.")

        # 광고 문구를 두 문장으로 분리
        ad_sentences = response.strip().split("\n")
        ad_sentences = [sentence.replace("\\", "").strip() for sentence in ad_sentences if sentence.strip()]

        # 두 문장이 부족한 경우 처리
        while len(ad_sentences) < 2:
            ad_sentences.append("더 많은 혜택을 지금 바로 만나보세요!")

        # 문장이 끊기지 않도록 보정
        for i, sentence in enumerate(ad_sentences):
            if not sentence.endswith((".", "!", "?")):
                ad_sentences[i] += ""

        # 결과 반환
        return {
            "cardId": card_id,
            "cardName": card_name,
            "benefits": benefits,
            "adCopy": "\n".join(ad_sentences[:2])
        }

    except Exception as e:
        # 디버깅 정보 출력
        import traceback
        print(f"오류 발생: {e}")
        print(traceback.format_exc())
        return {"error": f"광고 문구 생성 실패: {e}"}
