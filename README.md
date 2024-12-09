# final_project

## 광고 카피라이팅

### 1. match score
- 관심도 점수 모델링을 통해 match score가 가장 높은 top1 카드를 선정
- 카테고리 코사인 유사도 점수와 상세설명 텍스트 벡터 코사인 유사도 점수를 통해 top1카드에 대한 2개의 유사 카드 선정

### 2. llm_ad
- langchain과 rag를 활용한 cardId별 광고 문구 생성

### 3. top1_ad
- userId를 검색했을 때 userId의 top1 카드를 뽑고 그 카드에 대한 광고문구 생성(langchain & RAG)
