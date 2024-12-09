import pandas as pd
import re

# 연회비를 추출하는 함수
def extract_fee(annualfee, keyword):
    # annualfee가 NaN인 경우 0을 반환
    if pd.isna(annualfee):
        return 0

    # 'keyword'와 숫자를 포함한 연회비 정보를 정규식으로 추출 (예: '국내 1만5천원')
    match = re.search(rf'{keyword}\s*(\d+)?만?(\d+)?천?(\d+)?백?원', annualfee)
    if not match:
        # 만 단위가 없는 경우 다시 검색 (예: '국내 5천원')
        match = re.search(rf'{keyword}\s*(\d+)?천?(\d+)?백?원', annualfee)

    # 매치된 경우
    if match:
        total = 0  # 최종 연회비 총합 초기화
        if match.group(1):  # '만' 단위 처리
            total += int(match.group(1)) * 10000
        if match.group(2):  # '천' 단위 처리
            total += int(match.group(2)) * 1000
        if match.group(3):  # '백' 단위 처리
            total += int(match.group(3)) * 100

        # 'keyword'와 함께 '천원' 단위만 매치되는 경우 처리
        match_simple = re.search(rf'{keyword}\s*(\d+)천원', annualfee)
        if match_simple:
            return int(match_simple.group(1)) * 1000  # 단순히 천원 단위 처리
        return total  # 계산된 총 연회비 반환

    return 0  # 매치되지 않는 경우 0 반환

# 데이터를 초기화하는 함수
def initialize_data():
    # 카드 데이터 로드 (카드 실적 관련 정보)
    card_data = pd.read_csv('./data/카드실적.csv')

    # 카테고리 데이터 로드 (카드별 카테고리 매핑 정보)
    category = pd.read_csv('./data/data/CardCategory.csv')

    # 관심도 데이터 로드 (카테고리별 관심도 점수)
    interest_data = pd.read_csv('./data/interest_score.csv')

    # 주요 카테고리 정보 로드
    main_category = pd.read_csv('./data/Category.csv')

    # 카드 상세 정보 로드
    card_info = pd.read_csv('./data/card_information.csv')

    # 'category' 열 생성 (main_category에 'categoryName'을 복사)
    if 'category' not in main_category.columns:
        main_category['category'] = main_category['categoryName']

    # interest_data의 'category' 열을 문자열 타입으로 강제 변환
    interest_data['category'] = interest_data['category'].astype(str)

    # main_category의 'category' 열을 문자열 타입으로 강제 변환
    main_category['category'] = main_category['category'].astype(str)

    # 관심도 데이터에 카테고리 ID를 병합
    interest_data = interest_data.merge(
        main_category[['category', 'categoryId']],  # 병합에 사용할 열 선택
        on='category', how='left'  # 'category'를 기준으로 병합 (좌측 기준)
    )

    # 관심도 데이터의 'category' 값이 NaN인 경우 0으로 채우고 정수형으로 변환
    interest_data['category'] = interest_data['category'].fillna(0).astype(int)

    # 카드 데이터의 'annualfee'에서 국내 연회비를 추출하고 'Domestic Annual Fee' 열 생성
    card_data['Domestic Annual Fee'] = card_data['annualfee'].apply(lambda x: extract_fee(x, '국내')).fillna(0)

    # 'Domestic Annual Fee'를 'Total Fee'로 복사 (추가 계산이 필요하면 변경 가능)
    card_data['Total Fee'] = card_data['Domestic Annual Fee']

    # 'annualfee'와 'pervSales' 열을 제거하여 card_data 정리
    card_data_path = card_data.drop(['annualfee', 'pervSales'], axis=1)

    # 카테고리 데이터 병합 (각 카드 ID에 대해 주요 카테고리를 문자열로 연결)
    category_data = category.groupby('cardId')['mainCtgId'].apply(
        lambda x: " ".join(map(str, x.unique()))  # 고유한 mainCtgId를 공백으로 구분해 연결
    ).reset_index()

    # 카드 데이터와 카테고리 데이터를 병합
    card_merge = card_data_path.merge(category_data, on='cardId', how='inner')

    # 병합된 카드 데이터, 관심도 데이터, 카드 상세 정보를 반환
    return card_merge, interest_data, card_info
