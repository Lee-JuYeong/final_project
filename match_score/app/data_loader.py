import pandas as pd
import re

def extract_fee(annualfee, keyword):
    if pd.isna(annualfee):
        return 0

    match = re.search(rf'{keyword}\s*(\d+)?만?(\d+)?천?(\d+)?백?원', annualfee)
    if not match:
        match = re.search(rf'{keyword}\s*(\d+)?천?(\d+)?백?원', annualfee)

    if match:
        total = 0
        if match.group(1):
            total += int(match.group(1)) * 10000
        if match.group(2):
            total += int(match.group(2)) * 1000
        if match.group(3):
            total += int(match.group(3)) * 100

        match_simple = re.search(rf'{keyword}\s*(\d+)천원', annualfee)
        if match_simple:
            return int(match_simple.group(1)) * 1000
        return total

    return 0

def initialize_data():
    # 데이터 로드
    card_data = pd.read_csv('./data/카드실적.csv')
    category = pd.read_csv('./data/CardCategory.csv')
    interest_data = pd.read_csv('./data/interest_score.csv')
    main_category = pd.read_csv('./data/Category.csv')
    card_info = pd.read_csv('./data/card_information.csv')

    # 'category' 열 생성
    if 'category' not in main_category.columns:
        main_category['category'] = main_category['categoryName']  # 'categoryName'을 'category'로 설정

    # 데이터 타입 강제 변환
    interest_data['category'] = interest_data['category'].astype(str)
    main_category['category'] = main_category['category'].astype(str)

    # 관심도 데이터 병합
    interest_data = interest_data.merge(
        main_category[['category', 'categoryId']],
        on='category', how='left'
    )
    interest_data['category'] = interest_data['category'].fillna(0).astype(int)

    # 연회비 처리
    card_data['Domestic Annual Fee'] = card_data['annualfee'].apply(lambda x: extract_fee(x, '국내')).fillna(0)
    card_data['Total Fee'] = card_data['Domestic Annual Fee']
    card_data_path = card_data.drop(['annualfee', 'pervSales'], axis=1)

    # 카테고리 데이터 병합
    category_data = category.groupby('cardId')['mainCtgId'].apply(
        lambda x: " ".join(map(str, x.unique()))
    ).reset_index()
    card_merge = card_data_path.merge(category_data, on='cardId', how='inner')

    return card_merge, interest_data, card_info
