import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# 저장 경로
save_dir = r"D:/drl-candlesticks-trader-main1/paper1/data/news"
os.makedirs(save_dir, exist_ok=True)
output_csv = os.path.join(save_dir, "bitcoin_news_5years.csv")

# NewsAPI 무료 버전 예시 (https://newsapi.org/)
# API 키 필요: https://newsapi.org/register
API_KEY = "YOUR_NEWSAPI_KEY"  # 여기에 본인 키 입력
QUERY = "bitcoin OR BTC"
BASE_URL = "https://newsapi.org/v2/everything"

# 5년치 기간 설정
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)

# NewsAPI는 한 번에 100개 기사, 1일 단위로 쿼리 권장
cur = start_date
all_articles = []

while cur <= end_date:
    from_str = cur.strftime("%Y-%m-%d")
    to_str = (cur + timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "q": QUERY,
        "from": from_str,
        "to": to_str,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    try:
        resp = requests.get(BASE_URL, params=params)
        data = resp.json()
        if data.get("status") == "ok" and data.get("articles"):
            for art in data["articles"]:
                all_articles.append({
                    "date": art["publishedAt"],
                    "title": art["title"],
                    "description": art["description"],
                    "url": art["url"]
                })
            print(f"{from_str} 기사 {len(data['articles'])}건 저장!")
        else:
            print(f"{from_str} 기사 없음 또는 에러: {data.get('message')}")
    except Exception as ex:
        print(f"{from_str} 기사 수집 실패: {ex}")
    cur += timedelta(days=1)

# 데이터프레임 저장
if all_articles:
    df = pd.DataFrame(all_articles)
    df.to_csv(output_csv, index=False)
    print(f"총 {len(df)}건 기사 저장 완료! → {output_csv}")
else:
    print("저장할 뉴스 데이터가 없습니다.")

# 참고: 무료 API는 일일/분당 쿼터 제한이 있으니, 실제 실행 시 주기적 저장/재시도 로직 추가 권장 