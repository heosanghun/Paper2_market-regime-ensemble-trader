import os
import pandas as pd
from datetime import datetime, timedelta
import feedparser
from newspaper import Article
import time
from concurrent.futures import ThreadPoolExecutor

# 저장 경로
save_dir = r"D:/drl-candlesticks-trader-main1/paper1/data/news"
os.makedirs(save_dir, exist_ok=True)
output_csv = os.path.join(save_dir, "crypto_news_5years_rss.csv")

# Coindesk RSS 피드
RSS_FEED = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"

# 5년치 기간 필터
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)

def fetch_article(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art
    except Exception as ex:
        print(f"기사 수집 실패: {url} - {ex}")
        return None

def process_entry(entry):
    try:
        # 날짜 파싱
        pub_date = None
        for key in ["published_parsed", "updated_parsed"]:
            if key in entry and entry[key]:
                pub_date = datetime(*entry[key][:6])
                break
        if not pub_date:
            return None
        if not (start_date <= pub_date <= end_date):
            return None
            
        date_str = pub_date.strftime("%Y-%m-%d")
        art = fetch_article(entry.link)
        if not art:
            return None
            
        return {
            "site": "Coindesk",
            "date": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
            "title": art.title or entry.title,
            "text": art.text[:500],
            "url": entry.link
        }
    except Exception as ex:
        print(f"기사 처리 실패: {entry.get('link', '')} - {ex}")
        return None

def fetch_news_data():
    all_articles = []
    collected_dates = set()
    
    print("RSS 피드에서 기사 목록 가져오는 중...")
    feed = feedparser.parse(RSS_FEED)
    
    # 병렬로 기사 처리
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_entry, entry) for entry in feed.entries]
        for future in futures:
            result = future.result()
            if result:
                date_str = result['date'].split()[0]
                if date_str not in collected_dates:
                    all_articles.append(result)
                    collected_dates.add(date_str)
                    print(f"[Coindesk] {date_str} {result['title']} 저장!")
    
    return all_articles

def main():
    while True:
        print("뉴스 데이터 수집 시작...")
        all_articles = fetch_news_data()
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df["date_only"] = pd.to_datetime(df["date"]).dt.date
            min_date = df["date_only"].min()
            max_date = df["date_only"].max()
            print(f"수집된 데이터 범위: {min_date} ~ {max_date} (총 {len(df)}건)")
            df.drop(columns=["date_only"], inplace=True)
            df.to_csv(output_csv, index=False)
            
            if min_date <= start_date.date() and max_date >= end_date.date():
                print(f"5년치 데이터 수집 완료! → {output_csv}")
                break
            else:
                print("아직 5년치 데이터가 충분하지 않습니다. 5초 후 재시도...")
                time.sleep(5)
        else:
            print("저장할 뉴스 데이터가 없습니다. 5초 후 재시도...")
            time.sleep(5)

if __name__ == "__main__":
    main() 