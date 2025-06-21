import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
import asyncio
from datetime import datetime, timedelta
import os
import re
from collections import Counter
import platform
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

# 형태소 분석기 및 머신러닝 라이브러리
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

# 시각화 라이브러리
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# 설정 파일 임포트
import config

# --- 전역 객체 및 설정 초기화 ---
okt = Okt()

def setup_visualization():
    """시각화를 위한 폰트 및 스타일 설정"""
    try:
        if config.FONT_PATH and os.path.exists(config.FONT_PATH):
            plt.rcParams['font.family'] = fm.FontProperties(fname=config.FONT_PATH).get_name()
        elif platform.system() == 'Darwin': # MacOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else: # Windows/Linux
            # 기본으로 내장된 나눔고딕을 찾아서 설정
            font_list = [f.name for f in fm.fontManager.ttflist]
            if 'NanumGothic' in font_list:
                plt.rcParams['font.family'] = 'NanumGothic'
            else:
                print("나눔고딕 폰트가 설치되어 있지 않습니다. 폰트가 깨질 수 있습니다.")

        plt.rcParams['axes.unicode_minus'] = False
        print(f"폰트 설정 완료: {plt.rcParams['font.family']}")

    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 진행합니다.")


# --- 1. 데이터 크롤링 모듈 ---

async def _fetch(session: aiohttp.ClientSession, url: str) -> str:
    """주어진 URL에서 재시도 로직을 포함하여 HTML을 가져옴"""
    retry_options = ExponentialRetry(attempts=3)
    retry_client = RetryClient(client_session=session, retry_options=retry_options)
    async with retry_client.get(url, headers=config.HEADERS) as response:
        response.raise_for_status()
        return await response.text()

async def _get_article_content(session: aiohttp.ClientSession, link: str, semaphore: asyncio.Semaphore) -> str:
    """기사 본문 내용을 파싱"""
    async with semaphore:
        try:
            html = await _fetch(session, link)
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.find('div', {'id': 'newsct_article'})
            return content.text.strip().replace('\n', ' ') if content else ''
        except Exception as e:
            print(f"Error fetching content from {link}: {e}")
            return ''

async def _crawl_date(session: aiohttp.ClientSession, date: str, semaphore: asyncio.Semaphore) -> List[Dict]:
    """특정 날짜의 기사 목록과 본문을 크롤링"""
    list_url = config.BASE_URL.format(date=date)
    try:
        html = await _fetch(session, list_url)
        soup = BeautifulSoup(html, 'html.parser')

        articles = [
            {'title': element.text.strip(), 'link': element.get('href'), 'date': date}
            for element in soup.select('.sa_text_title')
        ]

        content_tasks = [_get_article_content(session, article['link'], semaphore) for article in articles]
        contents = await asyncio.gather(*content_tasks)

        for article, content in zip(articles, contents):
            article['content'] = content
        return [article for article in articles if article['content']]
    except Exception as e:
        print(f"Error crawling date {date}: {e}")
        return []

async def crawl_date_range(start_date: str, end_date: str) -> List[Dict]:
    """주어진 기간 동안의 모든 기사를 비동기적으로 크롤링"""
    conn = aiohttp.TCPConnector(limit=config.TCP_CONNECTOR_LIMIT, ttl_dns_cache=config.TCP_CONNECTOR_TTL_DNS_CACHE)
    async with aiohttp.ClientSession(connector=conn) as session:
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates = [(current_date + timedelta(days=i)).strftime("%Y%m%d") for i in range((end - current_date).days + 1)]

        semaphore = asyncio.Semaphore(config.SEMAPHORE_LIMIT)
        tasks = [_crawl_date(session, date, semaphore) for date in dates]

        all_articles = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(dates), desc=f"Crawling {start_date[:4]}", ncols=70):
            articles_per_day = await future
            all_articles.extend(articles_per_day)
    return all_articles

def run_crawling_multiprocess(start_date: str, end_date: str) -> List[Dict]:
    """멀티프로세싱을 사용하여 크롤러 실행"""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(crawl_date_range(start_date, end_date))

# --- 2. 데이터 분석 및 전처리 모듈 ---

def preprocess_text(text: str) -> List[str]:
    """텍스트 전처리: 정규식, 명사 추출, 불용어 제거"""
    replacements = {
        r'\b생성형\s*AI\b': '생성형AI', r'\b인공\s*지능\b': '인공지능',
        r'\b블록\s*체인\b': '블록체인', r'\b메타\s*버스\b': '메타버스',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r'\b디지털데일리\b|\b디지털 데일리\b|\b디지털 일리\b', '', text)
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)

    tokens = okt.nouns(text)
    processed_tokens = ['생성형AI' if token == '성형' else token for token in tokens]
    return [word for word in processed_tokens if len(word) > 1 and word not in config.STOPWORDS]

def analyze_data(df_dict: Dict[int, pd.DataFrame]) -> Tuple[Dict, Dict]:
    """연도별 데이터프레임을 받아 전체 분석을 수행"""
    print("텍스트 전처리 시작...")
    processed_texts = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for year, df in df_dict.items():
            processed_texts[year] = list(tqdm(executor.map(preprocess_text, df['content'],chunksize=100 ), total=len(df), desc=f"Preprocessing {year}"))

    print("TF-IDF 벡터화 및 토픽 모델링...")
    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES, tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf_matrix = {}
    tfidf_matrix[2023] = vectorizer.fit_transform(processed_texts[2023])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix[2024] = vectorizer.transform(processed_texts[2024])

    nmf_models, topics = {}, {}
    for year in config.YEARS_TO_ANALYZE:
        nmf = NMF(n_components=config.N_COMPONENTS, random_state=42, max_iter=200)
        W = nmf.fit_transform(tfidf_matrix[year])
        df_dict[year]['topic'] = W.argmax(axis=1)
        nmf_models[year] = nmf
        topic_words = []
        for topic_idx, topic_vec in enumerate(nmf.components_):
            top_features_ind = topic_vec.argsort()[:-config.TOP_N_TOPIC_WORDS - 1:-1]
            topic_words.append([feature_names[i] for i in top_features_ind])
        topics[year] = topic_words

    results = {'topics': topics, 'top_keywords': {}, 'top_bigrams': {}, 'ai_ratio': {}}
    for year in config.YEARS_TO_ANALYZE:
        tfidf_sum = tfidf_matrix[year].sum(axis=0).A1
        word_tfidf = dict(zip(feature_names, tfidf_sum))
        results['top_keywords'][year] = dict(sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)[:config.TOP_N_KEYWORDS])

        joined_texts = [' '.join(tokens) for tokens in processed_texts[year]]
        vec = CountVectorizer(ngram_range=(2, 2), min_df=3).fit(joined_texts)
        bow = vec.transform(joined_texts)
        sum_words = bow.sum(axis=0)
        words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)
        results['top_bigrams'][year] = dict(words_freq[:config.TOP_N_BIGRAMS])

        df_dict[year]['is_ai'] = df_dict[year]['content'].apply(lambda x: 1 if any(keyword in x for keyword in config.AI_KEYWORDS) else 0)
        results['ai_ratio'][year] = df_dict[year]['is_ai'].mean()

    return results, df_dict

# --- 3. 시각화 모듈 ---

def plot_top_keywords(keywords: Dict, year: int, output_dir: str):
    """상위 100개 키워드를 막대 그래프로 시각화"""
    plt.figure(figsize=(30, 20))
    bar_color = '#4285F4'
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 24, 'axes.labelsize': 20})

    plt.subplot(1, 2, 1)
    plt.barh(range(50), list(keywords.values())[:50], align='center', height=0.7, color=bar_color)
    plt.yticks(range(50), list(keywords.keys())[:50], fontsize=16)
    plt.title(f'{year}년 상위 1-50위 키워드', fontsize=28, pad=20)
    plt.xlabel('중요도 점수', fontsize=22)
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', labelsize=16)

    plt.subplot(1, 2, 2)
    plt.barh(range(50), list(keywords.values())[50:100], align='center', height=0.7, color=bar_color)
    plt.yticks(range(50), list(keywords.keys())[50:100], fontsize=16)
    plt.title(f'{year}년 상위 51-100위 키워드', fontsize=28, pad=20)
    plt.xlabel('중요도 점수', fontsize=22)
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', labelsize=16)

    plt.tight_layout(pad=4.0)
    output_path = os.path.join(output_dir, f"top_100_keywords_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"키워드 그래프 저장 완료: {output_path}")

def plot_top_bigrams(bigrams_by_year: Dict, output_dir: str):
    """연도별 상위 바이그램 비교 막대 그래프 생성"""
    plt.figure(figsize=(20, 10))

    bigrams_2023 = bigrams_by_year[2023]
    bigrams_2024 = bigrams_by_year[2024]

    plt.subplot(1, 2, 1)
    sns.barplot(x=list(bigrams_2023.values()), y=list(bigrams_2023.keys()), orient='h')
    plt.title('2023년 상위 20개 바이그램', fontsize=16)
    plt.xlabel('빈도', fontsize=12)

    plt.subplot(1, 2, 2)
    sns.barplot(x=list(bigrams_2024.values()), y=list(bigrams_2024.keys()), orient='h')
    plt.title('2024년 상위 20개 바이그램', fontsize=16)
    plt.xlabel('빈도', fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "top_20_bigrams_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"바이그램 그래프 저장 완료: {output_path}")

def plot_ai_article_ratio(ratios_by_year: Dict, output_dir: str):
    """AI 관련 기사 비율 변화를 시각화"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ratio_2023 = ratios_by_year[2023]
    ratio_2024 = ratios_by_year[2024]

    years = ['2023', '2024']
    ratios = [ratio_2023, ratio_2024]
    colors = ['skyblue', 'lightgreen']

    bars = ax.bar(years, ratios, color=colors, width=0.5)
    ax.set_ylim(0, max(ratios) * 1.2)
    ax.set_ylabel('AI 관련 기사 비율', fontsize=12)
    ax.set_title('연도별 AI 관련 기사 비율 변화', fontsize=16, pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2%}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ai_article_ratio.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AI 기사 비율 그래프 저장 완료: {output_path}")

def plot_topic_comparison(topics_by_year: Dict, df_by_year: Dict, output_dir: str):
    """연도별 토픽 분포 비교 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
    axes = [ax1, ax2]

    for idx, year in enumerate(config.YEARS_TO_ANALYZE):
        ax = axes[idx]
        df = df_by_year[year]
        topic_keywords = [' '.join(topic[:5]) for topic in topics_by_year[year]]
        topic_counts = df['topic'].value_counts(normalize=True).sort_index()
        y_pos = np.arange(len(topic_keywords))

        ax.barh(y_pos, topic_counts, align='center', color='skyblue', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_keywords, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel('문서 비율', fontsize=12)
        ax.set_title(f'{year}년 주요 토픽', fontsize=16, pad=15)

        for i, v in enumerate(topic_counts):
            ax.text(v + 0.005, i, f'Topic {i+1} ({v:.2%})', color='navy', fontweight='bold', va='center')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "topic_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"토픽 비교 그래프 저장 완료: {output_path}")

def create_wordcloud(word_freq: Dict, year: int, output_dir: str):
    """워드클라우드 생성"""
    if not os.path.exists(config.WORDCLOUD_MASK_PATH):
        print(f"워드클라우드 마스크 파일을 찾을 수 없습니다: {config.WORDCLOUD_MASK_PATH}")
        return

    cloud_mask = np.array(Image.open(config.WORDCLOUD_MASK_PATH))
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        mask=cloud_mask, font_path=config.FONT_PATH,
        prefer_horizontal=0.9, colormap='viridis'
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{year}년 주요 키워드', fontsize=20, pad=20)
    plt.tight_layout(pad=0)
    output_path = os.path.join(output_dir, f"wordcloud_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"워드클라우드 저장 완료: {output_path}")

def generate_visualizations(results: Dict, df_dict: Dict, output_dir: str):
    """모든 시각화 자료를 생성"""
    print("\n--- 시각화 자료 생성 시작 ---")

    # 연도별 키워드 그래프 및 워드클라우드 생성
    for year, keywords in results['top_keywords'].items():
        plot_top_keywords(keywords, year, output_dir)
        create_wordcloud(keywords, year, output_dir)

    # 비교 그래프 생성
    plot_top_bigrams(results['top_bigrams'], output_dir)
    plot_ai_article_ratio(results['ai_ratio'], output_dir)
    plot_topic_comparison(results['topics'], df_dict, output_dir)

    print("--- 모든 시각화 자료 생성 완료 ---")

# --- 4. 메인 실행 로직 ---

def get_half_year_range(year: int) -> Tuple[str, str]:
    """해당 연도의 상반기 날짜 범위를 반환"""
    start = datetime(year, config.CRAWL_START_MONTH, config.CRAWL_START_DAY)
    end = datetime(year, config.CRAWL_END_MONTH, config.CRAWL_END_DAY)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

def main():
    """메인 실행 함수"""
    os.makedirs(config.CRAWLED_DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    setup_visualization()

    dataframes = {}
    for year in config.YEARS_TO_ANALYZE:
        filepath = os.path.join(config.CRAWLED_DATA_DIR, f"crawled_data_{year}.parquet")
        if os.path.exists(filepath):
            print(f"{year}년 크롤링 데이터를 불러옵니다: {filepath}")
            dataframes[year] = pd.read_parquet(filepath)
        else:
            print(f"{year}년 데이터가 없어 크롤링을 시작합니다.")
            start_date, end_date = get_half_year_range(year)
            crawled_articles = run_crawling_multiprocess(start_date, end_date)
            df = pd.DataFrame(crawled_articles)
            df.to_parquet(filepath, index=False)
            dataframes[year] = df

    analysis_results, updated_dfs = analyze_data(dataframes)

    print("\n=== 최종 분석 결과 요약 ===")
    print("\n1. AI 관련 기사 비율:")
    for year in config.YEARS_TO_ANALYZE:
        print(f"  - {year}년: {analysis_results['ai_ratio'][year]:.2%}")

    print("\n2. 주요 토픽:")
    for year in config.YEARS_TO_ANALYZE:
        print(f"\n  - {year}년 주요 토픽:")
        for i, topic_words in enumerate(analysis_results['topics'][year], 1):
            print(f"    - 토픽 {i}: {', '.join(topic_words)}")

    generate_visualizations(analysis_results, updated_dfs, config.OUTPUT_DIR)

    print(f"\n모든 과정이 완료되었습니다. 결과물은 '{config.OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()