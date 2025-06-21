import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import platform
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from PIL import Image
from aiohttp_retry import RetryClient, ExponentialRetry
import seaborn as sns

# HTTP 요청에 사용할 헤더 정보
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 분석에서 제외할 불용어 목록
stopwords = ['뉴스', '기자',
             '보도','대표', '활용', '제공', '지원', '관리', '시장',
             '분기', '대비', '기록', '지난해','디지털데일리','디지털','일리','디지털 데일리','디지털 일리','통해',
             '위해','대한','위한','의해','이번','기업', '사업', '고객', '산업', '글로벌', '국내', '한국', '서비스', '솔루션', '플랫폼', '기반', '구축',
             '운영', '진행', '계획', '예정', '도입', '출시', '발표', '확대', '전략', '협력', '파트너', '혁신', '효율',
             '지속', '추진', '전환', '적용', '통합', '올해', '지난', '이상', '최근', '시간', '매출', '달러', '억원',
             '비용', '투자', '실적', '영업', '기술', '시스템', '소프트웨어', '데이터', '정보', '개발', '사용', '사용자',
             '기능','업무']

# 폰트 설정
font_path = r'C:\Windows\Fonts\NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.sans-serif'] = ['NanumGothic']
plt.rcParams['axes.unicode_minus'] = False

def plot_top_bigrams(bigrams_2023, bigrams_2024, output_filename):

    """
    2023년과 2024년의 상위 20개 바이그램을 비교하는 막대 그래프를 생성합니다.

    :param bigrams_2023: 2023년 바이그램 딕셔너리
    :param bigrams_2024: 2024년 바이그램 딕셔너리
    :param output_filename: 저장할 이미지 파일 이름
    """

    plt.figure(figsize=(20, 10))

    # 2023년 데이터
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(bigrams_2023.values())[:20], y=list(bigrams_2023.keys())[:20], orient='h')
    plt.title('2023년 상위 20개 바이그램', fontsize=16)
    plt.xlabel('빈도', fontsize=12)

    # 2024년 데이터
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(bigrams_2024.values())[:20], y=list(bigrams_2024.keys())[:20], orient='h')
    plt.title('2024년 상위 20개 바이그램', fontsize=16)
    plt.xlabel('빈도', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"바이그램 분석 결과가 '{output_filename}'로 저장되었습니다.")

async def fetch_with_retry(session, url, semaphore):

    """
    주어진 URL에서 데이터를 비동기적으로 가져오며, 실패 시 재시도합니다.

    :param session: aiohttp 세션 객체
    :param url: 가져올 데이터의 URL
    :param semaphore: 동시 요청 수를 제한하는 세마포어
    :return: 가져온 HTML 내용
    """

    async with semaphore:
        retry_options = ExponentialRetry(attempts=3)
        retry_client = RetryClient(client_session=session, retry_options=retry_options)
        async with retry_client.get(url, headers=headers) as response:
            return await response.text()

async def get_article_content(session, link, semaphore):
    """
    주어진 링크에서 기사 내용을 가져옵니다.

    :param session: aiohttp 세션 객체
    :param link: 기사 링크
    :param semaphore: 동시 요청 수를 제한하는 세마포어
    :return: 기사 내용 텍스트
    """

    try:
        html = await fetch_with_retry(session, link, semaphore)
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', {'id': 'newsct_article'})
        content_text = content.text.strip().replace('\n', ' ') if content else ''
        return content_text
    except Exception as e:
        print(f"Error fetching {link}: {e}")
        return ''

async def get_articles(session, date, semaphore):
    url = f'https://news.naver.com/breakingnews/section/105/283?date={date}'
    html = await fetch_with_retry(session, url, semaphore)
    soup = BeautifulSoup(html, 'html.parser')

    articles = []
    for i, element in enumerate(soup.select('.sa_text_title'), 1):
        title = element.text.strip()
        link = element.get('href')
        articles.append({'number': i, 'title': title, 'link': link, 'date': date})

    return articles

async def crawl_date(session, date, semaphore):
    articles = await get_articles(session, date, semaphore)
    tasks = [get_article_content(session, article['link'], semaphore) for article in articles]
    contents = await asyncio.gather(*tasks)
    for article, content in zip(articles, contents):
        article['content'] = content
    return articles

async def crawl_date_range(start_date, end_date):
    conn = aiohttp.TCPConnector(limit=300, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates = [(current_date + timedelta(days=i)).strftime("%Y%m%d")
                 for i in range((end - current_date).days + 1)]

        semaphore = asyncio.Semaphore(100)
        tasks = [crawl_date(session, date, semaphore) for date in dates]
        all_articles = []

        for future in tqdm(asyncio.as_completed(tasks), total=len(dates), desc="Crawling Progress", ncols=70):
            articles = await future
            all_articles.extend(articles)

    return all_articles

def crawl_with_multiprocessing(start_date, end_date):
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(crawl_date_range(start_date, end_date))

def get_half_year_range(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 6, 30)
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")

okt = Okt()


def preprocess(text):

    """
    텍스트를 전처리합니다.

    1. 특정 단어 결합 (예: '생성형 AI' -> '생성형AI')
    2. 불필요한 텍스트 제거
    3. 특수문자 제거
    4. 명사 추출
    5. 특정 단어 처리 및 불용어 제거

    :param text: 전처리할 원본 텍스트
    :return : 전처리된 단어 리스트
    """


    text = re.sub(r'\b생성형\s*AI\b', '생성형AI', text, flags=re.IGNORECASE)
    text = re.sub(r'\b인공\s*지능\b', '인공지능', text, flags=re.IGNORECASE)
    text = re.sub(r'\b블록\s*체인\b', '블록체인', text, flags=re.IGNORECASE)
    text = re.sub(r'\b메타\s*버스\b', '메타버스', text, flags=re.IGNORECASE)



    text = re.sub(r'\b디지털데일리\b', '', text)
    text = re.sub(r'\b디지털 데일리\b', '', text)
    text = re.sub(r'\b디지털 일리\b', '', text)



    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)



    tokens = okt.nouns(text)




    processed_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '성형':
            processed_tokens.append('생성형AI')
        elif i < len(tokens) - 1:
            if tokens[i] == '블록' and tokens[i+1] == '체인':
                processed_tokens.append('블록체인')
                i += 1
            elif tokens[i] == '메타' and tokens[i+1] == '버스':
                processed_tokens.append('메타버스')
                i += 1
            else:
                processed_tokens.append(tokens[i])
        else:
            processed_tokens.append(tokens[i])
        i += 1

    return [word for word in processed_tokens
            if len(word) > 1 and word not in stopwords
            and word not in ['디지털', '일리', '데일리']]


def get_top_words(tfidf_matrix, feature_names, top_n=100):
    tfidf_sum = tfidf_matrix.sum(axis=0).A1
    word_tfidf = {word: tfidf_sum[i] for i, word in enumerate(feature_names)}
    return dict(sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n])

def get_top_ngrams(texts, n=2, top_k=20):


    texts = [re.sub(r'\b생성형\s*AI\b', '생성형AI', text, flags=re.IGNORECASE) for text in texts]
    texts = [re.sub(r'\b블록\s*체인\b', '블록체인', text, flags=re.IGNORECASE) for text in texts]
    texts = [re.sub(r'\b메타\s*버스\b', '메타버스', text, flags=re.IGNORECASE) for text in texts]

    vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r'\b\w+\b', min_df=1)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    total_count = X.sum(axis=0).A1
    word_freq = dict(zip(words, total_count))


    word_freq = {k: v for k, v in word_freq.items() if k not in ['디지털 일리', '디지털 데일리']}


    word_freq = {k.replace('성형', '생성형AI')
                 .replace('블록 체인', '블록체인')
                 .replace('메타 버스', '메타버스'): v
                 for k, v in word_freq.items()
                 if '성형' not in k or '생성형AI' in k}

    return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k])

def get_topics(model, feature_names, top_n=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-top_n - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind if len(feature_names[i]) > 1]
        topics.append(top_features)
    return topics

def plot_top_100_keywords(top_keywords, year, output_filename):
    plt.figure(figsize=(30, 20))

    # 단일 색상 설정
    bar_color = '#4285F4'

    # 글꼴 설정
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20

    # 왼쪽 부분 (1-50위)
    plt.subplot(1, 2, 1)
    bars = plt.barh(range(50), list(top_keywords.values())[:50], align='center', height=0.7, color=bar_color)
    plt.yticks(range(50), list(top_keywords.keys())[:50], fontsize=16)
    plt.title(f'{year}년 상위 1-50위 키워드', fontsize=28, pad=20)
    plt.xlabel('중요도 점수', fontsize=22)
    plt.gca().invert_yaxis()


    plt.tick_params(axis='x', labelsize=16)

    # 오른쪽 부분 (51-100위)
    plt.subplot(1, 2, 2)
    bars = plt.barh(range(50), list(top_keywords.values())[50:100], align='center', height=0.7, color=bar_color)
    plt.yticks(range(50), list(top_keywords.keys())[50:100], fontsize=16)
    plt.title(f'{year}년 상위 51-100위 키워드', fontsize=28, pad=20)
    plt.xlabel('중요도 점수', fontsize=22)
    plt.gca().invert_yaxis()


    plt.tick_params(axis='x', labelsize=16)

    plt.tight_layout(pad=4.0)


    plt.figtext(0.5, 0.02, "막대가 길수록 해당 키워드가 더 중요함을 나타냅니다.",
                ha="center", fontsize=20, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{year}년 상위 100개 키워드 이미지가 '{output_filename}'로 저장되었습니다.")


# AI 관련 키워드 정의
ai_keywords = set(['인공지능', 'AI', '머신러닝', '딥러닝', '신경망',
                   '자연어처리', 'NLP', '컴퓨터비전','GPT','제미나이','생성형AI','gemini'])

def classify_article(text):

    words = set(text.split())
    return 1 if words.intersection(ai_keywords) else 0

def filter_tokens(tokens):
    return [token for token in tokens if token not in ['디지털', '일리', '데일리'] and
            '디지털' not in token and '일리' not in token and '데일리' not in token]

def postprocess_results(results):
    for year in ['2023', '2024']:
        results['top_keywords'][year] = {k: v for k, v in results['top_keywords'][year].items() if k != '성형'}
        results['top_bigrams'][year] = {k: v for k, v in results['top_bigrams'][year].items() if '성형' not in k}
        results['topics'][year] = [[word if word != '성형' else '생성형AI' for word in topic] for topic in results['topics'][year]]
    return results

def plot_ai_article_ratio(ratio_2023, ratio_2024, output_filename):
    fig, ax = plt.subplots(figsize=(23, 8))

    center_2023 = (0.25, 0.5)
    center_2024 = (0.75, 0.5)
    radius_outer = 0.2

    def draw_circles(center, ratio, year):
        outer_circle = plt.Circle(center, radius_outer, fill=False, color='blue')
        inner_circle = plt.Circle(center, radius_outer * ratio, fill=True, color='skyblue', alpha=0.6)
        ax.add_artist(outer_circle)
        ax.add_artist(inner_circle)
        ax.text(center[0], center[1] + radius_outer + 0.05, f'{year}년', ha='center', va='bottom', fontsize=14)
        ax.text(center[0], center[1], f'{ratio:.2%}', ha='center', va='center', fontsize=14)

    draw_circles(center_2023, ratio_2023, 2023)
    draw_circles(center_2024, ratio_2024, 2024)

    # 원 상단을 연결하는 선
    ax.plot([center_2023[0], center_2024[0]],
            [center_2023[1] + radius_outer, center_2024[1] + radius_outer],
            color='red', linestyle='--')

    # 원 하단을 연결하는 선
    ax.plot([center_2023[0], center_2024[0]],
            [center_2023[1] - radius_outer, center_2024[1] - radius_outer],
            color='red', linestyle='--')

    # 내부 원 상단을 연결하는 선
    ax.plot([center_2023[0], center_2024[0]],
            [center_2023[1] + radius_outer * ratio_2023, center_2024[1] + radius_outer * ratio_2024],
            color='green', linestyle='--')

    # 내부 원 하단을 연결하는 선
    ax.plot([center_2023[0], center_2024[0]],
            [center_2023[1] - radius_outer * ratio_2023, center_2024[1] - radius_outer * ratio_2024],
            color='green', linestyle='--')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    plt.title('AI 관련 기사 비율 변화', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

import pandas as pd

def plot_topic_comparison(topics, df_2023, df_2024):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for idx, (year, ax, df) in enumerate(zip(['2023', '2024'], [ax1, ax2], [df_2023, df_2024])):
        topic_keywords = [' '.join(topic[:5]) for topic in topics[year]]
        topic_counts = df['topic'].value_counts(normalize=True).sort_index()
        y_pos = range(len(topic_keywords))

        ax.barh(y_pos, topic_counts, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_keywords)
        ax.invert_yaxis()
        ax.set_xlabel('문서 비율')
        ax.set_title(f'{year} Top 3 Topics')


        for i, v in enumerate(topic_counts):
            ax.text(v, i, f'Topic {i+1} ({v:.2%})', color='navy', fontweight='bold', va='center')

    plt.tight_layout()
    plt.savefig('topic_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("토픽 비교 시각화가 'topic_comparison.png'로 저장되었습니다.")

def analyze_and_compare(df_2023, df_2024):
    try:
        print("텍스트 전처리 중...")
        with ProcessPoolExecutor() as executor:
            df_2023['processed'] = list(tqdm(executor.map(preprocess, df_2023['content']), total=len(df_2023), desc="2023 전처리"))
            df_2024['processed'] = list(tqdm(executor.map(preprocess, df_2024['content']), total=len(df_2024), desc="2024 전처리"))

        print("추가 필터링 중...")
        df_2023['processed'] = df_2023['processed'].apply(filter_tokens)
        df_2024['processed'] = df_2024['processed'].apply(filter_tokens)

        # AI 관련 기사 분류
        df_2023['is_ai'] = df_2023['content'].apply(classify_article)
        df_2024['is_ai'] = df_2024['content'].apply(classify_article)

        print("\nTF-IDF 계산 중...")
        vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=False,
            max_features=1000,
            stop_words=None,
            token_pattern=r'\S+'
        )
        tfidf_2023 = vectorizer.fit_transform(df_2023['processed'].apply(lambda x: ' '.join(x)))
        tfidf_2024 = vectorizer.transform(df_2024['processed'].apply(lambda x: ' '.join(x)))

        print("\n토픽 모델링 중...")
        nmf_2023 = NMF(n_components=3, random_state=42, max_iter=200)
        nmf_2024 = NMF(n_components=3, random_state=42, max_iter=200)

        nmf_2023.fit(tfidf_2023)
        nmf_2024.fit(tfidf_2024)




        df_2023['topic'] = nmf_2023.transform(tfidf_2023).argmax(axis=1)
        df_2024['topic'] = nmf_2024.transform(tfidf_2024).argmax(axis=1)


        results = {}
        results['article_count'] = {
            '2023': len(df_2023),
            '2024': len(df_2024)
        }

        print("\n키워드 추출 중...")
        feature_names = vectorizer.get_feature_names_out()
        results['top_keywords'] = {
            '2023': get_top_words(tfidf_2023, feature_names),
            '2024': get_top_words(tfidf_2024, feature_names)
        }

        print("\n바이그램 추출 중...")
        results['top_bigrams'] = {
            '2023': get_top_ngrams(df_2023['processed'].apply(' '.join)),
            '2024': get_top_ngrams(df_2024['processed'].apply(' '.join))
        }


        results['topics'] = {
            '2023': get_topics(nmf_2023, feature_names),
            '2024': get_topics(nmf_2024, feature_names)
        }


        results = postprocess_results(results)

        print("\n=== 비교 분석 결과 ===")
        print("\n1. 기사 수 비교:")
        print(f"2023년 상반기: {results['article_count']['2023']}개")
        print(f"2024년 상반기: {results['article_count']['2024']}개")

        print("\n2. AI 관련 기사 비율:")
        print(f"2023년 상반기: {df_2023['is_ai'].mean():.2%}")
        print(f"2024년 상반기: {df_2024['is_ai'].mean():.2%}")
        plot_ai_article_ratio(df_2023['is_ai'].mean(), df_2024['is_ai'].mean(), 'ai_article_ratio.png')

        print("\n3. 상위 100개 키워드:")
        print("2023년 상반기 상위 100개 키워드:")
        keywords_2023 = list(results['top_keywords']['2023'].keys())
        for i in range(0, 100, 10):
            print(', '.join(keywords_2023[i:i+10]))

        print("\n2024년 상반기 상위 100개 키워드:")
        keywords_2024 = list(results['top_keywords']['2024'].keys())
        for i in range(0, 100, 10):
            print(', '.join(keywords_2024[i:i+10]))

        print("\n전체 100개 키워드의 시각화는 'top_100_keywords_2023.png'와 'top_100_keywords_2024.png' 파일에서 확인할 수 있습니다.")
        print("\n4. 상위 20개 바이그램 비교:")
        print("2023년 상반기:", ', '.join(results['top_bigrams']['2023'].keys()))
        print("2024년 상반기:", ', '.join(results['top_bigrams']['2024'].keys()))
        plot_top_bigrams(results['top_bigrams']['2023'], results['top_bigrams']['2024'], 'top_20_bigrams_comparison.png')

        print("\n5. 주요 토픽 비교:")
        for year in ['2023', '2024']:
            print(f"\n{year}년 상반기 주요 토픽:")
            for i, topic in enumerate(results['topics'][year], 1):
                print(f"  토픽 {i}: {', '.join(topic[:5])}")


        plot_top_100_keywords(results['top_keywords']['2023'], '2023', 'top_100_keywords_2023.png')
        plot_top_100_keywords(results['top_keywords']['2024'], '2024', 'top_100_keywords_2024.png')
        plot_topic_comparison(results['topics'], df_2023, df_2024)


        def create_wordcloud(word_freq, year, output_filename, mask_path):
            cloud_mask = np.array(Image.open(mask_path))

            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  mask=cloud_mask,
                                  contour_color='steelblue',
                                  font_path=font_path,
                                  min_font_size=5,
                                  max_font_size=100,
                                  prefer_horizontal=0.7).generate_from_frequencies(word_freq)

            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{year}년 주요 키워드', fontsize=20, pad=20)
            plt.tight_layout(pad=0)
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()

        # 2023년 워드클라우드 생성
        create_wordcloud(results['top_keywords']['2023'], '2023', 'wordcloud_2023.png','images/cloudshape.png')

        # 2024년 워드클라우드 생성
        create_wordcloud(results['top_keywords']['2024'], '2024', 'wordcloud_2024.png','images/cloudshape.png')

        print("워드클라우드가 생성되었습니다. 'wordcloud_2023.png'와 'wordcloud_2024.png' 파일을 확인해주세요.")

        print("\n분석이 완료되었습니다. 결과 이미지가 저장되었습니다.")
    except Exception as e:
        print(f"분석 중 오류가 발생했습니다: {e}")
        raise

def safe_save_excel(df, filepath):
    tries = 0
    while tries < 3:
        try:
            df.to_excel(filepath, index=False)
            print(f"파일 저장 성공: {filepath}")
            return
        except PermissionError:
            print(f"파일 {filepath}에 접근할 수 없습니다. 파일이 다른 프로그램에서 열려있는지 확인해주세요.")
            input("파일을 닫은 후 엔터를 눌러 다시 시도하세요...")
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            input("문제를 해결한 후 엔터를 눌러 다시 시도하세요...")
        tries += 1

    print(f"파일 {filepath} 저장 실패. 수동으로 데이터를 저장해주세요.")

def save_crawled_data(df, year):
    filename = f'crawled_data_{year}.parquet'
    df.to_parquet(filename, compression='snappy')
    print(f"{year}년 크롤링 데이터가 {filename}으로 저장되었습니다.")

def load_crawled_data(year):
    filename = f'crawled_data_{year}.parquet'
    if os.path.exists(filename):
        df = pd.read_parquet(filename)
        print(f"{year}년 크롤링 데이터를 {filename}에서 불러왔습니다.")
        return df
    return None

async def main():
    try:
        years = [2023, 2024]
        dataframes = {}

        for year in years:
            df = load_crawled_data(year)
            if df is None:
                start_date, end_date = get_half_year_range(year)
                print(f"{year}년 상반기 뉴스 크롤링을 시작합니다... (기간: {start_date} ~ {end_date})")

                with ProcessPoolExecutor() as executor:
                    df = await asyncio.get_event_loop().run_in_executor(
                        executor, crawl_with_multiprocessing, start_date, end_date)
                df = pd.DataFrame(df)

                save_crawled_data(df, year)

            dataframes[year] = df

        print("뉴스 분석 및 비교를 시작합니다...")
        analyze_and_compare(dataframes[2023], dataframes[2024])
        print("모든 과정이 완료되었습니다.")

    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())