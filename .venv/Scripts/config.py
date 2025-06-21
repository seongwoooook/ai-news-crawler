from datetime import datetime, timedelta

# --- 기본 설정 ---
YEARS_TO_ANALYZE = [2023, 2024]
BASE_URL = "https://news.naver.com/breakingnews/section/105/283?date={date}"
CRAWL_START_MONTH = 1
CRAWL_START_DAY = 1
CRAWL_END_MONTH = 6
CRAWL_END_DAY = 30

# --- 크롤링 설정 ---
# 비동기 요청 시 동시 요청 수
SEMAPHORE_LIMIT = 100
# TCP 커넥터 설정
TCP_CONNECTOR_LIMIT = 300
TCP_CONNECTOR_TTL_DNS_CACHE = 300
# 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- 데이터 저장 경로 ---
# 크롤링된 원본 데이터 저장 경로
CRAWLED_DATA_DIR = "data"
# 분석 결과 및 시각화 이미지 저장 경로
OUTPUT_DIR = "output"
# 워드클라우드 마스크 이미지 경로
WORDCLOUD_MASK_PATH = "images/cloudshape.png"

# --- 텍스트 전처리 설정 ---
# 불용어 리스트
STOPWORDS = [
    '뉴스', '기자', '보도', '대표', '활용', '제공', '지원', '관리', '시장', '분기', '대비', '기록',
    '지난해', '디지털데일리', '디지털', '일리', '디지털 데일리', '디지털 일리', '통해', '위해', '대한',
    '위한', '의해', '이번', '기업', '사업', '고객', '산업', '글로벌', '국내', '한국', '서비스', '솔루션',
    '플랫폼', '기반', '구축', '운영', '진행', '계획', '예정', '도입', '출시', '발표', '확대', '전략',
    '협력', '파트너', '혁신', '효율', '지속', '추진', '전환', '적용', '통합', '올해', '지난', '이상',
    '최근', '시간', '매출', '달러', '억원', '비용', '투자', '실적', '영업', '기술', '시스템',
    '소프트웨어', '데이터', '정보', '개발', '사용', '사용자', '기능', '업무'
]

# AI 관련 기사 분류 키워드
AI_KEYWORDS = {'인공지능', 'AI', '머신러닝', '딥러닝', '신경망', '자연어처리', 'NLP', '컴퓨터비전', 'GPT', '제미나이', '생성형AI', 'gemini'}


# --- 분석 모델 파라미터 ---
# TF-IDF 최대 피처 수
MAX_FEATURES = 1000
# 토픽 모델링(NMF) 토픽 개수
N_COMPONENTS = 3
# 상위 키워드/바이그램 개수
TOP_N_KEYWORDS = 100
TOP_N_BIGRAMS = 20
TOP_N_TOPIC_WORDS = 10

# --- 시각화 설정 ---
FONT_PATH = r'C:\Windows\Fonts\NanumGothic.ttf'