"""
Branch 1: Demonstration Retrieval
Bing News search + static AG News corpus + BM25 top-k selection.
"""

import io
import os

import requests
import pandas as pd
from ddgs import DDGS
from rank_bm25 import BM25Okapi

from src.config import VI_NEWS_CORPUS_PATH
from src.utils import clean_query, truncate_text, log_retrieval_to_csv
from src.labels import generate_demo_label


# def load_news_corpus(url: str = VI_NEWS_CORPUS_PATH) -> list:
#     """
#     Tải tập dữ liệu AG News và nạp vào danh sách các tài liệu văn bản.
    
     
#     1. Gửi yêu cầu GET để tải CSV từ URL.
#     2. Chuyển nội dung phản hồi thành luồng dữ liệu (StringIO).
#     3. Sử dụng pandas để đọc CSV với các cột: class, title, desc.
#     4. Gộp 'title' và 'desc' thành một chuỗi duy nhất cho mỗi dòng.
#     5. Trả về danh sách các văn bản đã gộp.
#     """
#     print(f" Downloading News Corpus from {url}...")
#     try:
#         response = requests.get(url)
#         response.raise_for_status()

#         csv_content = io.StringIO(response.text)
#         df = pd.read_csv(csv_content, header=None, names=["class", "title", "desc"])
#         corpus_texts = (df["title"] + " " + df["desc"]).tolist()

#         print(f" Loaded {len(corpus_texts)} documents from AG News.")
#         return corpus_texts
#     except Exception as e:
#         print(f" Error downloading corpus: {e}")
#         return []


import pandas as pd
from src.config import VI_NEWS_CORPUS_PATH

def load_news_corpus(corpus_path: str = None, max_samples: int = 60000) -> list:
    """
    Tải tập dữ liệu tin tức tiếng Việt từ file CSV và lấy mẫu ngẫu nhiên.
    
    Args:
        corpus_path: Đường dẫn đến file CSV.
        max_samples: Số lượng mẫu tối đa cần lấy (mặc định 50000).
    """
    if corpus_path is None:
        corpus_path = VI_NEWS_CORPUS_PATH

    if not os.path.exists(corpus_path):
        print(f" Warning: Corpus file not found at {corpus_path}. Using empty corpus.")
        return []

    try:
        df = pd.read_csv(corpus_path)
        
        # Xác định cột văn bản
        if 'text' in df.columns:
            text_series = df['text']
        else:
            print(" Error: CSV must contain 'text' column.")
            return []
        
        # Lấy mẫu ngẫu nhiên nếu số lượng vượt quá max_samples
        if len(df) > max_samples:
            df_sample = df.sample(n=max_samples, random_state=42)
            text_series = df_sample['text']
            print(f" Sampled {max_samples} documents from {len(df)} total.")
        else:
            text_series = text_series
        
        # Chuyển sang string, thay thế NaN bằng chuỗi rỗng
        text_series = text_series.fillna('').astype(str)
        
        # Loại bỏ các dòng rỗng
        corpus_texts = [t.strip() for t in text_series.tolist() if t.strip()]
        
        print(f" Loaded {len(corpus_texts)} documents from Vietnamese corpus.")
        return corpus_texts
    except Exception as e:
        print(f" Error loading corpus: {e}")
        return []
def search_news(query: str, max_results: int = 10, region: str = "vn-vi") -> list:
    """
    Tìm kiếm các đoạn tin tức mới nhất qua DuckDuckGo (backend Bing).
    
     
    1. Làm sạch (clean) và cắt ngắn (truncate) truy vấn đầu vào.
    2. Khởi tạo DuckDuckGo Search với timeout.
    3. Gọi API tìm kiếm news với các tham số: region, safesearch, backend="bing".
    4. Lặp qua các kết quả để lấy 'title' và 'body'.
    5. Ghi nhật ký (log) thông tin tìm kiếm vào file CSV.
    6. Trả về danh sách các đoạn văn bản tin tức.
    """
    query = clean_query(query)
    query = truncate_text(query, max_length=50)

    news_items = []
    try:
        with DDGS(timeout=20) as ddgs:
            results_gen = ddgs.news(
                query=query,
                region=region,
                safesearch="off",
                timelimit=None,
                max_results=max_results,
                backend="bing",
            )

            for i, result in enumerate(results_gen):
                if i >= max_results:
                    break

                title = result.get("title", "")
                body = result.get("body", "")
                news_items.append(f"{title}\n{body}")
                url = result.get("url", result.get("href", ""))
                log_retrieval_to_csv("search_news", query, title, url, body)
    except Exception:
        pass
    return news_items


def retrieve_demonstrations(query: str, corpus_items: list, k: int = 4) -> list:
    """
    Sử dụng thuật toán BM25 để lấy ra top-k ví dụ minh họa (demonstrations).
    
     
    1. Kiểm tra nếu corpus trống thì trả về danh sách rỗng.
    2. Tokenize (chia từ) và chuyển corpus sang chữ thường.
    3. Khởi tạo mô hình xếp hạng BM25Okapi với corpus đã tokenize.
    4. Tính toán điểm số BM25 cho truy vấn (đã tokenize).
    5. Sắp xếp các tài liệu theo điểm số giảm dần và lấy top-k chỉ số (indices).
    6. Với mỗi chỉ số trong top-k:
       - Lấy nội dung văn bản.
       - Tạo nhãn giả (pseudo-label) ngẫu nhiên bằng `generate_demo_label`.
       - Đóng gói thành object kèm thông tin nguồn.
    7. Trả về danh sách các demonstrations.
    """
    if not corpus_items:
        return []

    tokenized_corpus = [doc.lower().split() for doc in corpus_items]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in scored_indices[:k]]

    demonstrations = []
    for i in top_k:
        content = corpus_items[i]
        demonstrations.append(
            {
                "text": content,
                "label": generate_demo_label(content),
                "source": "Bing/Retrieved",
            }
        )
    return demonstrations
