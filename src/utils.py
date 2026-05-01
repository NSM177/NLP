"""
Utility functions for MRCD Framework.
Text preprocessing, cleaning, seeding, and debug logging.
"""

import os
import re
import csv
import random
import unicodedata

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Thiết lập seed ngẫu nhiên để đảm bảo tính tái lập (reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt dùng chung cho cả SLM và MRCD inference.
    
    Các bước:
    1. Chuyển về chuỗi, chuẩn hóa Unicode (NFC).
    2. Chuyển thành chữ thường (lowercase).
    3. Loại bỏ URL, @mentions.
    4. Giữ lại chữ cái (kể cả có dấu), số, khoảng trắng, dấu #.
    5. Chuẩn hóa khoảng trắng.
    """
    text = str(text)
    # Chuẩn hóa Unicode: đưa các tổ hợp dấu về dạng ký tự đơn (viết liền)
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    # Loại bỏ URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Loại bỏ mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Giữ lại: chữ cái (có dấu), số, khoảng trắng, dấu #
    # re.UNICODE để \w bao gồm chữ cái tiếng Việt
    text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    # Chuẩn hóa khoảng trắng (xóa space thừa)
    text = " ".join(text.split())
    return text


def clean_query(text: str) -> str:
    """
    Làm sạch truy vấn để gửi lên search engine:
    - Chuẩn hóa Unicode (NFC)
    - Chỉ giữ chữ cái (có dấu), số, khoảng trắng
    - Xóa dấu câu, ký tự đặc biệt
    """
    text = unicodedata.normalize("NFC", str(text))
    # Giữ lại chữ cái (kể cả có dấu), số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Cắt ngắn văn bản đến độ dài max_length, ưu tiên cắt tại ranh giới từ.
    """
    if len(text) <= max_length:
        return text
    cut_pos = text.rfind(" ", 0, max_length)
    if cut_pos == -1:
        return text[:max_length] + "..."
    return text[:cut_pos] + "..."


def log_retrieval_to_csv(
    func_name: str,
    query: str,
    title: str,
    url: str,
    snippet: str,
    filepath: str = None,
):
    """Ghi nhật ký kết quả truy xuất vào file CSV để debug."""
    from src.config import RETRIEVAL_DEBUG_CSV

    filepath = filepath or RETRIEVAL_DEBUG_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["source_function", "query", "title", "url", "snippet"]
                )
            writer.writerow([func_name, query, title, url, snippet])
    except Exception:
        pass


def log_prediction_to_csv(
    event_id: int,
    text: str,
    label: int,
    conf: float,
    round_id: int,
    status: str,
    filepath: str = None,
):
    """Ghi kết quả dự đoán cuối cùng của một sự kiện vào file CSV."""
    from src.config import RESULTS_CSV

    filepath = filepath or RESULTS_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["event_id", "label", "confidence", "round", "status", "text_snippet"]
                )
            text_snippet = text.replace("\n", " ")
            writer.writerow([event_id, label, conf, round_id, status, text_snippet])
    except Exception:
        pass


def log_round_trace_to_csv(
    round_id: str | int,
    event_id: int,
    text: str,
    y_slm: int,
    y_llm: int,
    ground_truth: int | str,
    conf_slm: float,
    prompt: str,
    filepath: str = None,
):
    """Ghi nhật ký chi tiết từng vòng (trace) cho từng sự kiện."""
    from src.config import TRACE_CSV

    filepath = filepath or TRACE_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "round",
                        "event_id",
                        "y_slm",
                        "y_llm",
                        "ground_truth",
                        "conf_slm",
                        "input",
                        "prompt",
                    ]
                )
            writer.writerow(
                [
                    round_id,
                    event_id,
                    y_slm,
                    y_llm,
                    ground_truth if ground_truth is not None else "N/A",
                    f"{conf_slm:.4f}",
                    text,
                    str(prompt),
                ]
            )
    except Exception:
        pass