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
    
     
    1. Thiết lập seed cho thư viện `random` của Python.
    2. Thiết lập seed cho `numpy`.
    3. Thiết lập seed cho `torch` (CPU).
    4. Kiểm tra nếu có CUDA (GPU), thiết lập seed cho tất cả các GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản dùng chung cho cả SLM và MRCD inference.
    
     
    1. Chuyển đổi đầu vào sang kiểu chuỗi (str).
    2. Chuyển văn bản thành chữ thường (lowercase).
    3. Loại bỏ các URL (http/https/www) bằng regex.
    4. Loại bỏ các lượt nhắc tên (@mentions).
    5. Chỉ giữ lại các ký tự chữ cái, số, dấu gạch dưới, dấu # và khoảng trắng.
    6. Chuẩn hóa khoảng trắng (loại bỏ khoảng trắng thừa).
    7. Trả về văn bản đã xử lý.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    # FIX FOR VIETNAMESE: Giữ Unicode diacritics, loại bỏ dấu câu
    text = re.sub(r"[^\w\s#\u0300-\u036F]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())
    return text

def clean_query(text: str) -> str:
    """
    Làm sạch truy vấn: chuẩn hóa unicode, loại bỏ dấu câu, thu gọn khoảng trắng.
    
     
    1. Chuẩn hóa Unicode bằng NFKC để xử lý các ký tự đặc biệt.
    2. Loại bỏ tất cả các ký tự không phải chữ cái/số hoặc khoảng trắng.
    3. Thu gọn nhiều khoảng trắng liên tiếp thành một và cắt bỏ ở hai đầu.
    4. Trả về kết quả (str).
    """
    text = unicodedata.normalize("NFKC", text)
    # FIX FOR VIETNAMESE: Giữ Unicode diacritics, loại bỏ dấu câu
    text = re.sub(r"[^\w\s\u0300-\u036F]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Cắt ngắn văn bản đến độ dài max_length, ưu tiên cắt tại ranh giới từ.
    
     
    1. Nếu độ dài văn bản nhỏ hơn hoặc bằng max_length, trả về nguyên bản.
    2. Tìm vị trí khoảng trắng cuối cùng trong phạm vi max_length.
    3. Nếu tìm thấy khoảng trắng (cut_pos != -1), cắt tại đó và thêm dấu "...".
    4. Nếu không tìm thấy khoảng trắng (từ quá dài), cắt cứng tại max_length và thêm "...".
    5. Trả về chuỗi đã cắt.
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
    """
    Ghi nhật ký kết quả truy xuất vào file CSV để debug.
    
     
    1. Lấy đường dẫn file từ tham số hoặc từ cấu hình mặc định (RETRIEVAL_DEBUG_CSV).
    2. Nếu không có đường dẫn, bỏ qua (logging bị vô hiệu hóa).
    3. Kiểm tra xem file đã tồn tại chưa.
    4. Tạo thư mục chứa file nếu chưa có.
    5. Mở file ở chế độ append ("a").
    6. Nếu file mới tạo, ghi dòng tiêu đề (header).
    7. Ghi dòng dữ liệu mới bao gồm: tên hàm, truy vấn, tiêu đề, url và nội dung đoạn trích.
    8. Đóng file ngay sau khi ghi (thông qua `with`).
    9. Bắt lỗi ngoại lệ (nếu có) để tránh làm dừng chương trình chính.
    """
    from src.config import RETRIEVAL_DEBUG_CSV

    filepath = filepath or RETRIEVAL_DEBUG_CSV
    if not filepath:
        return  # Logging disabled

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
    """
    Ghi kết quả dự đoán cuối cùng của một sự kiện vào file CSV theo chế độ append.
    Đảm bảo tiết kiệm RAM ngay cả khi xử lý hàng nghìn sự kiện.
    
     
    1. Xác định đường dẫn file log kết quả (mặc định từ config: RESULTS_CSV).
    2. Nếu không có đường dẫn, thoát hàm.
    3. Kiểm tra sự tồn tại của file để quyết định có ghi header hay không.
    4. Đảm bảo thư mục đích tồn tại.
    5. Mở file bằng chế độ "a" (append) để ghi nối tiếp vào cuối file.
    6. Ghi dòng chứa thông tin chi tiết về dự đoán và trạng thái của round.
    """
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
    """
    Ghi nhật ký chi tiết từng vòng (trace) cho từng sự kiện.
    Lưu lại: Round, Input, SLM Pred, LLM Pred, Ground Truth, Confidence và Prompt.
    Prompt của LLM được ghi đầy đủ, không cắt ngắn.
    
     
    1. Xác định đường dẫn file trace (mặc định từ config: TRACE_CSV).
    2. Nếu không có đường dẫn, thoát hàm.
    3. Kiểm tra sự tồn tại của file để quyết định ghi header.
    4. Mở file ở chế độ append ("a").
    """
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
