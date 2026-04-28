"""
Dataset and data loading for SLM (RoBERTa-based fake news classifier).
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import TRAIN_CSV, VAL_CSV, TEST_CSV
from src.utils import preprocess_text


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news classification with RoBERTa tokenizer."""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        """
        Khởi tạo tập dữ liệu FakeNewsDataset cho PhoBERT.
        
         
        1. Gán danh sách văn bản (texts) và nhãn (labels).
        2. Gán bộ mã hóa (tokenizer) của AutoTokenizer (hỗ trợ PhoBERT).
        3. Thiết lập độ dài tối đa cho chuỗi token (max_len=256 cho tiếng Việt).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Trả về tổng số mẫu trong tập dữ liệu.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu tại chỉ số idx.
        
         
        1. Lấy văn bản tại vị trí idx và thực hiện tiền xử lý (`preprocess_text`).
        2. Sử dụng tokenizer để chuyển văn bản thành các token (input_ids, attention_mask).
        3. Thiết lập padding và truncation để khớp với `max_len`.
        4. Chuyển đổi nhãn tại vị trí idx sang kiểu LongTensor của PyTorch.
        5. Trả về một dictionary chứa các tensors cần thiết cho mô hình.
        """
        text = preprocess_text(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data_from_csv(
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    test_csv: str = TEST_CSV,
    text_column: str = "text",
):
    """
    Nạp dữ liệu từ các file CSV (train, val, test).
    
     
    1. Định nghĩa hàm helper `load_csv_file` để đọc từng file.
    2. Trong `load_csv_file`:
       - Kiểm tra file tồn tại.
       - Đọc CSV bằng pandas, tự động phát hiện cột văn bản (text hoặc content).
       - Tiền xử lý văn bản và chuyển đổi nhãn sang nhị phân (0 cho True, 1 cho Fake).
    3. Nạp lần lượt dữ liệu huấn luyện, kiểm định và thử nghiệm.
    4. Gộp (merge) tập huấn luyện (train) và tập kiểm định (val) thành một tập huấn luyện lớn.
    5. Trả về bộ dữ liệu đã được gộp và xử lý.
    """

    def load_csv_file(filepath, text_col):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return [], []

        try:
            df = pd.read_csv(filepath)
            
            # Auto-detect text column
            if text_col not in df.columns:
                # Fallback: try 'content' if 'text' not found, or vice versa
                alt_col = "content" if text_col == "text" else "text"
                if alt_col in df.columns:
                    text_col = alt_col
                else:
                    print(f"Warning: Neither '{text_col}' nor '{alt_col}' found in {filepath}")
                    print(f"Available columns: {list(df.columns)}")
                    return [], []
            
            texts = [preprocess_text(t) for t in df[text_col].astype(str).tolist()]
            # Support Vietnamese dataset: 0 = Real/True, 1 = Fake
            labels = []
            for label in df["label"].astype(str).tolist():
                label_str = str(label).strip().lower()
                if label_str in ["0", "true", "real", "non-rumor", "xác thực", "thật"]:
                    labels.append(0)
                elif label_str in ["1", "false", "fake", "rumor", "giả mạo", "giả"]:
                    labels.append(1)
                else:
                    # Fallback: assume numeric value
                    try:
                        labels.append(int(float(label_str)))
                    except ValueError:
                        print(f"Warning: Unknown label value: {label_str}, defaulting to 1 (Fake)")
                        labels.append(1)
            print(f"Loaded {len(texts)} samples from {filepath}")
            return texts, labels
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return [], []

    print("Loading data from CSV files...")
    train_texts_raw, train_labels_raw = load_csv_file(train_csv, text_column)
    val_texts_raw, val_labels_raw = load_csv_file(val_csv, text_column)
    test_texts, test_labels = load_csv_file(test_csv, text_column)

    train_texts = train_texts_raw + val_texts_raw
    train_labels = train_labels_raw + val_labels_raw

    print(f"Merged train size (train+val): {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")

    return train_texts, train_labels, test_texts, test_labels
