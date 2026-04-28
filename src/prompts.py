"""Prompt builders for MRCD Framework (Vietnamese Language Support).

Two classification prompt variants:
- wiki_only: chỉ dùng K_wiki (định nghĩa thực thể)
- full: dùng K_wiki + K_fact (các báo cáo đã xác minh)

LLM chỉ trả về "Real" hoặc "Fake".
"""


def build_dual_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt để trích xuất thực thể và tạo truy vấn tìm kiếm (tiếng Việt).
    """
    prompt = (
        "Bạn là chuyên gia Trích xuất Kiểm chứng Sự kiện. Nhiệm vụ của bạn là xử lý văn bản tin tức thô "
        "và tạo ra hai kết quả đồng thời cho Hệ thống Truy xuất hai giai đoạn.\n\n"
        "NHIỆM VỤ 1: CÁC THỰC THỂ WIKIPEDIA (Để Truy xuất Kiến thức)\n"
        "Trích xuất 1 đến 4 thực thể được đặt tên chính (Người, Tổ chức, Địa điểm, Sự kiện) từ văn bản "
        "những cái quan trọng để xác minh tuyên bố và có khả năng cao có trang Wikipedia.\n\n"
        "NHIỆM VỤ 2: TRY VẤN TRUNG LẬP (Để Tìm kiếm Bài viết Kiểm chứng Sự kiện)\n"
        "Tạo một truy vấn tìm kiếm duy nhất, ngắn gọn để truy xuất các bài viết thực tế. "
        "QUY TẮC CỨNG: Chỉ tập trung vào các chủ đề thực tế cốt lõi. LOẠI BỎ tất cả các từ clickbait, hoa mỹ, "
        "hoặc cảm xúc (ví dụ: 'nóng hổi', 'khẩn cấp', 'chữa được', 'bí mật'). "
        "KHÔNG sử dụng dấu ngoặc kép (\"\") hoặc bất kỳ toán tử tìm kiếm nào.\n\n"
        "ĐỊNH DẠNG ĐẦU RA:\n"
        "Chỉ trả về một đối tượng JSON hợp lệ. KHÔNG bao bọc trong các thẻ markdown (như ```json), không mở đầu, không giải thích.\n"
        'Lược đồ: {"entities": ["entity_1", "entity_2"], "query": "query"}\n\n'
        f"Văn bản đầu vào: {text}"
    )
    return prompt

def build_entity_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt CHỈ trích xuất thực thể cho chế độ wiki_only (tiếng Việt).
    """
    prompt = (
        "Bạn là chuyên gia Trích xuất Kiểm chứng Sự kiện.\n\n"
        "NHIỆM VỤ: CÁC THỰC THỂ WIKIPEDIA (Để Truy xuất Kiến thức)\n"
        "Trích xuất 1 đến 4 thực thể được đặt tên chính (Người, Tổ chức, Địa điểm, Sự kiện) từ văn bản "
        "những cái quan trọng để xác minh tuyên bố và có khả năng cao có trang Wikipedia.\n\n"
        "ĐỊNH DẠNG ĐẦU RA:\n"
        "Chỉ trả về một đối tượng JSON hợp lệ. KHÔNG bao bọc trong các thẻ markdown (như ```json), không mở đầu, không giải thích.\n"
        'Lược đồ: {"entities": ["entity_1", "entity_2"]}\n\n'
        f"Văn bản đầu vào: {text}"
    )
    return prompt


def build_classification_prompt(
    text: str,
    knowledge_k: str,
    demos: list,
) -> str:
    """
    Xây dựng prompt phân loại bằng tiếng Việt.
    """
    # 1. HEADER
    header = f"""Bạn là chuyên gia phát hiện tin giả nâng cao.

NỀN TẢNG THÔNG TIN:
{knowledge_k}

HƯỚNG DẪN:
Phân loại bài viết tin tức dưới đây là Thật (Real) hoặc Giả (Fake).
- "Real" nghĩa là bài viết chính xác về sự kiện và đáng tin cậy.
- "Fake" nghĩa là bài viết giả mạo, gây hiểu lầm, hoặc chưa được xác minh.

QUY TẮC CỨNG: Chỉ viết MỘT từ: "Real" hoặc "Fake". 
Không có giải thích, không có dấu câu.
Chỉ viết từ đó.

VÍ DỤ:"""

    # 2. FEW-SHOT DEMOS
    examples = _build_demo_section(demos)

    # 3. TAIL & TARGET
    tail = f"""
----------------------------------------
BÀI VIẾT CẦN PHÂN LOẠI:
Nội dung: "{text.strip()}"

Kết luận:"""

    return header + examples + tail


def _build_demo_section(demos: list) -> str:
    """
    Xây dựng phần danh sách các ví dụ few-shot cho prompt (tiếng Việt).\n    
    1. Kiểm tra nếu danh sách demos trống, trả về thông báo rỗng.
    2. Lặp qua danh sách demos (bắt đầu từ index 1).
    3. Với mỗi demo:
       - Lấy nhãn (label) và nội dung văn bản (giới hạn 1000 ký tự đầu).
       - Định dạng theo cấu trúc: [Ví dụ n], Nội dung: "...", Kết luận: ...
    4. Ghép tất cả các ví dụ thành một chuỗi duy nhất và trả về.
    """
    if not demos:
        return "\n(Không có ví dụ)\n"

    examples = ""
    for i, demo in enumerate(demos, start=1):
        label_str = demo.get("label", "Chưa xác định")
        text_demo = demo.get("text", "")[:1000].strip()
        examples += f'\n[Ví dụ {i}]\nNội dung: "{text_demo}..."\nKết luận: {label_str}\n'
    return examples
