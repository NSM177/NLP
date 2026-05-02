"""
Knowledge Agent: Two-mode orchestration.

Mode "wiki_only": Chỉ lấy Wikipedia entity definitions (K_wiki)
Mode "full":      Wikipedia + fact-check crawl & rerank (K_wiki + K_fact)

Per-pipeline-run caching supported.
"""

import wikipedia

from src.config import KNOWLEDGE_MODE
from src.retrieval.knowledge_retrieval import (
    retrieve_fact_evidence,
    analyze_claim_entities_and_query,
)


def query_wikipedia(entity: str, lang: str = "vi", fetch_full: bool = False) -> str:
    """
    Truy vấn nội dung từ Wikipedia cho một thực thể.
    
    Flow triển khai:
    1. Thiết lập ngôn ngữ truy vấn (mặc định là tiếng Việt "vi").
    2. Nếu fetch_full = True: Lấy toàn bộ nội dung trang bằng `wikipedia.page().content`.
    3. Ngược lại: Gọi hàm `wikipedia.summary` để lấy đoạn tóm tắt.
    4. Trả về nội dung đã lấy hoặc chuỗi "Not found" nếu có lỗi.
    """
    try:
        wikipedia.set_lang(lang)
        if fetch_full:
            page = wikipedia.page(entity, auto_suggest=True)
            return page.content
        else:
            return wikipedia.summary(entity, auto_suggest=True)
    except Exception:
        return "Not found"


def extract_wiki_knowledge_from_entities(entities: list, fetch_full: bool = False) -> dict:
    """
    Xây dựng định nghĩa/nội dung thực thể từ danh sách các thực thể đã trích xuất.
    
    1. Khởi tạo dictionary kết quả `res`.
    2. Duyệt qua từng thực thể trong danh sách đầu vào.
    3. Chuẩn hóa tên thực thể (xử lý cả kiểu dữ liệu chuỗi hoặc dictionary).
    4. Gọi `query_wikipedia` với cờ `fetch_full`.
    5. Nếu tìm thấy nội dung, lưu vào `res` với key là tên thực thể.
    6. Trả về dictionary chứa các nội dung đã tìm thấy.
    """
    res = {}
    for ent in entities:
        if isinstance(ent, str) and ent.strip():
            ent_text = ent.strip()
        elif isinstance(ent, dict) and ent.get("entity"):
            ent_text = str(ent.get("entity", "")).strip()
        else:
            continue

        summ = query_wikipedia(ent_text, "vi", fetch_full=fetch_full)
        if "Not found" not in summ:
            res[ent_text] = summ
    return res


def format_verified_reports(top_chunks: list) -> str:
    """
    Định dạng các đoạn hội thoại/báo cáo xác thực tin tức để đưa vào prompt.
    
     
    1. Kiểm tra nếu danh sách kết quả trống, trả về thông báo mặc định "No verified report found".
    2. Khởi tạo danh sách `lines`.
    3. Với mỗi đoạn trích (chunk):
       - Lấy tiêu đề nguồn và nội dung văn bản.
       - Định dạng theo cấu trúc: - Title: ... và - Key Information: ...
    4. Ghép các dòng lại thành một chuỗi văn bản hoàn chỉnh.
    """
    if not top_chunks:
        return (
            "- Title: No verified report found\n\n"
            "- Key Information: No trusted fact evidence found."
        )

    lines = []
    for item in top_chunks:
        title = item.get("title", "Unknown source")
        key_info = item.get("chunk_text", "")
        lines.append(f"- Title: {title}")
        lines.append("")
        lines.append(f"- Key Information: {key_info}")
        lines.append("")
    return "\n".join(lines).strip()


def format_entity_definitions(wiki_res: dict) -> str:
    """
    Định dạng các định nghĩa thực thể từ Wikipedia để đưa vào prompt.
    
     
    1. Kiểm tra nếu kết quả wiki trống, trả về thông báo mặc định.
    2. Khởi tạo danh sách `lines`.
    3. Với mỗi thực thể và định nghĩa trong `wiki_res`:
       - Định dạng theo cấu trúc: - Entity: ... và - Definition: ...
    4. Ghép các dòng lại và trả về chuỗi văn bản.
    """
    if not wiki_res:
        return "- Entity: N/A\n\n- Definition: No entity definition found."

    lines = []
    for entity, definition in wiki_res.items():
        lines.append(f"- Entity: {entity}")
        lines.append("")
        lines.append(f"- Definition: {definition}")
        lines.append("")
    return "\n".join(lines).strip()


# ============================================================
# Mode 1: Wiki Only
# ============================================================
def build_knowledge_wiki_only(text: str, wiki_fetch_full: bool = False) -> dict:
    """
    Chế độ wiki_only: chỉ trích xuất thực thể và truy vấn Wikipedia.
    Không thực hiện crawl các trang web xác thực tin tức (fact-check sites).
    
     
    1. Phân tích văn bản để lấy các thực thể bằng `analyze_claim_entities_and_query`.
    2. Trích xuất kiến thức Wiki từ các thực thể này.
    3. Định dạng các định nghĩa thành văn bản.
    4. Đóng gói vào thẻ xml `<ENTITY_DEFINITIONS>`.
    5. Trả về dictionary chứa kết quả và thông tin mode.
    """
    analysis = analyze_claim_entities_and_query(text, mode="wiki_only")
    entities = analysis.get("entities", [])

    wiki_res = extract_wiki_knowledge_from_entities(entities, fetch_full=wiki_fetch_full)
    entity_definitions = format_entity_definitions(wiki_res)

    combined_text = (
        "<ENTITY_DEFINITIONS>\n\n"
        f"{entity_definitions}\n\n"
        "</ENTITY_DEFINITIONS>"
    )

    return {"combined_text": combined_text, "mode": "wiki_only"}


# ============================================================
# Mode 2: Full (Wiki + Fact-check)
# ============================================================
def build_knowledge_full(text: str, fact_top_k: int = 3, wiki_fetch_full: bool = False) -> dict:
    """
    Chế độ full: Kết hợp Wikipedia và crawl các trang xác thực tin tức tin cậy.
    
     
    1. Gọi `retrieve_fact_evidence` để crawl và lấy các đoạn trích dẫn tin cậy từ Internet.
    2. Lấy danh sách thực thể từ kết quả phân tích của bước crawl.
    3. Trích xuất kiến thức Wikipedia cho các thực thể đó.
    4. Định dạng cả báo cáo xác thực (verified reports) và định nghĩa thực thể.
    5. Kết hợp tất cả vào một chuỗi văn bản lớn với các thẻ xml tương ứng.
    6. Trả về kết quả kèm theo thông tin mode "full".
    """
    fact_output = retrieve_fact_evidence(
        raw_query=text,
        max_urls=12,
        top_k_chunks=fact_top_k,
    )
    analysis = fact_output.get("analysis", {})
    entities = analysis.get("entities", [])

    wiki_res = extract_wiki_knowledge_from_entities(entities, fetch_full=wiki_fetch_full)
    fact_chunks = fact_output.get("top_chunks", [])

    verified_reports = format_verified_reports(fact_chunks)
    entity_definitions = format_entity_definitions(wiki_res)

    combined_text = (
        "<VERIFIED_REPORTS>\n\n"
        f"{verified_reports}\n\n"
        "</VERIFIED_REPORTS>\n\n"
        "<ENTITY_DEFINITIONS>\n\n"
        f"{entity_definitions}\n\n"
        "</ENTITY_DEFINITIONS>"
    )

    return {"combined_text": combined_text, "mode": "full"}


# ============================================================
# Dispatcher
# ============================================================
def build_knowledge_bundle(
    text: str,
    fact_top_k: int = 3,
    mode: str = None,
    wiki_fetch_full: bool = False,
) -> dict:
    """
    Xây dựng gói kiến thức (knowledge bundle) dựa trên chế độ được chọn.
    
     
    1. Xác định mode (mặc định lấy từ cấu hình KNOWLEDGE_MODE).
    2. Nếu mode là "wiki_only": Gọi `build_knowledge_wiki_only`.
    3. Nếu không: Gọi `build_knowledge_full` (mặc định cho "full").
    4. Trả về dictionary kết quả.
    """
    mode = mode or KNOWLEDGE_MODE

    if mode == "wiki_only":
        return build_knowledge_wiki_only(text, wiki_fetch_full=wiki_fetch_full)
    else:
        return build_knowledge_full(text, fact_top_k=fact_top_k, wiki_fetch_full=wiki_fetch_full)


def get_cached_knowledge_bundle_local(
    text: str,
    cache: dict,
    fact_top_k: int = 3,
    mode: str = None,
    wiki_fetch_full: bool = False,
) -> dict:
    """
    Lưu trữ ngữ cảnh kiến thức vào bộ nhớ đệm (cache) trong lần chạy hiện tại.
    Giúp tránh việc truy vấn lại cùng một nội dung nhiều lần.
    
     
    1. Nếu cache không tồn tại (None), gọi trực tiếp `build_knowledge_bundle`.
    2. Tạo cache_key dựa trên: văn bản, chế độ (mode) và số lượng đoạn trích (fact_top_k).
    3. Kiểm tra xem cache_key đã có trong `cache` chưa.
    4. Nếu có: Trả về kết quả từ cache.
    5. Nếu chưa: Gọi `build_knowledge_bundle`, lưu kết quả vào cache và trả về.
    """
    if cache is None:
        return build_knowledge_bundle(text, fact_top_k=fact_top_k, mode=mode, wiki_fetch_full=wiki_fetch_full)

    mode = mode or KNOWLEDGE_MODE
    cache_key = f"{text}||mode={mode}||fact_top_k={fact_top_k}||wiki_full={wiki_fetch_full}"
    if cache_key in cache:
        return cache[cache_key]

    bundle = build_knowledge_bundle(text, fact_top_k=fact_top_k, mode=mode, wiki_fetch_full=wiki_fetch_full)
    cache[cache_key] = bundle
    return bundle
