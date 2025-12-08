# ==================== CÀI ĐẶT THAM SỐ CHO INFERENCE ====================
# Đặt các hằng số này ở đầu file hoặc ngay trước hàm infer_qwen()
MAX_NEW_TOKENS_FOR_REASONING = 512  # Token tối đa cho phần output (Đáp án + Lý do)
CONTEXT_MAX_LENGTH = 1024          # Token tối đa cho Context Input
REASONING_MAX_LENGTH = 500         # Giới hạn độ dài tối đa của reasoning khi parsing

# ==================== CELL 11 — MODULE 3: SUY LUẬN (QWEN + LUẬT + QUY TẮC) ====================

def infer_qwen(q: str, options: List[str], scene: str, labels: List[str],
               ocr: List[str], laws: List[str], temporal: List[str]) -> Tuple[str, str]:
    """Suy luận với Qwen (ĐÃ SỬA PROMPT VÀ TOKENS)"""
    model, tok = init_qwen()
    
    # Fallback nếu model không load được
    if model is None or tok is None:
        return rule_based_inference(q, options, scene, labels, ocr, laws, temporal)

    try:
        # Build context
        ctx_parts = []
        law_context_str_full = "" # Lưu toàn bộ luật
        
        # Thêm thông tin từ video vào Context
        if scene:
            ctx_parts.append(f"Cảnh: {scene}")
        if temporal:
            ctx_parts.append(f"Thời gian: {temporal[0]}") 
        if labels:
            ctx_parts.append(f"Nhãn: {', '.join(labels)}")
        if ocr and any(kw in q.lower() for kw in ['tốc độ', 'km/h', 'số', 'biển']):
            ocr_nums = [x for x in ocr if re.search(r'\d', x)]
            if ocr_nums:
                ctx_parts.append(f"OCR: {', '.join(ocr_nums)}") 

        # Thêm Luật Giao thông từ RAG vào Context
        if laws:
            law_context_str_full = " ".join(laws)
            ctx_parts.append(f"Luật Giao thông liên quan: {law_context_str_full}")

        # Gộp tất cả context và giới hạn độ dài tổng thể
        context = " | ".join(ctx_parts)
        if len(context) > CONTEXT_MAX_LENGTH:
             context = context[:CONTEXT_MAX_LENGTH] + "..."
             
        # Build prompt (PHẦN SỬA CHỮA QUAN TRỌNG CHO REASONING)
        opt_txt = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        # 1. Sửa system_msg: Yêu cầu cả đáp án và lý do
        system_msg = "Bạn là trợ lý lái xe chuyên nghiệp. Hãy phân tích ngữ cảnh, câu hỏi và luật giao thông để đưa ra **chữ cái đáp án** đúng nhất (A/B/C/D) cùng với **lý do giải thích chi tiết**."
        
        # 2. Sửa user_msg: Cấu trúc lại và yêu cầu trả lời đủ
        user_msg = f"""
Thông tin Ngữ cảnh & Luật liên quan:
--------------------------------
{context}
--------------------------------

Câu hỏi: {q}
Lựa chọn (Chỉ chọn 1):
{opt_txt}

Hãy trả lời bằng cách đưa ra chữ cái đáp án, sau đó là Lý do giải thích:
"""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Tăng max_length cho input (cần thiết nếu context lớn)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device) 
        prompt_len = inp.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS_FOR_REASONING, # ✅ SỬA LỖI NÀY: Đảm bảo không bị cắt cụt
                do_sample=True,
                temperature=0.15,
                top_p=0.9,
                pad_token_id=tok.pad_token_id or tok.eos_token_id
            )

        gen_tokens = out[0][prompt_len:]
        response = tok.decode(gen_tokens, skip_special_tokens=True).strip()

        del out, inp, gen_tokens
        torch.cuda.empty_cache()

        # Parse (Cải thiện logic parsing để lấy toàn bộ Reasoning)
        ans = None
        
        # 1. Tìm chữ cái đáp án
        match_ans = re.search(r'\b([ABCD])\b', response, re.IGNORECASE)
        if match_ans:
            ans = match_ans.group(1).upper()
            
        # 2. Lấy phần reasoning
        # Loại bỏ chữ cái đáp án để lấy phần còn lại làm reasoning
        reasoning = response.replace(ans, '').strip() if ans else response.strip()
        
        # Cắt reasoning nếu quá dài
        if len(reasoning) > REASONING_MAX_LENGTH:
            reasoning = reasoning[:REASONING_MAX_LENGTH] + "..."
            
        # Kiểm tra tính hợp lệ của đáp án (Ans)
        if ans and 0 <= ord(ans) - 65 < len(options):
            # Trả về đáp án và lý do (reasoning)
            return ans, reasoning 

        # Fallback (Nếu Qwen không trả lời đúng format)
        return rule_based_inference(q, options, scene, labels, ocr, laws, temporal)

    except Exception as e:
        print(f"❌ Qwen error: {e}")
        return rule_based_inference(q, options, scene, labels, ocr, laws, temporal)


def rule_based_inference(q: str, options: List[str], scene: str, labels: List[str],
                         ocr: List[str], laws: List[str], temporal: List[str]) -> Tuple[str, str]:
    """Rule-based fallback (Giữ nguyên logic quy tắc)"""
    import random

    q_lower = q.lower()

    # Dùng luật trích ra làm phần giải thích mặc định
    law_context_str = ""
    if laws:
        law_context_str = " | Luật: " + " | ".join(laws)

    # Rule 1: Traffic lights
    if any(kw in q_lower for kw in ['đèn đỏ', 'đèn xanh', 'đèn vàng']):
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if 'đèn đỏ' in q_lower and 'dừng' in opt_lower:
                return chr(65 + i), f"Rule: Đèn đỏ phải dừng" + law_context_str
            if 'đèn xanh' in q_lower and 'đi' in opt_lower:
                return chr(65 + i), f"Rule: Đèn xanh được đi" + law_context_str
            if 'đèn vàng' in q_lower and ('chậm' in opt_lower or 'giảm' in opt_lower):
                return chr(65 + i), f"Rule: Đèn vàng giảm tốc" + law_context_str

    # Rule 2: Speed limits from OCR
    if any(kw in q_lower for kw in ['tốc độ', 'km/h', 'giới hạn']):
        numbers = []
        for txt in ocr:
            match = re.findall(r'\d+', txt)
            numbers.extend([int(x) for x in match if 0 < int(x) <= 120])

        if numbers:
            speed_limit = max(numbers)
            for i, opt in enumerate(options):
                if str(speed_limit) in opt:
                    return chr(65 + i), f"Rule: OCR speed={speed_limit}" + law_context_str

    # Rule 3: Stop/Yield signs
    if any(kw in q_lower for kw in ['dừng', 'stop', 'nhường đường']):
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if any(kw in opt_lower for kw in ['dừng', 'stop', 'nhường']):
                return chr(65 + i), f"Rule: Sign detected" + law_context_str

    # Rule 4: Check labels
    if labels:
        label_str = ' '.join(labels).lower()
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if any(word in label_str for word in opt_lower.split() if len(word) > 3):
                return chr(65 + i), f"Rule: Label match" + law_context_str

    # Rule 5: Law context
    if laws:
        law_text = ' '.join(laws).lower()
        best_match = -1
        best_score = 0

        for i, opt in enumerate(options):
            opt_words = set(opt.lower().split())
            score = sum(1 for word in opt_words if word in law_text and len(word) > 3)
            if score > best_score:
                best_score = score
                best_match = i

        if best_match >= 0:
            return chr(65 + best_match), f"Rule: Law context" + law_context_str

    # Rule 6: Longest/most detailed option (heuristic)
    longest_idx = max(range(len(options)), key=lambda i: len(options[i]))
    return chr(65 + longest_idx), "Rule: Longest option" + law_context_str
