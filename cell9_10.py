# ==================== CELL 9: CONFIGURATION & IMPORTS ====================
# Chạy cell này TRƯỚC khi chạy cell 10

import torch
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== CONFIGURATION =====
USE_QWEN = True  # Đặt False để bỏ qua Qwen

# ===== TÙY CHỌN MODEL (Chọn 1 trong các model dưới) =====
# Model chính thức (yêu cầu đăng nhập HF nếu private):
# QWEN_MODEL = "Qwen/Qwen2.5-4B-Instruct"

# Model thử nghiệm khác (public, không cần đăng nhập):
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Nhẹ hơn, dễ load
# QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Siêu nhẹ
# QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Cực nhẹ

print(f"✅ Configuration loaded - USE_QWEN={USE_QWEN}")
print(f"✅ Model name: {QWEN_MODEL}")

# ==================== HELPER FUNCTIONS - Scene & Temporal ====================

def scene_text_for_video(video_key, scene_map):
    """
    Lấy mô tả scene cho video

    Args:
        video_key: ID video (vd: "video001")
        scene_map: Dict mapping video_id -> scene description

    Returns:
        str: Scene description hoặc chuỗi rỗng
    """
    if not scene_map or not isinstance(scene_map, dict):
        return ""

    # Thử các cách match khác nhau
    # 1. Exact match
    if video_key in scene_map:
        scene = scene_map[video_key]
        if isinstance(scene, str) and scene.strip():
            return scene.strip()

    # 2. Case-insensitive match
    video_key_lower = video_key.lower()
    for k, v in scene_map.items():
        if k.lower() == video_key_lower:
            if isinstance(v, str) and v.strip():
                return v.strip()

    # 3. Partial match (nếu video_key là substring)
    for k, v in scene_map.items():
        if video_key in k or k in video_key:
            if isinstance(v, str) and v.strip():
                return v.strip()

    return ""


def rule_engine_temporal(video_key, temporal_map):
    """
    Trích xuất temporal facts từ video

    Args:
        video_key: ID video
        temporal_map: Dict mapping video_id -> temporal info

    Returns:
        list: List các temporal facts
    """
    if not temporal_map or not isinstance(temporal_map, dict):
        return []

    temporal_facts = []

    # 1. Exact match
    if video_key in temporal_map:
        temp_data = temporal_map[video_key]
        temporal_facts.extend(_parse_temporal_data(temp_data))

    # 2. Case-insensitive match
    if not temporal_facts:
        video_key_lower = video_key.lower()
        for k, v in temporal_map.items():
            if k.lower() == video_key_lower:
                temporal_facts.extend(_parse_temporal_data(v))
                break

    # 3. Partial match
    if not temporal_facts:
        for k, v in temporal_map.items():
            if video_key in k or k in video_key:
                temporal_facts.extend(_parse_temporal_data(v))
                break

    return temporal_facts


def _parse_temporal_data(temp_data):
    """
    Parse temporal data từ nhiều format khác nhau

    Args:
        temp_data: Có thể là str, list, hoặc dict

    Returns:
        list: List các temporal facts
    """
    facts = []

    # Case 1: Nếu là string
    if isinstance(temp_data, str):
        if temp_data.strip():
            facts.append(temp_data.strip())

    # Case 2: Nếu là list
    elif isinstance(temp_data, list):
        for item in temp_data:
            if isinstance(item, str) and item.strip():
                facts.append(item.strip())
            elif isinstance(item, dict):
                # Extract từ dict
                for key in ['time', 'weather', 'condition', 'description', 'text']:
                    if key in item and isinstance(item[key], str):
                        facts.append(item[key].strip())

    # Case 3: Nếu là dict
    elif isinstance(temp_data, dict):
        # Thử các keys phổ biến
        common_keys = ['time', 'weather', 'condition', 'time_of_day',
                      'weather_condition', 'temporal', 'description', 'text']

        for key in common_keys:
            if key in temp_data:
                value = temp_data[key]
                if isinstance(value, str) and value.strip():
                    facts.append(value.strip())
                elif isinstance(value, list):
                    facts.extend(_parse_temporal_data(value))

        # Nếu không tìm thấy, lấy tất cả string values
        if not facts:
            for v in temp_data.values():
                if isinstance(v, str) and v.strip():
                    facts.append(v.strip())

    return facts


# ==================== UTILITY: EXTRACT VIDEO KEY ====================

def extract_video_key(question_id):
    """
    Trích xuất video key từ question ID

    Args:
        question_id: vd "testa_0201", "testb_0305"

    Returns:
        str: video key, vd "video201", "video305"
    """
    import re

    # Pattern 1: "testa_0201" -> "video201" hoặc "0201"
    match = re.search(r'_(\d+)', question_id)
    if match:
        num = match.group(1)

        # Thử cả 2 format
        keys_to_try = [
            f"video{num}",           # "video0201"
            f"video{int(num)}",      # "video201" (bỏ leading zeros)
            num,                     # "0201"
            str(int(num))           # "201"
        ]

        return keys_to_try

    return [question_id]


# ==================== ENHANCED PIPELINE INTEGRATION ====================

def get_video_context(question_id, scene_map, temporal_map):
    """
    Lấy toàn bộ context từ video (scene + temporal)

    Args:
        question_id: ID câu hỏi
        scene_map: Scene mapping
        temporal_map: Temporal mapping

    Returns:
        tuple: (scene_text, temporal_facts)
    """
    video_keys = extract_video_key(question_id)

    scene_text = ""
    temporal_facts = []

    # Thử tất cả các video keys
    for vkey in video_keys:
        if not scene_text:
            scene_text = scene_text_for_video(vkey, scene_map)

        if not temporal_facts:
            temporal_facts = rule_engine_temporal(vkey, temporal_map)

        # Nếu đã có cả 2, break
        if scene_text and temporal_facts:
            break

    return scene_text, temporal_facts


# ==================== TEST FUNCTIONS ====================

def test_helper_functions():
    """Test các helper functions"""

    # Mock data
    test_scene_map = {
        "video201": "Xe đang lưu thông trên đường cao tốc",
        "video305": "Ngã tư có đèn giao thông, nhiều xe"
    }

    test_temporal_map = {
        "video201": ["Ban ngày", "Trời nắng"],
        "video305": {
            "time": "Ban đêm",
            "weather": "Trời mưa"
        }
    }

    print("🧪 TESTING HELPER FUNCTIONS\n")

    # Test 1: Scene extraction
    print("1️⃣ Scene Text:")
    scene1 = scene_text_for_video("video201", test_scene_map)
    print(f"   video201: {scene1}")

    scene2 = scene_text_for_video("video305", test_scene_map)
    print(f"   video305: {scene2}")

    # Test 2: Temporal extraction
    print("\n2️⃣ Temporal Facts:")
    temp1 = rule_engine_temporal("video201", test_temporal_map)
    print(f"   video201: {temp1}")

    temp2 = rule_engine_temporal("video305", test_temporal_map)
    print(f"   video305: {temp2}")

    # Test 3: Video key extraction
    print("\n3️⃣ Video Key Extraction:")
    keys1 = extract_video_key("testa_0201")
    print(f"   testa_0201 -> {keys1}")

    keys2 = extract_video_key("testb_0305")
    print(f"   testb_0305 -> {keys2}")

    # Test 4: Full context
    print("\n4️⃣ Full Context:")
    scene, temp = get_video_context("testa_0201", test_scene_map, test_temporal_map)
    print(f"   Scene: {scene}")
    print(f"   Temporal: {temp}")

    print("\n✅ All tests completed!")


# ==================== VALIDATION ====================
print("✅ Helper functions loaded:")
print("   - scene_text_for_video()")
print("   - rule_engine_temporal()")
print("   - extract_video_key()")
print("   - get_video_context()")

# Uncomment để test:
# test_helper_functions()


# ==================== CELL 10: QWEN2.5-4B MODEL - LOAD & INFERENCE ====================

# ===== PHẦN 1: LOAD MODEL VỚI AUTO-FALLBACK =====
# _USE_QWEN_OK = False
# tok = None
# qwen = None

# if USE_QWEN:
#     print("🔄 Loading Qwen2.5-4B-Instruct model...")

#     try:
#         # ===== TRY 1: 4-BIT QUANTIZATION (BEST) =====
#         try:
#             print("   Attempting 4-bit quantization...")
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4"
#             )

#             tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
#             qwen = AutoModelForCausalLM.from_pretrained(
#                 QWEN_MODEL,
#                 quantization_config=bnb_config,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 low_cpu_mem_usage=True,
#                 torch_dtype=torch.float16
#             )
#             _USE_QWEN_OK = True
#             print("✅ Qwen2.5-4B loaded successfully (4-bit)")

#         except Exception as e4:
#             print(f"⚠️ 4-bit failed: {str(e4)[:100]}")
#             print("🔄 Trying 8-bit quantization...")

#             # ===== TRY 2: 8-BIT QUANTIZATION (FALLBACK 1) =====
#             try:
#                 bnb_config = BitsAndBytesConfig(
#                     load_in_8bit=True,
#                     llm_int8_threshold=6.0,
#                     llm_int8_has_fp16_weight=False
#                 )

#                 tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
#                 qwen = AutoModelForCausalLM.from_pretrained(
#                     QWEN_MODEL,
#                     quantization_config=bnb_config,
#                     device_map="auto",
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True
#                 )
#                 _USE_QWEN_OK = True
#                 print("✅ Qwen2.5-4B loaded successfully (8-bit)")

#             except Exception as e8:
#                 print(f"⚠️ 8-bit failed: {str(e8)[:100]}")
#                 print("🔄 Trying float16 (no quantization)...")

#                 # ===== TRY 3: FLOAT16 NO QUANTIZATION (FALLBACK 2) =====
#                 tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
#                 qwen = AutoModelForCausalLM.from_pretrained(
#                     QWEN_MODEL,
#                     torch_dtype=torch.float16,
#                     device_map="auto",
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True
#                 )
#                 _USE_QWEN_OK = True
#                 print("✅ Qwen2.5-4B loaded successfully (float16)")

#         # ===== CHECK GPU MEMORY =====
#         if torch.cuda.is_available() and _USE_QWEN_OK:
#             allocated = torch.cuda.memory_allocated(0) / 1e9
#             reserved = torch.cuda.memory_reserved(0) / 1e9
#             total = torch.cuda.get_device_properties(0).total_memory / 1e9
#             print(f"📊 GPU Memory Status:")
#             print(f"   Allocated: {allocated:.2f} GB")
#             print(f"   Reserved:  {reserved:.2f} GB")
#             print(f"   Total:     {total:.2f} GB")
#             print(f"   Free:      {total - reserved:.2f} GB")

#     except Exception as e:
#         print(f"❌ All loading attempts failed: {str(e)[:150]}")
#         _USE_QWEN_OK = False
#         tok = None
#         qwen = None

# else:
#     print("⚠️ USE_QWEN=False, skipping model load")

# print(f"\n{'='*60}")
# print(f"🎯 Qwen2.5-4B Status: {'✅ READY' if _USE_QWEN_OK else '❌ NOT LOADED'}")
# print(f"{'='*60}\n")


# # ===== PHẦN 2: INFERENCE FUNCTION =====
# def infer_qwen(q, options, scene, labels, ocr, laws, temp):
#     """
#     Suy luận với Qwen2.5-4B-Instruct model

#     Args:
#         q: Câu hỏi
#         options: List các đáp án
#         scene: Mô tả cảnh
#         labels: Nhãn phát hiện
#         ocr: Text từ OCR
#         laws: Luật giao thông từ RAG
#         temp: Temporal facts

#     Returns:
#         (answer_letter, explanation)
#     """

#     # ===== CHECK MODEL STATUS =====
#     if not _USE_QWEN_OK or qwen is None or tok is None:
#         import random
#         fallback = random.choice(['A', 'B', 'C', 'D'])
#         return fallback, "⚠️ Model chưa được load"

#     try:
#         # ===== BUILD CONTEXT =====
#         ctx_parts = []

#         # Scene (prioritize)
#         if scene and scene.strip():
#             ctx_parts.append(f"Cảnh: {scene[:150]}")

#         # Labels (top 4)
#         if labels:
#             ctx_parts.append(f"Nhãn: {', '.join(labels[:4])}")

#         # Temporal facts (high priority)
#         if temp:
#             ctx_parts.append(f"Thời gian: {'; '.join(temp[:2])}")

#         # Laws (only if relevant)
#         if laws:
#             law_keywords = ['luật', 'quy định', 'phạt', 'điều', 'km/h', 'tốc độ', 'biển báo']
#             if any(kw in q.lower() for kw in law_keywords):
#                 ctx_parts.append(f"Luật: {laws[0][:100]}")

#         # OCR (only numbers)
#         if ocr:
#             ocr_nums = [x for x in ocr if re.search(r'\d', x)]
#             if ocr_nums:
#                 ctx_parts.append(f"OCR: {', '.join(ocr_nums[:3])}")

#         context = " | ".join(ctx_parts) if ctx_parts else "Không có thông tin bổ sung."

#         # ===== BUILD PROMPT (QWEN CHAT FORMAT) =====
#         opt_txt = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

#         messages = [
#             {
#                 "role": "system",
#                 "content": "Bạn là trợ lý lái xe thông minh. Trả lời câu hỏi bằng cách chọn đáp án đúng nhất."
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Câu hỏi: {q}\n\n"
#                     f"Các đáp án:\n{opt_txt}\n\n"
#                     f"Thông tin bổ sung:\n{context}\n\n"
#                     "Hãy trả lời theo format:\n"
#                     "- Dòng đầu: Chỉ ghi 1 chữ cái A, B, C hoặc D\n"
#                     "- Dòng sau: Lý do ngắn gọn\n\n"
#                     "Ví dụ:\n"
#                     "A\n"
#                     "Theo luật giao thông, đèn đỏ phải dừng lại."
#                 )
#             }
#         ]

#         # ===== APPLY CHAT TEMPLATE =====
#         prompt = tok.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         # ===== TOKENIZE =====
#         inp = tok(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=1024
#         ).to(qwen.device)

#         prompt_len = inp.input_ids.shape[1]

#         # ===== GENERATE =====
#         with torch.no_grad():
#             out = qwen.generate(
#                 **inp,
#                 max_new_tokens=80,
#                 do_sample=True,
#                 temperature=0.3,
#                 top_p=0.85,
#                 top_k=40,
#                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
#                 eos_token_id=tok.eos_token_id,
#                 repetition_penalty=1.1
#             )

#         # ===== DECODE =====
#         gen_tokens = out[0][prompt_len:]
#         response = tok.decode(gen_tokens, skip_special_tokens=True).strip()

#         # ===== CLEANUP MEMORY =====
#         del out, inp, gen_tokens
#         torch.cuda.empty_cache()

#         # ===== PARSE RESPONSE =====
#         if response:
#             lines = [l.strip() for l in response.split('\n') if l.strip()]

#             patterns = [
#                 r"^([ABCD])\s*$",
#                 r"^([ABCD])\s*[\.:\-\)]",
#                 r"^đáp\s*án[:\s]*([ABCD])\b",
#                 r"^chọn[:\s]*([ABCD])\b",
#                 r"^trả\s*lời[:\s]*([ABCD])\b",
#                 r"^([ABCD])\s+\w+",
#                 r"\b([ABCD])\b"
#             ]

#             for line in lines[:3]:
#                 for pattern in patterns:
#                     match = re.search(pattern, line, re.IGNORECASE)
#                     if match:
#                         ans_letter = match.group(1).upper()
#                         ans_idx = ord(ans_letter) - 65

#                         if 0 <= ans_idx < len(options):
#                             expl_lines = [l for l in lines[1:4] if l and not re.match(r'^[ABCD]\s*$', l)]
#                             expl = " ".join(expl_lines[:2]) if expl_lines else f"Chọn {ans_letter}"
#                             return ans_letter, expl[:200]

#         # ===== RETRY WITH SIMPLE PROMPT =====
#         print(f"⚠️ Parse failed, retrying...")

#         simple_messages = [
#             {
#                 "role": "system",
#                 "content": "Trả lời bằng 1 chữ cái duy nhất: A, B, C hoặc D"
#             },
#             {
#                 "role": "user",
#                 "content": f"Câu hỏi: {q}\n\n{opt_txt}\n\nTrả lời:"
#             }
#         ]

#         simple_prompt = tok.apply_chat_template(
#             simple_messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inp2 = tok(simple_prompt, return_tensors="pt", truncation=True, max_length=600).to(qwen.device)
#         prompt_len2 = inp2.input_ids.shape[1]

#         with torch.no_grad():
#             out2 = qwen.generate(
#                 **inp2,
#                 max_new_tokens=5,
#                 do_sample=False,
#                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
#                 eos_token_id=tok.eos_token_id
#             )

#         gen_tokens2 = out2[0][prompt_len2:]
#         response2 = tok.decode(gen_tokens2, skip_special_tokens=True).strip()

#         del out2, inp2, gen_tokens2
#         torch.cuda.empty_cache()

#         match2 = re.search(r'\b([ABCD])\b', response2, re.IGNORECASE)
#         if match2:
#             ans2 = match2.group(1).upper()
#             if 0 <= ord(ans2) - 65 < len(options):
#                 return ans2, "⚠️ Retry thành công"

#         # ===== RANDOM FALLBACK =====
#         import random
#         fallback = random.choice(['A', 'B', 'C', 'D'])
#         return fallback, "⚠️ Không parse được, chọn ngẫu nhiên"

#     except torch.cuda.OutOfMemoryError:
#         print("❌ GPU Out of Memory!")
#         torch.cuda.empty_cache()
#         gc.collect()
#         import random
#         return random.choice(['A', 'B', 'C', 'D']), "❌ OOM"

#     except Exception as e:
#         print(f"❌ Error: {str(e)[:150]}")
#         torch.cuda.empty_cache()
#         import random
#         return random.choice(['A', 'B', 'C', 'D']), f"❌ Error: {str(e)[:100]}"


# print("✅ Qwen2.5-4B inference function ready")

# ==================== FIXED QWEN INFERENCE - CELL 10 ====================

def infer_qwen(q, options, scene, labels, ocr, laws, temp):
    """
    Suy luận với Qwen - FIXED VERSION
    
    Cải tiến:
    - Rút gọn context (max 400 chars)
    - Giảm temperature (0.15)
    - Giới hạn max_new_tokens (25)
    - Prompt đơn giản hơn
    - Parse pattern chặt chẽ hơn
    """
    
    # ===== CHECK MODEL =====
    if not _USE_QWEN_OK or qwen is None or tok is None:
        import random
        return random.choice(['A', 'B', 'C', 'D']), "⚠️ Model chưa load"
    
    try:
        # ===== BUILD COMPACT CONTEXT =====
        ctx_parts = []
        
        # 1. Scene (PRIORITY 1 - Max 100 chars)
        if scene and scene.strip():
            scene_short = scene[:100].strip()
            ctx_parts.append(f"Cảnh: {scene_short}")
        
        # 2. Temporal (PRIORITY 2 - Only first fact)
        if temp:
            ctx_parts.append(f"Thời gian: {temp[0][:50]}")
        
        # 3. Labels (Top 3 only)
        if labels:
            top_labels = labels[:3]
            ctx_parts.append(f"Nhãn: {', '.join(top_labels)}")
        
        # 4. OCR Numbers (ONLY if question has number keywords)
        if ocr and any(kw in q.lower() for kw in ['tốc độ', 'km/h', 'số', 'biển']):
            ocr_nums = [x for x in ocr if re.search(r'\d', x)]
            if ocr_nums:
                ctx_parts.append(f"Số: {ocr_nums[0]}")
        
        # 5. Laws (ONLY if question mentions law)
        if laws and any(kw in q.lower() for kw in ['luật', 'quy định', 'phạt']):
            ctx_parts.append(f"Luật: {laws[0][:80]}")
        
        # Combine context - MAX 400 chars
        context = " | ".join(ctx_parts)[:400]
        
        # ===== SIMPLE PROMPT (Qwen-specific format) =====
        opt_txt = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        
        # Qwen loves concise prompts
        system_msg = "Bạn là trợ lý lái xe. Trả lời ĐÚNG format: chữ cái A/B/C/D, xuống dòng, lý do ngắn."
        
        user_msg = (
            f"Câu hỏi: {q}\n\n"
            f"Đáp án:\n{opt_txt}\n\n"
            f"Thông tin: {context}\n\n"
            "Trả lời (chỉ 1 chữ cái, xuống dòng, lý do):\n"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # ===== APPLY CHAT TEMPLATE =====
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # ===== TOKENIZE (Max 800 tokens) =====
        inp = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800  # Giảm từ 1024
        ).to(qwen.device)
        
        prompt_len = inp.input_ids.shape[1]
        
        # ===== GENERATE (STRICTER PARAMS) =====
        with torch.no_grad():
            out = qwen.generate(
                **inp,
                max_new_tokens=25,       # Giảm từ 80 → 25
                do_sample=True,
                temperature=0.15,        # Giảm từ 0.3 → 0.15
                top_p=0.9,               # Tăng từ 0.85 → 0.9 (focus hơn)
                top_k=20,                # Giảm từ 40 → 20
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05  # Giảm từ 1.1
            )
        
        # ===== DECODE =====
        gen_tokens = out[0][prompt_len:]
        response = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # ===== CLEANUP =====
        del out, inp, gen_tokens
        torch.cuda.empty_cache()
        
        # ===== ENHANCED PARSING (9 PATTERNS) =====
        if response:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            
            # Patterns từ chặt → lỏng
            patterns = [
                r"^([ABCD])\s*$",                          # "A"
                r"^([ABCD])\s*[\.\:\-\)]",                 # "A." "A:" "A-"
                r"^([ABCD])\s*\n",                         # "A\n"
                r"^đáp\s*án[:\s]*([ABCD])\b",              # "Đáp án A"
                r"^chọn[:\s]*([ABCD])\b",                  # "Chọn A"
                r"^câu\s*trả\s*lời[:\s]*([ABCD])\b",       # "Câu trả lời: A"
                r"\(([ABCD])\)",                           # "(A)"
                r"^([ABCD])\s+[^\n]{5,}",                  # "A Lý do..."
                r"\b([ABCD])\b"                            # Fallback: any "A"
            ]
            
            # Search in first 2 lines only
            for line in lines[:2]:
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        ans_letter = match.group(1).upper()
                        ans_idx = ord(ans_letter) - 65
                        
                        # Validate
                        if 0 <= ans_idx < len(options):
                            # Extract explanation
                            expl_lines = [l for l in lines[1:3] 
                                         if l and not re.match(r'^[ABCD]\s*[\.\:\-\)]?\s*$', l)]
                            expl = " ".join(expl_lines[:1]) if expl_lines else f"Chọn {ans_letter}"
                            
                            return ans_letter, expl[:150]
        
        # ===== RETRY WITH ULTRA-SIMPLE PROMPT =====
        print(f"⚠️ Parse failed ('{response[:50]}...'), retry ultra-simple...")
        
        simple_prompt = f"{q}\n\n{opt_txt}\n\nChỉ trả lời 1 chữ cái A, B, C hoặc D:"
        
        inp2 = tok(
            simple_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400  # Rất ngắn
        ).to(qwen.device)
        
        prompt_len2 = inp2.input_ids.shape[1]
        
        with torch.no_grad():
            out2 = qwen.generate(
                **inp2,
                max_new_tokens=3,      # Chỉ cho phép "A\n"
                do_sample=False,       # Greedy = stable nhất
                temperature=0.1,       # Cực thấp
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        gen_tokens2 = out2[0][prompt_len2:]
        response2 = tok.decode(gen_tokens2, skip_special_tokens=True).strip()
        
        del out2, inp2, gen_tokens2
        torch.cuda.empty_cache()
        
        # Parse retry
        match2 = re.search(r'\b([ABCD])\b', response2, re.IGNORECASE)
        if match2:
            ans2 = match2.group(1).upper()
            if 0 <= ord(ans2) - 65 < len(options):
                return ans2, "✅ Retry thành công"
        
        # ===== FINAL FALLBACK: HEURISTIC =====
        print(f"⚠️ Retry failed ('{response2}'), using heuristic...")
        
        # Heuristic: Chọn đáp án dài nhất (thường đúng trong MCQ)
        longest_idx = max(range(len(options)), key=lambda i: len(options[i]))
        longest_letter = chr(65 + longest_idx)
        
        return longest_letter, "⚠️ Heuristic: chọn đáp án chi tiết nhất"
        
    except torch.cuda.OutOfMemoryError:
        print("❌ OOM!")
        torch.cuda.empty_cache()
        gc.collect()
        import random
        return random.choice(['A', 'B', 'C', 'D']), "❌ OOM"
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        torch.cuda.empty_cache()
        import random
        return random.choice(['A', 'B', 'C', 'D']), f"❌ {str(e)[:80]}"


print("✅ Fixed Qwen inference loaded")


# ===== TEST FUNCTION =====
def test_qwen_parsing():
    """Test parsing với nhiều format output"""
    
    test_cases = [
        "A\nĐèn đỏ phải dừng",
        "A. Dừng lại theo luật",
        "Đáp án: A\nVì an toàn",
        "Chọn A",
        "(A) Dừng xe",
        "A Theo quy định phải dừng",
        "Câu trả lời là A",
        "B",
        "Tôi nghĩ nên chọn C vì..."
    ]
    
    print("\n🧪 TESTING PARSING:")
    for i, resp in enumerate(test_cases, 1):
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        
        patterns = [
            r"^([ABCD])\s*$",
            r"^([ABCD])\s*[\.\:\-\)]",
            r"^đáp\s*án[:\s]*([ABCD])\b",
            r"^chọn[:\s]*([ABCD])\b",
            r"\(([ABCD])\)",
            r"^([ABCD])\s+[^\n]{5,}",
            r"\b([ABCD])\b"
        ]
        
        found = False
        for line in lines[:2]:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    ans = match.group(1).upper()
                    print(f"   {i}. '{resp[:30]}...' → {ans} ✅")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"   {i}. '{resp[:30]}...' → ❌ FAILED")

# Uncomment để test:
# test_qwen_parsing()
