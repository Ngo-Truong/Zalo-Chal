# ==================== CELL WRAPPER - FIXED VERSION ====================
# Chạy cell này SAU CELL 10 (infer_qwen)

def infer_phi(q, options, scene, labels, ocr, laws, temp):
    """
    Wrapper function - Tương thích với code cũ
    
    Logic:
    1. Kiểm tra xem Qwen có sẵn sàng không
    2. Nếu có → dùng infer_qwen()
    3. Nếu không → dùng rule-based fallback
    
    Args:
        q: Câu hỏi
        options: List các đáp án
        scene: Mô tả cảnh
        labels: Nhãn phát hiện
        ocr: Text từ OCR
        laws: Luật giao thông từ RAG
        temp: Temporal facts
    
    Returns:
        (answer_letter, explanation)
    """
    
    # ===== CHECK 1: Qwen có sẵn sàng không? =====
    qwen_ready = (
        '_USE_QWEN_OK' in globals() and 
        _USE_QWEN_OK and 
        'qwen' in globals() and 
        qwen is not None and
        'tok' in globals() and
        tok is not None
    )
    
    if qwen_ready:
        # Dùng Qwen
        try:
            return infer_qwen(q, options, scene, labels, ocr, laws, temp)
        except Exception as e:
            print(f"⚠️ Qwen failed: {str(e)[:80]}, fallback to rules")
            # Nếu Qwen lỗi → fallback
            return rule_based_inference(q, options, scene, labels, ocr, laws, temp)
    else:
        # Không có Qwen → dùng rules
        print("⚠️ Qwen not available, using rule-based")
        return rule_based_inference(q, options, scene, labels, ocr, laws, temp)


def rule_based_inference(q, options, scene, labels, ocr, laws, temp):
    """
    Rule-based inference - Fallback khi không có model
    
    Ưu tiên:
    1. Traffic lights (đèn đỏ/xanh/vàng)
    2. Speed signs (tốc độ từ OCR)
    3. Safety keywords (an toàn, nên, phải)
    4. Temporal context (thời gian/thời tiết)
    5. Laws (từ RAG)
    6. Scene matching
    7. Heuristic (chọn đáp án dài nhất)
    """
    import random
    
    q_lower = q.lower()
    
    # ===== RULE 1: TRAFFIC LIGHT (Ưu tiên cao nhất) =====
    if any(kw in q_lower for kw in ['đèn đỏ', 'đèn xanh', 'đèn vàng', 'tín hiệu đèn', 'đèn giao thông']):
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            # Đèn đỏ → Dừng
            if 'đèn đỏ' in q_lower and any(kw in opt_lower for kw in ['dừng', 'stop', 'không đi']):
                return chr(65+i), "Đèn đỏ phải dừng lại theo luật"
            # Đèn xanh → Đi
            if 'đèn xanh' in q_lower and any(kw in opt_lower for kw in ['đi', 'tiếp tục', 'cho phép']):
                return chr(65+i), "Đèn xanh được phép đi"
            # Đèn vàng → Giảm tốc
            if 'đèn vàng' in q_lower and any(kw in opt_lower for kw in ['giảm', 'chậm', 'chuẩn bị']):
                return chr(65+i), "Đèn vàng cần giảm tốc độ"
    
    # ===== RULE 2: SPEED SIGNS (từ OCR) =====
    if any(kw in q_lower for kw in ['tốc độ', 'km/h', 'giới hạn', 'tốc', 'biển báo']):
        # Tìm số trong OCR
        speed_nums = []
        if ocr:
            for text in ocr:
                nums = re.findall(r'\d+', text)
                speed_nums.extend(nums)
        
        # Match với options
        if speed_nums:
            for i, opt in enumerate(options):
                if speed_nums[0] in opt:
                    return chr(65+i), f"Phát hiện biển báo {speed_nums[0]} km/h"
    
    # ===== RULE 3: SAFETY QUESTIONS (an toàn > rủi ro) =====
    if any(kw in q_lower for kw in ['an toàn', 'nên', 'phải', 'cần', 'tốt nhất']):
        # Tìm đáp án có keywords an toàn
        safety_keywords = [
            'giảm tốc', 'chú ý', 'quan sát', 'cảnh giác', 
            'dừng lại', 'chậm', 'nhường', 'cẩn thận', 
            'kiểm tra', 'đảm bảo', 'an toàn'
        ]
        
        best_idx = -1
        best_score = 0
        
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            score = sum(1 for kw in safety_keywords if kw in opt_lower)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0:
            return chr(65+best_idx), "Chọn phương án an toàn nhất"
    
    # ===== RULE 4: TIME/WEATHER (từ temporal) =====
    if temp:
        time_info = temp[0].lower()
        
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            
            # Match temporal keywords
            if 'ban ngày' in time_info and 'ngày' in opt_lower:
                return chr(65+i), "Dựa vào thời gian trong video"
            if 'ban đêm' in time_info and 'đêm' in opt_lower:
                return chr(65+i), "Dựa vào thời gian trong video"
            if 'mưa' in time_info and 'mưa' in opt_lower:
                return chr(65+i), "Dựa vào thời tiết (mưa)"
            if 'nắng' in time_info and 'nắng' in opt_lower:
                return chr(65+i), "Dựa vào thời tiết (nắng)"
    
    # ===== RULE 5: COUNTING OBJECTS (từ labels) =====
    if any(kw in q_lower for kw in ['bao nhiêu', 'số lượng', 'mấy', 'có', 'đếm']):
        if 'xe' in q_lower:
            vehicle_count = sum(1 for l in labels if any(v in l.lower() 
                              for v in ['car', 'truck', 'vehicle', 'bus', 'motorbike']))
            
            for i, opt in enumerate(options):
                if str(vehicle_count) in opt:
                    return chr(65+i), f"Đếm được {vehicle_count} xe"
        
        if 'người' in q_lower:
            person_count = sum(1 for l in labels if any(p in l.lower() 
                             for p in ['person', 'pedestrian', 'people']))
            
            for i, opt in enumerate(options):
                if str(person_count) in opt:
                    return chr(65+i), f"Đếm được {person_count} người"
    
    # ===== RULE 6: TRAFFIC SIGNS (từ labels) =====
    if 'biển báo' in q_lower and labels:
        sign_labels = [l for l in labels if any(kw in l.lower() 
                      for kw in ['sign', 'traffic', 'signal', 'board'])]
        
        if sign_labels:
            # Chọn đáp án match nhiều nhất
            best_idx = 0
            best_count = 0
            
            for i, opt in enumerate(options):
                opt_lower = opt.lower()
                count = sum(1 for label in sign_labels if any(
                    word in opt_lower for word in label.lower().split()
                ))
                
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_count > 0:
                return chr(65+best_idx), f"Phát hiện {len(sign_labels)} biển báo phù hợp"
    
    # ===== RULE 7: LAWS (từ RAG) =====
    if laws:
        law_text = laws[0].lower()
        
        # Tìm đáp án có nhiều từ chung với law nhất
        best_idx = 0
        best_score = 0
        
        for i, opt in enumerate(options):
            opt_words = set(opt.lower().split())
            law_words = set(law_text.split())
            common_words = opt_words & law_words
            
            # Loại bỏ stopwords
            stopwords = {'của', 'và', 'các', 'có', 'trong', 'là', 'cho', 'với'}
            common_words -= stopwords
            
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_score >= 2:  # Ít nhất 2 từ chung
            return chr(65+best_idx), "Dựa vào quy định giao thông"
    
    # ===== RULE 8: SCENE MATCHING =====
    if scene:
        scene_lower = scene.lower()
        scene_words = set(scene_lower.split())
        
        best_idx = 0
        best_score = 0
        
        for i, opt in enumerate(options):
            opt_words = set(opt.lower().split())
            common = len(scene_words & opt_words)
            
            if common > best_score:
                best_score = common
                best_idx = i
        
        if best_score >= 2:
            return chr(65+best_idx), "Dựa vào ngữ cảnh của cảnh"
    
    # ===== RULE 9: HEURISTIC FALLBACK =====
    # Trong MCQ, đáp án dài thường chứa nhiều thông tin → cao hơn
    longest_idx = max(range(len(options)), key=lambda i: len(options[i]))
    
    # Kiểm tra xem đáp án dài nhất có keywords an toàn không
    longest_opt = options[longest_idx].lower()
    has_safety = any(kw in longest_opt for kw in 
                    ['an toàn', 'cẩn thận', 'chú ý', 'giảm', 'dừng'])
    
    if has_safety:
        return chr(65+longest_idx), "Heuristic: đáp án chi tiết + an toàn"
    else:
        return chr(65+longest_idx), "Heuristic: đáp án chi tiết nhất"


# ===== VALIDATION =====
print("✅ Wrapper function loaded:")
print("   - infer_phi() → Router to Qwen or Rules")
print("   - rule_based_inference() → 9 rules")

# Check status
if '_USE_QWEN_OK' in globals() and _USE_QWEN_OK:
    print("   - Qwen status: ✅ READY")
else:
    print("   - Qwen status: ❌ NOT LOADED (will use rules)")


# ===== TEST FUNCTION =====
def test_wrapper():
    """Test wrapper với nhiều loại câu hỏi"""
    
    test_cases = [
        {
            "q": "Xe nên làm gì khi gặp đèn đỏ?",
            "opts": ["Tiếp tục đi", "Dừng lại", "Tăng tốc", "Rẽ phải"],
            "expected": "B"
        },
        {
            "q": "Tốc độ tối đa cho phép là bao nhiêu?",
            "opts": ["40 km/h", "60 km/h", "80 km/h", "100 km/h"],
            "expected": None  # Phụ thuộc OCR
        },
        {
            "q": "Biển báo này có ý nghĩa gì?",
            "opts": ["Cấm dừng", "Cấm đỗ", "Hạn chế tốc độ", "Nguy hiểm"],
            "expected": None  # Phụ thuộc labels
        }
    ]
    
    print("\n🧪 TESTING WRAPPER:")
    
    for i, test in enumerate(test_cases, 1):
        ans, expl = infer_phi(
            test["q"], 
            test["opts"],
            scene="",
            labels=[],
            ocr=[],
            laws=[],
            temp=[]
        )
        
        result = "✅" if (test["expected"] is None or ans == test["expected"]) else "❌"
        
        print(f"\n{i}. {test['q'][:50]}...")
        print(f"   Answer: {ans} - {test['opts'][ord(ans)-65]}")
        print(f"   Explain: {expl[:80]}...")
        print(f"   Result: {result}")

# Uncomment để test:
# test_wrapper()
