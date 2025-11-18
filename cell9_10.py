# ==================== CELL 9-10: CONFIG + LOAD QWEN + INFERENCE ====================

import torch
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== CONFIGURATION =====
USE_QWEN = True  # Đặt False để bỏ qua Qwen

# ===== MODEL SELECTION (Chọn 1 model) =====
# QWEN_MODEL = "Qwen/Qwen2.5-4B-Instruct"    # Nặng (~8GB VRAM)
# QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"    # Vừa (~6GB VRAM)
# QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Nhẹ (~4GB VRAM)
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"    # Siêu nhẹ (~2GB VRAM) - KHUYẾN NGHỊ TEST

print(f"✅ Configuration: USE_QWEN={USE_QWEN}, MODEL={QWEN_MODEL}")

# ==================== HELPER FUNCTIONS ====================

def scene_text_for_video(video_key, scene_map):
    """Lấy mô tả scene cho video"""
    if not scene_map or not isinstance(scene_map, dict):
        return ""
    
    # Exact match
    if video_key in scene_map:
        scene = scene_map[video_key]
        if isinstance(scene, str) and scene.strip():
            return scene.strip()
    
    # Case-insensitive match
    video_key_lower = video_key.lower()
    for k, v in scene_map.items():
        if k.lower() == video_key_lower:
            if isinstance(v, str) and v.strip():
                return v.strip()
    
    # Partial match
    for k, v in scene_map.items():
        if video_key in k or k in video_key:
            if isinstance(v, str) and v.strip():
                return v.strip()
    
    return ""


def rule_engine_temporal(video_key, temporal_map):
    """Trích xuất temporal facts từ video"""
    if not temporal_map or not isinstance(temporal_map, dict):
        return []
    
    temporal_facts = []
    
    # Exact match
    if video_key in temporal_map:
        temp_data = temporal_map[video_key]
        temporal_facts.extend(_parse_temporal_data(temp_data))
    
    # Case-insensitive match
    if not temporal_facts:
        video_key_lower = video_key.lower()
        for k, v in temporal_map.items():
            if k.lower() == video_key_lower:
                temporal_facts.extend(_parse_temporal_data(v))
                break
    
    # Partial match
    if not temporal_facts:
        for k, v in temporal_map.items():
            if video_key in k or k in video_key:
                temporal_facts.extend(_parse_temporal_data(v))
                break
    
    return temporal_facts


def _parse_temporal_data(temp_data):
    """Parse temporal data từ nhiều format"""
    facts = []
    
    if isinstance(temp_data, str):
        if temp_data.strip():
            facts.append(temp_data.strip())
    
    elif isinstance(temp_data, list):
        for item in temp_data:
            if isinstance(item, str) and item.strip():
                facts.append(item.strip())
            elif isinstance(item, dict):
                for key in ['time', 'weather', 'condition', 'description', 'text']:
                    if key in item and isinstance(item[key], str):
                        facts.append(item[key].strip())
    
    elif isinstance(temp_data, dict):
        common_keys = ['time', 'weather', 'condition', 'time_of_day',
                      'weather_condition', 'temporal', 'description', 'text']
        
        for key in common_keys:
            if key in temp_data:
                value = temp_data[key]
                if isinstance(value, str) and value.strip():
                    facts.append(value.strip())
                elif isinstance(value, list):
                    facts.extend(_parse_temporal_data(value))
        
        if not facts:
            for v in temp_data.values():
                if isinstance(v, str) and v.strip():
                    facts.append(v.strip())
    
    return facts


def extract_video_key(question_id):
    """Trích xuất video key từ question ID"""
    match = re.search(r'_(\d+)', question_id)
    if match:
        num = match.group(1)
        keys_to_try = [
            f"video{num}",
            f"video{int(num)}",
            num,
            str(int(num))
        ]
        return keys_to_try
    return [question_id]


def get_video_context(question_id, scene_map, temporal_map):
    """Lấy toàn bộ context từ video"""
    video_keys = extract_video_key(question_id)
    scene_text = ""
    temporal_facts = []
    
    for vkey in video_keys:
        if not scene_text:
            scene_text = scene_text_for_video(vkey, scene_map)
        if not temporal_facts:
            temporal_facts = rule_engine_temporal(vkey, temporal_map)
        if scene_text and temporal_facts:
            break
    
    return scene_text, temporal_facts


print("✅ Helper functions loaded")

# ==================== LOAD QWEN MODEL ====================

_USE_QWEN_OK = False
tok = None
qwen = None

if USE_QWEN:
    print(f"\n🔄 Loading Qwen model: {QWEN_MODEL}")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_vram = total_vram - (torch.cuda.memory_allocated(0) / 1e9)
        print(f"✅ GPU: {gpu_name}")
        print(f"   Total VRAM: {total_vram:.2f} GB")
        print(f"   Free VRAM: {free_vram:.2f} GB")
        
        if free_vram < 3:
            print("⚠️ WARNING: Free VRAM < 3GB, cleaning up...")
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("❌ NO GPU! Enable GPU: Runtime > Change runtime type > T4 GPU")
    
    print("="*70)
    
    try:
        # Try 4-bit quantization
        try:
            print("\n🔹 Attempting 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
            qwen = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            _USE_QWEN_OK = True
            print("✅ SUCCESS: 4-bit quantization")
            
        except Exception as e4:
            print(f"❌ 4-bit failed: {str(e4)[:120]}")
            print("\n🔹 Attempting 8-bit quantization...")
            
            # Try 8-bit quantization
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                
                tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
                qwen = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                _USE_QWEN_OK = True
                print("✅ SUCCESS: 8-bit quantization")
                
            except Exception as e8:
                print(f"❌ 8-bit failed: {str(e8)[:120]}")
                print("\n🔹 Attempting float16...")
                
                # Try float16
                tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
                qwen = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                _USE_QWEN_OK = True
                print("✅ SUCCESS: float16")
        
        # Check final status
        if _USE_QWEN_OK and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n📊 GPU Memory After Loading:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved:  {reserved:.2f} GB")
            print(f"   Free:      {total - reserved:.2f} GB")
    
    except Exception as e:
        print(f"\n❌ ALL LOADING FAILED!")
        print(f"   Error: {str(e)[:150]}")
        print("\n💡 Suggestions:")
        print("   1. Enable GPU: Runtime > Change runtime type > T4 GPU")
        print("   2. Use lighter model: QWEN_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'")
        print("   3. Restart runtime and try again")
        
        _USE_QWEN_OK = False
        tok = None
        qwen = None

else:
    print("⚠️ USE_QWEN=False, skipping model load")

# Final status
print("\n" + "="*70)
if _USE_QWEN_OK:
    print("🎉 QWEN MODEL READY!")
    print(f"   Model: {QWEN_MODEL}")
    print(f"   Status: _USE_QWEN_OK={_USE_QWEN_OK}")
else:
    print("❌ QWEN NOT LOADED - Will use rule-based inference")
print("="*70)

# ==================== INFERENCE FUNCTION ====================

def infer_qwen(q, options, scene, labels, ocr, laws, temp):
    """
    Suy luận với Qwen - FIXED VERSION
    Rút gọn context, giảm temperature, parse chặt chẽ
    """
    
    # Check model
    if not _USE_QWEN_OK or qwen is None or tok is None:
        import random
        return random.choice(['A', 'B', 'C', 'D']), "⚠️ Model chưa load"
    
    try:
        # Build compact context (max 400 chars)
        ctx_parts = []
        
        if scene and scene.strip():
            ctx_parts.append(f"Cảnh: {scene[:100].strip()}")
        
        if temp:
            ctx_parts.append(f"Thời gian: {temp[0][:50]}")
        
        if labels:
            ctx_parts.append(f"Nhãn: {', '.join(labels[:3])}")
        
        if ocr and any(kw in q.lower() for kw in ['tốc độ', 'km/h', 'số', 'biển']):
            ocr_nums = [x for x in ocr if re.search(r'\d', x)]
            if ocr_nums:
                ctx_parts.append(f"Số: {ocr_nums[0]}")
        
        if laws and any(kw in q.lower() for kw in ['luật', 'quy định', 'phạt']):
            ctx_parts.append(f"Luật: {laws[0][:80]}")
        
        context = " | ".join(ctx_parts)[:400]
        
        # Simple prompt
        opt_txt = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        
        system_msg = "Bạn là trợ lý lái xe. Trả lời ĐÚNG format: chữ cái A/B/C/D, xuống dòng, lý do ngắn."
        user_msg = f"Câu hỏi: {q}\n\nĐáp án:\n{opt_txt}\n\nThông tin: {context}\n\nTrả lời:"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Apply chat template
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=800).to(qwen.device)
        prompt_len = inp.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            out = qwen.generate(
                **inp,
                max_new_tokens=25,
                do_sample=True,
                temperature=0.15,
                top_p=0.9,
                top_k=20,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05
            )
        
        # Decode
        gen_tokens = out[0][prompt_len:]
        response = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Cleanup
        del out, inp, gen_tokens
        torch.cuda.empty_cache()
        
        # Parse (9 patterns)
        if response:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            
            patterns = [
                r"^([ABCD])\s*$",
                r"^([ABCD])\s*[\.\:\-\)]",
                r"^([ABCD])\s*\n",
                r"^đáp\s*án[:\s]*([ABCD])\b",
                r"^chọn[:\s]*([ABCD])\b",
                r"^câu\s*trả\s*lời[:\s]*([ABCD])\b",
                r"\(([ABCD])\)",
                r"^([ABCD])\s+[^\n]{5,}",
                r"\b([ABCD])\b"
            ]
            
            for line in lines[:2]:
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        ans_letter = match.group(1).upper()
                        ans_idx = ord(ans_letter) - 65
                        
                        if 0 <= ans_idx < len(options):
                            expl_lines = [l for l in lines[1:3] 
                                         if l and not re.match(r'^[ABCD]\s*[\.\:\-\)]?\s*$', l)]
                            expl = " ".join(expl_lines[:1]) if expl_lines else f"Chọn {ans_letter}"
                            return ans_letter, expl[:150]
        
        # Retry ultra-simple
        print(f"⚠️ Parse failed, retry...")
        
        simple_prompt = f"{q}\n\n{opt_txt}\n\nChỉ trả lời 1 chữ cái:"
        inp2 = tok(simple_prompt, return_tensors="pt", truncation=True, max_length=400).to(qwen.device)
        prompt_len2 = inp2.input_ids.shape[1]
        
        with torch.no_grad():
            out2 = qwen.generate(
                **inp2,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        gen_tokens2 = out2[0][prompt_len2:]
        response2 = tok.decode(gen_tokens2, skip_special_tokens=True).strip()
        
        del out2, inp2, gen_tokens2
        torch.cuda.empty_cache()
        
        match2 = re.search(r'\b([ABCD])\b', response2, re.IGNORECASE)
        if match2:
            ans2 = match2.group(1).upper()
            if 0 <= ord(ans2) - 65 < len(options):
                return ans2, "✅ Retry OK"
        
        # Heuristic fallback
        longest_idx = max(range(len(options)), key=lambda i: len(options[i]))
        return chr(65 + longest_idx), "⚠️ Heuristic"
        
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


print("✅ Qwen inference function ready")
