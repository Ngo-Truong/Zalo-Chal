# ==================== CELL 10.5: DEBUG QWEN STATUS ====================
print("\n🔍 CHECKING QWEN STATUS:")
print(f"   _USE_QWEN_OK exists: {'_USE_QWEN_OK' in globals()}")
if '_USE_QWEN_OK' in globals():
    print(f"   _USE_QWEN_OK value: {_USE_QWEN_OK}")
else:
    print(f"   _USE_QWEN_OK value: NOT DEFINED")

print(f"   qwen exists: {'qwen' in globals()}")
if 'qwen' in globals():
    print(f"   qwen value: {qwen}")
else:
    print(f"   qwen value: NOT DEFINED")

print(f"   tok exists: {'tok' in globals()}")
if 'tok' in globals():
    print(f"   tok value: {tok}")
else:
    print(f"   tok value: NOT DEFINED")

print(f"   infer_qwen exists: {'infer_qwen' in globals()}")

# Nếu tất cả OK
if all([
    '_USE_QWEN_OK' in globals(),
    globals().get('_USE_QWEN_OK') == True,
    'qwen' in globals(),
    globals().get('qwen') is not None,
    'tok' in globals(),
    globals().get('tok') is not None
]):
    print("\n✅ QWEN IS READY!")
else:
    print("\n❌ QWEN NOT READY - Will use rule-based")
```

---

## 📋 **Thứ tự chạy lại**

1. **CELL 9** (Config - có `USE_QWEN = True`)
2. **CELL 10** (Load Qwen model)
3. **CELL 10.5** (Debug - cell mới thêm)
4. **CELL WRAPPER** (Code đã update)
5. **CELL 11-13** (Runner)

---

## 🎯 **Kết quả mong đợi**

### **Nếu Qwen load thành công:**
```
🔍 CHECKING QWEN STATUS:
   _USE_QWEN_OK exists: True
   _USE_QWEN_OK value: True
   qwen exists: True
   qwen value: Qwen2ForCausalLM(...)
   tok exists: True
   tok value: QWenTokenizer(...)
   infer_qwen exists: True

✅ QWEN IS READY!
```

**Wrapper sẽ dùng Qwen** ✅

---

### **Nếu Qwen không load:**
```
🔍 CHECKING QWEN STATUS:
   _USE_QWEN_OK exists: True
   _USE_QWEN_OK value: False  ← LÝ DO
   ...

❌ QWEN NOT READY - Will use rule-based
