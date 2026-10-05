# Zalo AI Challenge 2025 — Dashcam RoadBuddy

Notebook `testfinal2.ipynb` là file triển khai pipeline RoadBuddy cho bài toán Dashcam trong khuôn khổ **Zalo AI Challenge 2025**. Notebook được thiết kế để chạy trực tiếp trên **Google Colab**. Repository này chỉ chứa một notebook; dữ liệu video, trọng số mô hình và tài liệu luật giao thông cần được người chạy cung cấp từ Google Drive hoặc tải lên phiên Colab.

## Mục tiêu

RoadBuddy xử lý video hành trình (dashcam) cùng bộ câu hỏi trắc nghiệm liên quan đến tình huống giao thông. Pipeline khai thác hình ảnh video để nhận diện các đối tượng giao thông như biển báo và đèn tín hiệu, tóm tắt thông tin theo từng khoảng thời gian, kết hợp với tri thức từ tài liệu luật để tạo ngữ cảnh cho bước trả lời câu hỏi. Kết quả cuối được xuất thành CSV gồm mã câu hỏi và đáp án.

Notebook hiện có ba phần chính:

| Phần | Nhiệm vụ | Đầu ra chính |
| --- | --- | --- |
| Module 1 | Lấy mẫu frame video và dùng YOLO phát hiện đối tượng | `metadata.json`, `detections.csv`, `scene_cache.jsonl` |
| Module 2 | Gom detection thành mô tả cảnh; chuyển PDF luật thành văn bản và xây kho truy xuất RAG | `scene_description.jsonl`, `chunks.jsonl`, `faiss.index` |
| Module 3 | Ghép câu hỏi với thông tin video và các đoạn luật truy xuất được, gọi hàm suy luận để tạo đáp án | `answers_public_test.csv` |

## Cấu trúc notebook

Toàn bộ mã được lưu trong `testfinal2.ipynb`; không cần cài đặt hoặc chạy một ứng dụng riêng trên máy tính. Những thư mục `roadbuddy/runtime` và `roadbuddy/kb/index` được notebook tạo ra trong phiên Colab khi chạy.

### Module 1 — Phân tích video dashcam

Module này quét thư mục video, đọc lần lượt từng video bằng OpenCV, lấy một frame sau mỗi `FRAME_STEP` frame, rồi chạy YOLO với ngưỡng tin cậy `CONF_TH`. Theo cấu hình mặc định, `FRAME_STEP = 3` và `CONF_TH = 0.35`.

Các biến cần cấu hình trong cell **CONFIG MODULE 1**:

- `VIDEO_ROOT`: thư mục chứa video đầu vào. Mặc định đang là `/content/drive/MyDrive/videos`.
- `WEIGHTS`: đường dẫn tới checkpoint YOLO đã huấn luyện, ví dụ `best.pt`. Mặc định đang là `/content/runs/detect/train/weights/best.pt`.
- `MAX_VIDEOS`: đặt `None` để xử lý tất cả video; đặt số nguyên để giới hạn số video khi chạy thử.

Module hỗ trợ các định dạng `.mp4`, `.avi`, `.mov` và `.mkv`. Kết quả detections chứa tên lớp, độ tin cậy, tọa độ bounding box, số frame và thời điểm tương ứng trong video.

### Module 2 — Mô tả cảnh và truy xuất luật giao thông

**Tạo mô tả cảnh:** các detections được đọc từ `scene_cache.jsonl` và gom theo từng video, theo các cửa sổ thời gian dài `WIN_SEC` giây (mặc định 3 giây). Tên nhãn biển báo và đèn tín hiệu được khai báo trong `SIGN_NAMES` và `LIGHT_NAMES`; hãy bảo đảm tên lớp ở đây phù hợp với nhãn mà checkpoint YOLO của bạn trả về.

**Tạo kho luật RAG:** notebook đọc PDF luật giao thông, trích xuất văn bản, chia nội dung thành các dòng/đoạn, tạo embedding bằng `sentence-transformers/all-MiniLM-L6-v2`, rồi lưu FAISS index để tìm các đoạn gần với câu hỏi. Trong cell **M2.5**, cập nhật:

- `LUAT_PDF`: đường dẫn tới PDF luật giao thông.
- `LUAT_TXT`: đường dẫn lưu văn bản TXT sau khi trích xuất.

Cell này vừa chuyển PDF, vừa xây index, nạp index và chạy thử truy vấn. Nếu đã có sẵn các tệp index từ một lần chạy trước, có thể điều chỉnh thứ tự hoặc bỏ qua bước tạo lại kho luật tương ứng.

### Module 3 — Trả lời câu hỏi và xuất kết quả

Module này đọc `public_test.json`, ánh xạ mỗi câu hỏi tới video theo tên file, ghép mô tả cảnh và các đoạn luật liên quan, rồi gọi `infer_phi(...)` để dự đoán đáp án. Kết quả được ghi ở `roadbuddy/runtime/answers_public_test.csv` với hai cột `id` và `answer`.

Đường dẫn `PUBLIC_TEST_JSON` trong cell **MODULE 3 — COMMON SETUP** hiện là `/content/data/public_test.json`; hãy sửa cho khớp vị trí dữ liệu của bạn.

Notebook giả định `public_test.json` có dạng danh sách, mỗi phần tử gồm các trường như sau:

```json
[
  {
    "id": "question_id",
    "q": "Nội dung câu hỏi",
    "opts": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
    "video": "public_test/videos/example.mp4"
  }
]
```

Tên video lấy từ trường `video` phải tương ứng với video đã được phân tích trong Module 1. Hàm `load_scene_description` đối chiếu theo tên file cuối cùng, chẳng hạn `example.mp4`.

## Chuẩn bị và chạy trên Google Colab

1. Tải `testfinal2.ipynb` lên Google Colab và mở notebook.
2. Chọn **Runtime → Change runtime type** và bật GPU nếu có. GPU giúp chạy YOLO và các mô hình embedding nhanh hơn; thời gian xử lý phụ thuộc số lượng, độ dài video và cấu hình runtime.
3. Nếu đầu vào đặt trong Google Drive, chạy cell mount Drive trước các cell xử lý:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Đưa video, checkpoint YOLO, PDF luật và `public_test.json` lên Drive/Colab.
5. Sửa các đường dẫn `VIDEO_ROOT`, `WEIGHTS`, `LUAT_PDF`, `LUAT_TXT` và `PUBLIC_TEST_JSON` theo vị trí tệp thực tế.
6. Chạy các cell từ trên xuống dưới. Luồng dự kiến:
   - Cell cài thư viện và các cell import/cấu hình.
   - Module 1 để tạo metadata, detections và `scene_cache.jsonl`.
   - Module 2.1 để tạo `scene_description.jsonl`.
   - Module 2.3–2.5 để trích xuất PDF và tạo/nạp kho luật.
   - Module 3 để nạp câu hỏi, dựng ngữ cảnh, suy luận và xuất CSV.
7. Sau khi hoàn tất, tải `answers_public_test.csv` về hoặc sao chép thư mục `roadbuddy` sang Google Drive. Các tệp lưu ở vùng runtime tạm thời có thể mất khi phiên Colab bị ngắt hoặc reset.

## Đầu ra được tạo

| Đường dẫn | Mô tả |
| --- | --- |
| `roadbuddy/runtime/metadata.json` | Thông tin video đã xử lý: FPS, số frame, kích thước và thời lượng |
| `roadbuddy/runtime/detections.csv` | Detection theo frame, gồm lớp, confidence và bounding box |
| `roadbuddy/runtime/scene_cache.jsonl` | Detections dạng JSONL, đầu vào cho bước gom cảnh |
| `roadbuddy/runtime/scene_description.jsonl` | Tóm tắt biển báo/đèn tín hiệu theo video và cửa sổ thời gian |
| `roadbuddy/kb/index/chunks.jsonl` | Các đoạn văn bản luật dùng trong tìm kiếm |
| `roadbuddy/kb/index/faiss.index` | Chỉ mục FAISS cho truy xuất đoạn luật |
| `roadbuddy/runtime/answers_public_test.csv` | Đáp án cuối cùng, gồm `id` và `answer` |

## Lưu ý và các phần cần hoàn thiện trong notebook

- **Thiếu import `dataclass`:** Module 1 khai báo `@dataclass` cho lớp `Detection`, nhưng cell imports hiện chưa có `from dataclasses import dataclass`. Hãy thêm import này trước cell định nghĩa lớp để tránh `NameError`.
- **Chưa thấy định nghĩa `infer_phi(...)`:** Module 3 gọi hàm này trong `answer_one`, nhưng hàm chưa được định nghĩa ở các cell hiện có. Cần bổ sung cell cài/nạp mô hình và định nghĩa hàm suy luận phù hợp trước khi chạy Module 3. Nếu không, phần trả lời sẽ báo lỗi; nhánh xử lý lỗi trong notebook hiện ghi đáp án dự phòng `A`.
- **Đầu vào ngoài notebook:** notebook không chứa dữ liệu, checkpoint YOLO, PDF luật hoặc bộ câu hỏi. Cần cung cấp đầy đủ các tệp này và đặt đường dẫn đúng.
- **Cài thư viện:** cell cài đặt gọi các gói PyTorch, Ultralytics, Transformers, OpenCV, FAISS và các thư viện liên quan. `flash-attn` và `xformers` có thể không cài được trên một số cấu hình Colab; chúng được dùng như gói tăng tốc tùy chọn, nên có thể bỏ qua lệnh cài nếu gặp lỗi tương thích.
- **Thứ tự chạy:** Module 2 yêu cầu đầu ra Module 1; Module 3 yêu cầu `scene_description.jsonl` và kho luật đã được tạo. Chạy đúng thứ tự để tránh lỗi thiếu tệp.
- **Nhãn YOLO:** tên lớp mô hình cần tương thích với `SIGN_NAMES` và `LIGHT_NAMES`; nếu khác, cập nhật danh sách trong cấu hình Module 2 để các đối tượng được nhận diện đúng loại.

## Tệp trong repository

```text
.
├── README.md
└── testfinal2.ipynb
```

`README.md` mô tả cách chuẩn bị và chạy notebook; mã xử lý chính nằm trong một file `testfinal2.ipynb` để sử dụng trực tiếp trên Google Colab.
