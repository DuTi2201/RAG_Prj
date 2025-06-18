# RAG - Ứng dụng Hỏi đáp Tài liệu Bài học AIO

Ứng dụng RAG (Retrieval Augmented Generation) để hỏi đáp các nội dung trong tài liệu PDF bài học AIO.

## Mô tả

Dự án này xây dựng một hệ thống RAG hoàn chỉnh cho phép người dùng:
- Upload file PDF tài liệu bài học
- Đặt câu hỏi về nội dung tài liệu
- Nhận câu trả lời chính xác dựa trên nội dung đã upload

## Công nghệ sử dụng (Tech Stack)

- **Python 3.11+**: Ngôn ngữ lập trình chính
- **LangChain**: Framework xây dựng ứng dụng LLM
- **Transformers**: Thư viện Hugging Face cho các mô hình AI
- **Streamlit**: Framework tạo giao diện web
- **Vicuna 7B**: Mô hình ngôn ngữ lớn (LLM)
- **Vietnamese Bi-Encoder**: Mô hình embedding tối ưu cho tiếng Việt
- **ChromaDB**: Vector database để lưu trữ embeddings
- **PyPDF**: Thư viện xử lý file PDF

## Tính năng chính

- ✂️ **Semantic Chunking**: Chia văn bản dựa trên ngữ nghĩa
- 🔍 **Vector Similarity Search**: Tìm kiếm thông tin liên quan
- 💾 **Model Caching**: Cache mô hình để tối ưu hiệu suất
- 🚀 **4-bit Quantization**: Tối ưu bộ nhớ cho LLM
- 🎨 **Streamlit Interface**: Giao diện thân thiện người dùng

## Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone https://github.com/DuTi2201/RAG_Prj.git
cd RAG_Prj
```

### 2. Tạo môi trường ảo Conda

```bash
conda create -n aio-rag python=3.11
conda activate aio-rag
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường (tùy chọn)

```bash
cp .env.example .env
# Chỉnh sửa file .env nếu cần thiết
```

### 5. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## Hướng dẫn sử dụng

1. **Khởi chạy ứng dụng**: Chạy `streamlit run app.py`
2. **Đợi tải models**: Lần đầu chạy sẽ tải Embedding Model và LLM (khoảng 5-10 phút)
3. **Upload PDF**: Chọn file PDF tài liệu bài học cần hỏi đáp
4. **Xử lý PDF**: Nhấn nút "Xử lý PDF" để phân tích và tạo vector database
5. **Đặt câu hỏi**: Nhập câu hỏi về nội dung tài liệu
6. **Nhận câu trả lời**: Hệ thống sẽ trả lời dựa trên nội dung đã upload

## Yêu cầu hệ thống

- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **GPU**: Tùy chọn, hỗ trợ CUDA để tăng tốc
- **Disk**: Khoảng 15GB cho models và dependencies
- **Python**: 3.11 hoặc cao hơn

## Cấu hình mặc định

- **Embedding Model**: `bkai-foundation-models/vietnamese-bi-encoder`
- **LLM**: `lmsys/vicuna-7b-v1.5`
- **Chunk Size**: Tối thiểu 500 ký tự
- **Semantic Threshold**: 95 percentile
- **Max New Tokens**: 512

## Troubleshooting

### Lỗi thiếu bộ nhớ
- Đảm bảo có đủ RAM (16GB khuyến nghị)
- Đóng các ứng dụng khác khi chạy

### Lỗi tải model
- Kiểm tra kết nối internet
- Thử chạy lại ứng dụng

### Lỗi xử lý PDF
- Đảm bảo file PDF không bị mã hóa
- Thử với file PDF khác

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## License

MIT License
