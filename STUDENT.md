# Thông tin học viên

- Họ và tên: Vũ Đức Minh
- MSSV: 2A202603673
- Lớp: L3A / L3B / khác: L3A
- Chủ đề và corpus: Luật tổ chức chính quyền địa phương tháng 2 và tháng 6/2025
- Repo bài nộp: repo này
- Commit chốt bài: commit mới nhất

## Lệnh chạy pipeline

Chạy các lệnh sau từ thư mục gốc repository; cần cấu hình `DATABASE_URL` và `GEMINI_API_KEY` trong `starter/.env`.

### Nạp dữ liệu

```bash
cd starter
python -m src.run ingest --strategy fixed --chunk-size 1200 --overlap 150
python -m src.run ingest --strategy paragraph --chunk-size 1200
```

`--strategy` chọn `fixed` hoặc `paragraph`. Kích thước chunk tính theo ký tự; overlap chỉ áp dụng cho `fixed`. Mặc định là 1200 ký tự và overlap 150 ký tự. Model và số chiều mặc định là `gemini-embedding-001` và 3072; có thể cấu hình qua `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` hoặc tùy chọn `--model`, `--dimensions`.

### Truy vấn top 3

```bash
python -m src.run query "Câu hỏi cần tìm" --strategy fixed
python -m src.run query --strategy paragraph
```

Lệnh thứ hai sẽ nhắc nhập câu hỏi. Chọn đúng strategy đã dùng khi nạp dữ liệu. Kết quả gồm điểm tương đồng, tài liệu/trang, nội dung chunk và metadata.

### Chạy bộ câu hỏi benchmark

```bash
python -m src.run test --strategy all
```

Mặc định lệnh đọc `../benchmark/query.csv`, chạy cả hai strategy và append Hit@1/Evidence Hit@3 của dev/test vào `../artifacts/experiment_log.csv`. Có thể chọn một strategy (`fixed` hoặc `paragraph`) hay đổi file bằng `--query-file` và `--log-file`. Truyền `--chunk-size` và `--overlap` tương ứng với cấu hình đã ingest để ghi đúng thông tin thí nghiệm; chúng không thay đổi chunk trong database. Dùng cùng model và số chiều đã dùng khi ingest.

Xem các tham số khác bằng `python -m src.run --help` hoặc thêm `--help` sau tên lệnh.
