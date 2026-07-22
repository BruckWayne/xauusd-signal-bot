XAUUSD AI Vision Scalp Bot

Phương pháp: H1 bias → M15 setup → M5 confirmation → M1 trigger.

Lỗi đã sửa

Bản cũ có sự không thống nhất:

README hướng dẫn run_mode=test nhưng workflow chỉ có input force_ai.

main.py không gửi TEST STARTED như README mô tả.

Prefilter/news veto/WAIT/NO TRADE có thể khiến workflow chạy xong mà Telegram im lặng.

Nếu secret thiếu, chương trình lỗi ngay khi import và không có chẩn đoán rõ.

Không có bước kiểm tra độc lập getMe và sendMessage của Telegram.

Bản này xử lý riêng kết nối Telegram trước khi lấy dữ liệu thị trường hoặc gọi Gemini.

Cách test chính xác

Ghi đè cả bốn file trong repository.

Đảm bảo workflow nằm đúng đường dẫn .github/workflows/signal.yml.

Commit các file vào default branch của repository.

Vào Actions → XAUUSD Scalp Signal → Run workflow.

Chọn run_mode = test.

Trong chế độ test, thứ tự phải là:

Validate Python code báo SELF_TEST_OK.

Verify required secrets báo cả bốn secret là configured.

Telegram connection test gửi tin:

XAUUSD SCALP BOT — TELEGRAM CONNECTED
Workflow test has started.

Sau đó bot mới lấy dữ liệu Twelve Data và gọi Gemini.

Test mode gửi cả WAIT/NO TRADE; nếu bước phân tích lỗi, bot cố gửi XAUUSD TEST FAILED.

Cách đọc lỗi trong Actions

TELEGRAM_TOKEN invalid: token sai hoặc đã bị thu hồi.

chat not found: TELEGRAM_CHAT_ID sai hoặc bot chưa được người dùng nhấn Start.

bot was blocked by the user: tài khoản đã chặn bot.

group chat was upgraded: cần cập nhật chat ID mới, thường có dạng số âm.

TWELVEDATA_API_KEY is missing: secret chưa tạo đúng tên.

Twelve Data không trả values: key/quota/symbol/data plan gặp lỗi.

Gemini HTTP 404: model không tồn tại với project hiện tại.

Gemini HTTP 429: vượt quota/rate limit.

GitHub Secrets bắt buộc

Tên phải giống tuyệt đối:

TWELVEDATA_API_KEY
GEMINI_API_KEY
TELEGRAM_TOKEN
TELEGRAM_CHAT_ID

Không thêm dấu nháy vào giá trị secret. Với chat riêng, hãy nhấn Start cho bot trước.

Lịch normal

- cron: "1-56/5 6-21 * * 1-5"

Normal mode chỉ gửi BUY/SELL. Test mode luôn gửi kết quả để chẩn đoán.

Cấu trúc repository

main.py
requirements.txt
README.md
.github/
└── workflows/
    └── signal.yml

Chỉ dùng cho nghiên cứu và paper-trading; không nối trực tiếp với chức năng tự động đặt lệnh.
