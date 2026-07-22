XAUUSD AI Vision Scalp Bot

Bản này chuyển hệ thống từ intraday sang scalp theo trình tự:

H1 bias → M15 setup → M5 confirmation → M1 trigger

Tần suất hợp lý

Workflow chạy mỗi 5 phút, gần sau khi nến M5 đóng. GitHub Actions không phù hợpvới gọi mỗi phút hoặc theo tick vì lịch có thể trễ. Code chỉ gọi Gemini khi prefilterphát hiện setup tiềm năng; nếu thị trường không đủ điều kiện thì chỉ ghi log.

Cách giảm API

Bot chỉ gọi Twelve Data một lần cho M1 (outputsize=5000) rồi resample thànhM5/M15/H1/H4. Nhờ vậy lịch 5 phút không cần gọi riêng từng timeframe.

Khung vận hành

Khung

Vai trò

H1

Bias và cản lớn

M15

Setup và location

M5

Xác nhận cấu trúc

M1

Trigger/entry

H4

Context phụ

Lịch mặc định

- cron: "1-56/5 6-21 * * 1-5"

Tức 06:01-21:56 UTC, tương đương khoảng 13:01-04:56 giờ Việt Nam. Có thể giảmchi phí bằng cách thu hẹp xuống 07:00-17:59 UTC.

Update GitHub

Ghi đè 4 file:

main.py
requirements.txt
README.md
.github/workflows/signal.yml

Secrets giữ nguyên:

TWELVEDATA_API_KEY

GEMINI_API_KEY

TELEGRAM_TOKEN

TELEGRAM_CHAT_ID

Sau khi upload, vào Actions → XAUUSD Scalp Signal → Run workflow và chọnforce_ai=true để test một lần.

Setting chính

Setting

Mặc định

MIN_SETUP_SCORE

72

MIN_RR

1.50

MIN_PREFILTER_SCORE

4

MAX_SIGNAL_AGE_MINUTES

8

MAX_ENTRY_DISTANCE_ATR_M5

0.45

NEWS veto trước/sau

20/15 phút

Mặc định chỉ gửi BUY/SELL, không gửi WAIT/NO TRADE để tránh spam.

Lưu ý

Chỉ dùng cho nghiên cứu và paper-trading. GitHub cron có thể trễ, AI Vision có thểđọc sai chart, và tín hiệu scalp hết hiệu lực nhanh. Không nối trực tiếp với chức năngđặt lệnh tự động.
