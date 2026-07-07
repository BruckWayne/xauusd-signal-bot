# XAUUSD Multi-Timeframe Vision Signal Bot

Bot tự động mỗi giờ: lấy dữ liệu XAUUSD 6 khung thời gian (M1→H4), vẽ chart,
gửi cho Gemini Vision phân tích price action, và bắn tín hiệu vào Telegram.
Chạy hoàn toàn miễn phí trên GitHub Actions.

## Cài đặt (làm 1 lần)

### 1. Fork / tạo repo này trên GitHub
- Đăng nhập GitHub → **New repository** → đặt tên (ví dụ `xauusd-signal-bot`) → Public → Create
- Upload toàn bộ các file trong thư mục này lên repo (kéo thả qua giao diện web,
  hoặc dùng `git push` nếu quen dùng git)

### 2. Lấy 4 API key / thông tin cần thiết

| Biến | Lấy ở đâu |
|---|---|
| `TWELVEDATA_API_KEY` | https://twelvedata.com/ → đăng ký free → Dashboard → API Key |
| `GEMINI_API_KEY` | https://aistudio.google.com/ → Get API key → Create API key |
| `TELEGRAM_TOKEN` | Token bot bạn đã tạo với @BotFather |
| `TELEGRAM_CHAT_ID` | Gửi 1 tin cho bot, sau đó mở `https://api.telegram.org/bot<TOKEN>/getUpdates`, tìm `"chat":{"id": ...}` |

### 3. Thêm secrets vào GitHub repo

Vào repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**,
thêm lần lượt 4 secret với đúng tên ở trên (viết hoa, đúng chính tả):

- `TWELVEDATA_API_KEY`
- `GEMINI_API_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

### 4. Bật GitHub Actions

Vào tab **Actions** của repo → nếu thấy thông báo "Workflows aren't running", bấm
**I understand my workflows, go ahead and enable them**.

### 5. Chạy thử ngay (không cần đợi đến giờ)

Vào tab **Actions** → chọn workflow **XAUUSD Hourly Signal** ở cột trái →
bấm **Run workflow** (nút màu xanh bên phải) → **Run workflow**.

Đợi khoảng 30-60 giây, kiểm tra Telegram xem có tin nhắn chưa.
Nếu lỗi, bấm vào lần chạy đó trong Actions để xem log chi tiết (rất dễ đọc, báo lỗi rõ ràng).

## Cách hoạt động

1. `main.py` gọi Twelve Data lấy nến M1, M5, M15, M30, H1, H4 của XAU/USD
2. Vẽ 6 chart bằng `mplfinance`, giữ trong bộ nhớ (không lưu file để tránh phình repo)
3. Gửi cả 6 ảnh + 1 prompt phân tích top-down cho model `gemini-2.5-flash`
   (model có vision, miễn phí, ~1500 request/ngày)
4. Gemini trả JSON gồm tín hiệu BUY/SELL/WAIT, entry, SL, TP, lý do
5. Bot format và gửi tin nhắn Telegram

## Lịch chạy

Định nghĩa trong `.github/workflows/signal.yml`, mặc định `cron: "2 * * * *"`
(phút thứ 2 mỗi giờ, giờ UTC). Muốn đổi tần suất (ví dụ 30 phút/lần), sửa thành
`*/30 * * * *`.

Lưu ý: cron của GitHub Actions **không đảm bảo chạy đúng giây/phút**, có thể trễ
vài phút vào giờ cao điểm. Đây là giới hạn chung của GitHub, không phải lỗi code.

## Giới hạn cần biết

- **GitHub tự tắt scheduled workflow nếu repo không có commit nào trong 60 ngày.**
  Nếu dùng lâu dài, thỉnh thoảng vào repo commit gì đó (sửa README chẳng hạn) để giữ workflow hoạt động.
- Free tier Twelve Data: ~800 request/ngày, 8 request/phút — với lịch chạy 1 lần/giờ x 6 khung
  = 144 request/ngày, thoải mái trong hạn mức.
- Free tier Gemini: model Flash gói free tier đủ dùng cho tần suất theo giờ, nhưng
  Google có thể thay đổi hạn mức bất kỳ lúc nào — nếu thấy lỗi 429 (quá hạn mức),
  giảm tần suất chạy hoặc đợi qua ngày hôm sau (hạn mức reset theo giờ Thái Bình Dương).
- **Đây KHÔNG phải công cụ phân tích kỹ thuật đáng tin cậy tuyệt đối.** Vision LLM
  có thể đọc sai chart, nhận diện sai xu hướng. Hãy test trên tài khoản demo một
  thời gian dài trước khi cân nhắc dùng cho giao dịch thật, và luôn tự quản lý
  rủi ro (không copy y nguyên SL/TP mà không kiểm tra lại).

## Tùy chỉnh thêm (gợi ý, tự làm)

- Thêm log lịch sử tín hiệu vào 1 file JSON/CSV trong repo để sau này đánh giá
  độ chính xác của bot
- Thêm điều kiện: chỉ gửi Telegram khi `tin_hieu != "WAIT"` để đỡ spam
- Luân phiên gọi thêm model khác (nếu có ngân sách) để so sánh chéo kết quả
