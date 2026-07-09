# XAUUSD Multi-Timeframe Vision Signal Bot — bản nâng cấp

Bot tự động lấy dữ liệu XAUUSD đa khung **H4/H1/M15/M5**, vẽ chart, gửi cho Gemini Vision phân tích Price Action + Order Block, sau đó chạy thêm một lớp **validator bằng code** trước khi gửi tín hiệu lên Telegram.

> Mục tiêu của bản nâng cấp: không để AI Vision quyết định một mình. Gemini phân tích chart, còn code kiểm duyệt lại tín hiệu bằng rule kỹ thuật, tin tức, R:R, ATR và confidence.

## Cấu trúc file

```text
.
├── main.py
├── requirements.txt
└── .github/workflows/signal.yml
```

## Chức năng chính

1. Lấy dữ liệu XAU/USD từ Twelve Data cho 4 khung: **H4, H1, M15, M5**.
2. Vẽ chart nến có thêm EMA 20/50/200 và đường giá hiện tại.
3. Tính dữ liệu định lượng:
   - ATR H1
   - EMA 20/50/200
   - bias H4/H1
   - swing high/swing low gần nhất
4. Kiểm tra lịch tin tức USD High Impact từ ForexFactory feed.
5. Gửi ảnh chart + dữ liệu định lượng cho Gemini Vision.
6. Gemini trả JSON theo schema: `BUY`, `SELL` hoặc `NO TRADE`.
7. Code chạy validator sau Gemini:
   - chặn lệnh nếu có tin USD High Impact trong vùng cấm,
   - chặn nếu không xác minh được lịch tin,
   - chặn nếu confidence thấp,
   - chặn nếu entry/SL/TP sai cấu trúc,
   - chặn nếu R:R thấp,
   - chặn nếu SL quá rộng/quá sát so với ATR,
   - chặn nếu entry quá xa giá hiện tại,
   - chặn nếu BUY/SELL ngược hoàn toàn bias H4/H1.
8. Gửi Telegram và ghi log lịch sử vào `logs/signals.csv` và `logs/signals.jsonl`.
9. Workflow GitHub upload log thành artifact sau mỗi lần chạy để bạn tải về kiểm tra.

## Cài đặt GitHub Secrets

Vào repo GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**, thêm 4 biến bắt buộc:

| Biến | Ý nghĩa |
|---|---|
| `TWELVEDATA_API_KEY` | API key Twelve Data |
| `GEMINI_API_KEY` | API key Google AI Studio/Gemini |
| `TELEGRAM_TOKEN` | Token bot Telegram từ BotFather |
| `TELEGRAM_CHAT_ID` | Chat ID nhận tín hiệu |

## Biến cấu hình tùy chọn

Có thể thêm trong workflow hoặc GitHub Secrets/Variables nếu muốn tinh chỉnh:

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `MIN_CONFIDENCE` | `70` | Confidence tối thiểu để cho phép BUY/SELL |
| `MIN_RR` | `1.2` | R:R tối thiểu với TP1 |
| `MAX_ENTRY_DISTANCE_ATR` | `0.45` | Entry không được cách giá hiện tại quá 0.45 ATR H1 |
| `MAX_SL_ATR_MULTIPLIER` | `1.35` | SL không được rộng hơn 1.35 ATR H1 |
| `MIN_SL_ATR_MULTIPLIER` | `0.08` | SL không được quá sát so với ATR H1 |
| `NEWS_LOOKAHEAD_HOURS` | `6` | Số giờ nhìn trước lịch tin High Impact |
| `HIGH_IMPACT_VETO_BEFORE_MINUTES` | `60` | Cấm giao dịch trước tin High Impact |
| `HIGH_IMPACT_VETO_AFTER_MINUTES` | `30` | Cấm giao dịch sau tin High Impact |
| `SEND_NO_TRADE` | `true` | Có gửi Telegram khi NO TRADE hay không |
| `LOG_DIR` | `logs` | Thư mục lưu log |

## Lịch chạy

Workflow mặc định chạy vào phút thứ 2 mỗi giờ theo UTC:

```yaml
- cron: "2 * * * *"
```

Nếu muốn chạy 30 phút/lần:

```yaml
- cron: "*/30 * * * *"
```

Lưu ý: GitHub Actions cron có thể chạy trễ vài phút vào giờ cao điểm.

## Cách chạy thử

1. Upload các file lên repo GitHub.
2. Đảm bảo file workflow nằm đúng đường dẫn: `.github/workflows/signal.yml`.
3. Vào tab **Actions**.
4. Chọn workflow **XAUUSD Hourly Signal**.
5. Bấm **Run workflow**.
6. Kiểm tra Telegram và log trong phần artifact của lần chạy.

## Cách đọc log

Sau mỗi lần chạy, workflow upload thư mục `logs` thành artifact tên `xauusd-signal-logs`.

Trong đó:

- `signals.csv`: dễ mở bằng Excel/Google Sheets.
- `signals.jsonl`: phù hợp để xử lý bằng Python sau này.

Các cột quan trọng:

| Cột | Ý nghĩa |
|---|---|
| `final_verdict` | Kết luận cuối sau validator |
| `confidence_percent` | Độ tin cậy AI trả về |
| `risk_reward_tp1` | R:R đến TP1 |
| `atr_h1` | ATR H1 tại thời điểm chạy |
| `h4_bias`, `h1_bias` | Bias định lượng từ EMA |
| `news_veto` | Có bị chặn bởi tin tức hay không |
| `ghi_chu` | Lý do AI/validator |

## Gợi ý vận hành thực tế

- Nên chạy demo tối thiểu vài tuần trước khi dùng cho quyết định thật.
- Không nên tự động vào lệnh ngay từ Telegram nếu chưa có thống kê hiệu quả.
- Sau khi có đủ log, hãy đánh giá:
  - tỷ lệ đúng sau 1h/3h/6h,
  - lệnh bị chặn bởi validator có hợp lý không,
  - R:R trung bình,
  - thời điểm bot hay sai nhất,
  - loại tin tức nào làm tín hiệu nhiễu nhiều nhất.

## Lưu ý rủi ro

Đây là công cụ hỗ trợ phân tích kỹ thuật bằng AI, không phải khuyến nghị đầu tư. AI Vision có thể đọc sai chart, dữ liệu API có thể lỗi hoặc trễ, lịch tin tức dùng nguồn công khai không chính thức. Luôn tự kiểm tra lại và quản lý rủi ro.
