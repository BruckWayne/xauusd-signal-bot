"""
XAUUSD Multi-Timeframe Vision Signal Bot
-----------------------------------------
Luồng chạy:
1. Lấy dữ liệu OHLC XAU/USD từ Twelve Data (M1, M5, M15, M30, H1, H4)
2. Vẽ chart nến cho từng khung bằng mplfinance
3. Gửi tất cả ảnh + prompt cho Gemini Vision phân tích
4. Parse kết quả JSON (tín hiệu, entry, sl, tp, lý do)
5. Gửi thông báo lên Telegram

Biến môi trường cần có (đặt trong GitHub Secrets):
- TWELVEDATA_API_KEY
- GEMINI_API_KEY
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import io
import json
import base64
import sys
from datetime import datetime, timezone

import requests
import pandas as pd
import mplfinance as mpf

# ---------- Cấu hình ----------
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
# Twelve Data interval string : (tên hiển thị, số nến lấy về)
TIMEFRAMES = {
    "1min": ("M1", 120),
    "5min": ("M5", 120),
    "15min": ("M15", 120),
    "30min": ("M30", 100),
    "1h": ("H1", 100),
    "4h": ("H4", 100),
}

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)


# ---------- Bước 1: Lấy dữ liệu ----------
def fetch_ohlc(interval: str, outputsize: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data trả lỗi cho {interval}: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0
    return df


# ---------- Bước 2: Vẽ chart ----------
def render_chart_png(df: pd.DataFrame, label: str) -> bytes:
    buf = io.BytesIO()
    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"XAUUSD - {label}",
        volume=False,
        savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
    )
    buf.seek(0)
    return buf.read()


# ---------- Bước 3: Gọi Gemini Vision ----------
PROMPT = """
Bạn là một trader chuyên nghiệp phân tích price action (KHÔNG dùng chỉ báo máy móc).
Bạn được cung cấp 6 ảnh chart nến của XAUUSD theo thứ tự: M1, M5, M15, M30, H1, H4.

Hãy phân tích đa khung thời gian theo phương pháp top-down (H4 -> H1 -> M30 -> M15 -> M5 -> M1):
- Xác định xu hướng chính trên khung lớn (H4, H1)
- Xác định vùng hỗ trợ/kháng cự, order block, vùng thanh khoản quan trọng
- Tìm điểm vào lệnh hợp lý trên khung nhỏ (M15/M5/M1) theo xu hướng khung lớn
- Chỉ đưa tín hiệu BUY/SELL khi có setup rõ ràng, nếu không có setup tốt thì trả lời "WAIT"

CHỈ trả lời bằng JSON hợp lệ, không thêm markdown, không thêm giải thích ngoài JSON:
{
  "tin_hieu": "BUY" hoặc "SELL" hoặc "WAIT",
  "do_tin_cay": "cao/trung binh/thap",
  "xu_huong_chinh": "mô tả ngắn gọn xu huong H4/H1",
  "entry": số hoặc null,
  "stop_loss": số hoặc null,
  "take_profit": số hoặc null,
  "ly_do": "giải thích ngắn gọn bằng tiếng Việt, tối đa 4 câu"
}
"""


def call_gemini(images: dict) -> dict:
    parts = [{"text": PROMPT}]
    for label, png_bytes in images.items():
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(png_bytes).decode("utf-8"),
                }
            }
        )

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2},
    }

    r = requests.post(GEMINI_URL, json=body, timeout=90)
    r.raise_for_status()
    data = r.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip()
    # Gemini đôi khi vẫn bọc ```json ... ``` dù đã yêu cầu không làm vậy
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("json", "", 1)
    return json.loads(text)


# ---------- Bước 4: Gửi Telegram ----------
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()


def format_message(signal: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tin_hieu = signal.get("tin_hieu", "N/A")
    emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪️"}.get(tin_hieu, "⚪️")

    msg = (
        f"{emoji} <b>XAUUSD - {tin_hieu}</b>\n"
        f"⏰ {now}\n\n"
        f"<b>Xu hướng chính:</b> {signal.get('xu_huong_chinh', 'N/A')}\n"
        f"<b>Độ tin cậy:</b> {signal.get('do_tin_cay', 'N/A')}\n"
    )
    if tin_hieu in ("BUY", "SELL"):
        msg += (
            f"\n<b>Entry:</b> {signal.get('entry')}\n"
            f"<b>SL:</b> {signal.get('stop_loss')}\n"
            f"<b>TP:</b> {signal.get('take_profit')}\n"
        )
    msg += f"\n<b>Lý do:</b> {signal.get('ly_do', '')}\n"
    msg += "\n⚠️ Đây là output từ AI, không phải khuyến nghị đầu tư. Tự quản lý rủi ro."
    return msg


# ---------- Main ----------
def main():
    images = {}
    for tv_interval, (label, size) in TIMEFRAMES.items():
        try:
            df = fetch_ohlc(tv_interval, size)
            images[label] = render_chart_png(df, label)
            print(f"OK: lấy dữ liệu và vẽ chart {label}")
        except Exception as e:
            print(f"LỖI khi xử lý khung {label}: {e}", file=sys.stderr)

    if len(images) < 3:
        send_telegram("⚠️ Bot lỗi: không lấy đủ dữ liệu chart để phân tích lần này.")
        sys.exit(1)

    try:
        signal = call_gemini(images)
    except Exception as e:
        print(f"LỖI khi gọi Gemini: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini: {e}")
        sys.exit(1)

    message = format_message(signal)
    send_telegram(message)
    print("Đã gửi tín hiệu lên Telegram thành công.")


if __name__ == "__main__":
    main()
