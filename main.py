"""
XAUUSD Vision Signal Bot — bản gọn
------------------------------------
1. Lấy chart H4/H1/M15/M5 từ Twelve Data
2. Tính ATR H1 (biên độ tham khảo)
3. Kiểm tra lịch tin tức USD (veto cứng nếu có High Impact trong 60 phút tới)
4. Gửi ảnh + 1 prompt Price Action/Order Block top-down cho Gemini Vision
5. Gửi bảng tín hiệu ngắn gọn lên Telegram

Biến môi trường (GitHub Secrets): TWELVEDATA_API_KEY, GEMINI_API_KEY,
TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

Giới hạn cần biết: bot không có Volume/DXY/US10Y. Lịch tin tức lấy từ nguồn công khai
không chính thức (ForexFactory feed) — nếu lỗi, bot vẫn chạy tiếp nhưng không lọc được
tin tức lần đó.
"""

import os
import io
import json
import base64
import sys
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import mplfinance as mpf

# ---------- Cấu hình ----------
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"

# Bộ khung tối giản, đủ cho phương pháp Top-Down Price Action + Order Block:
# H4/H1 để xác định xu hướng + vùng giá, M15/M5 để xác nhận điểm vào lệnh.
TIMEFRAMES = {
    "4h": ("H4", 100),
    "1h": ("H1", 100),
    "15min": ("M15", 120),
    "5min": ("M5", 120),
}
TWELVEDATA_REQUEST_DELAY_SEC = 1.0

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_LOOKAHEAD_HOURS = 6
HIGH_IMPACT_VETO_MINUTES = 60
NEWS_RELEVANT_CURRENCIES = {"USD"}

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)


# ---------- Dữ liệu & chart ----------
def fetch_ohlc(interval: str, outputsize: int) -> pd.DataFrame:
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": SYMBOL, "interval": interval, "outputsize": outputsize,
            "apikey": TWELVEDATA_API_KEY, "format": "JSON", "order": "ASC",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data lỗi cho {interval}: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def render_chart_png(df: pd.DataFrame, label: str) -> bytes:
    buf = io.BytesIO()
    mpf.plot(
        df, type="candle", style="charles", title=f"XAUUSD - {label}",
        volume=False, savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
    )
    buf.seek(0)
    return buf.read()


def calc_atr(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 2) if pd.notna(atr) else None


# ---------- Lịch tin tức (veto cứng, không phải điểm trừ) ----------
def fetch_economic_calendar():
    try:
        r = requests.get(FF_CALENDAR_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"CẢNH BÁO: không lấy được lịch kinh tế: {e}", file=sys.stderr)
        return None


def get_next_high_impact_event(raw_events, now_utc: datetime):
    """Trả về sự kiện USD High Impact gần nhất trong NEWS_LOOKAHEAD_HOURS tới, hoặc None."""
    if not raw_events:
        return None
    candidates = []
    for ev in raw_events:
        if ev.get("country") not in NEWS_RELEVANT_CURRENCIES or ev.get("impact") != "High":
            continue
        try:
            ev_time = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except Exception:
            continue
        delta_min = (ev_time - now_utc).total_seconds() / 60
        if 0 <= delta_min <= NEWS_LOOKAHEAD_HOURS * 60:
            candidates.append({"title": ev.get("title", "N/A"), "minutes_until": round(delta_min)})
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["minutes_until"])


def build_news_context(raw_events, now_utc: datetime):
    """Trả về (đoạn text ngắn cho prompt, có_veto: bool, sự_kiện_gần_nhất hoặc None)."""
    if raw_events is None:
        return "Lịch tin tức: không lấy được lần này (nguồn lỗi) — coi rủi ro tin tức là chưa xác minh được.", False, None

    ev = get_next_high_impact_event(raw_events, now_utc)
    if ev is None:
        return f"Lịch tin tức: không có tin USD High Impact nào trong {NEWS_LOOKAHEAD_HOURS}h tới.", False, None

    veto = ev["minutes_until"] <= HIGH_IMPACT_VETO_MINUTES
    text = f"Lịch tin tức: {ev['title']} (USD, High Impact) — còn {ev['minutes_until']} phút."
    if veto:
        text += f" ⚠️ Trong vòng {HIGH_IMPACT_VETO_MINUTES} phút tới → BẮT BUỘC NO TRADE."
    return text, veto, ev


# ---------- Prompt & schema ----------
SYSTEM_PROMPT = """
Bạn là trader chuyên nghiệp phân tích XAUUSD theo phương pháp Price Action + Order Block
đa khung thời gian (Top-Down) — phương pháp thực chiến phổ biến và hiệu quả cho intraday.

Bạn nhận {so_luong_anh} ảnh chart theo thứ tự lớn → nhỏ: {danh_sach_khung}.
{atr_context}
{news_context}

CÁCH PHÂN TÍCH (3 bước, phải đủ cả 3 mới ra tín hiệu):
1. Xu hướng chính: đọc từ khung lớn nhất trong bộ ảnh (H4/H1) — tăng, giảm, hay sideway.
2. Vùng giá quan trọng: tìm vùng giá gần nhất mà giá đang phản ứng — order block, vùng
   cung/cầu, hỗ trợ/kháng cự, hoặc thanh khoản vừa bị quét. Nêu rõ mức giá cụ thể.
3. Xác nhận vào lệnh: trên khung nhỏ (M15/M5), tìm tín hiệu xác nhận tại đúng vùng giá
   đó — phá cấu trúc (BOS/CHOCH), nến engulfing, pin bar, hoặc mẫu hình đảo chiều rõ.

QUYẾT ĐỊNH: chỉ ra BUY/SELL khi CẢ 3 bước trên cùng hướng và rõ ràng. Thiếu 1 trong 3 →
NO TRADE. Nếu lịch tin tức ở trên báo veto → luôn NO TRADE, bỏ qua phân tích kỹ thuật.

Nếu BUY/SELL: SL đặt ngay sau vùng cấu trúc bị phá (không phải số tròn tùy ý). TP1/TP2
đặt tại vùng thanh khoản/kháng cự-hỗ trợ tiếp theo — dùng CÙNG phương pháp đã dùng để
tìm entry, không áp tỷ lệ R:R cố định tách rời cấu trúc giá. Thời gian giữ lệnh đề xuất
1-6 giờ (đủ để entry M15/M5 phát triển theo H1/H4, không quá dài để tránh sideway/tin
tức ăn mòn lợi nhuận).

Trả lời NGẮN GỌN, đúng trọng tâm, đúng JSON schema đã cấu hình, không thêm chữ nào khác.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "final_verdict": {"type": "STRING", "enum": ["BUY", "SELL", "NO TRADE"]},
        "confidence_percent": {"type": "NUMBER"},
        "xu_huong": {"type": "STRING"},
        "vung_gia_quan_trong": {"type": "STRING"},
        "tin_hieu_vao_lenh": {"type": "STRING"},
        "entry": {"type": "NUMBER", "nullable": True},
        "stop_loss": {"type": "NUMBER", "nullable": True},
        "take_profit_1": {"type": "NUMBER", "nullable": True},
        "take_profit_2": {"type": "NUMBER", "nullable": True},
        "ly_do_tp": {"type": "STRING"},
        "thoi_gian_giu_lenh": {"type": "STRING"},
        "dieu_kien_vo_hieu": {"type": "STRING"},
        "ghi_chu": {"type": "STRING"},
    },
    "required": ["final_verdict", "confidence_percent", "xu_huong", "ghi_chu"],
}


def build_prompt(image_labels, atr_h1, news_context: str) -> str:
    atr_context = (
        f"ATR H1 (14 kỳ) ≈ {atr_h1} USD — dùng để kiểm tra TP có khả thi trong thời gian giữ lệnh hay không."
        if atr_h1 else "Không tính được ATR lần này — đánh giá biên độ trực tiếp qua chart."
    )
    return SYSTEM_PROMPT.format(
        so_luong_anh=len(image_labels),
        danh_sach_khung=", ".join(image_labels),
        atr_context=atr_context,
        news_context=news_context,
    )


def call_gemini(images: dict, news_context: str, atr_h1=None) -> dict:
    prompt = build_prompt(list(images.keys()), atr_h1, news_context)
    parts = [{"text": prompt}]
    for _, png_bytes in images.items():
        parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png_bytes).decode("utf-8")}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    r = requests.post(GEMINI_URL, json=body, timeout=120)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json\n", "", 1).replace("json", "", 1)
    return json.loads(text)


# ---------- Telegram ----------
def send_telegram(text: str):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    r.raise_for_status()


def send_telegram_photo(png_bytes: bytes, caption: str = ""):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", png_bytes, "image/png")},
        timeout=30,
    )
    r.raise_for_status()


def _fmt_price(v):
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "N/A"


def calc_risk_reward(entry, sl, tp1):
    if not all(isinstance(x, (int, float)) for x in (entry, sl, tp1)):
        return None, None, None
    sl_dist, tp_dist = abs(entry - sl), abs(entry - tp1)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
    return sl_dist, tp_dist, rr


def confidence_bar(percent):
    if not isinstance(percent, (int, float)):
        return "⬜⬜⬜⬜⬜"
    filled = min(5, max(0, round(percent / 20)))
    return "🟩" * filled + "⬜" * (5 - filled)


def format_message(signal: dict) -> str:
    now_vn = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    verdict = signal.get("final_verdict", "NO TRADE")
    emoji = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪️"}.get(verdict, "⚪️")
    conf = signal.get("confidence_percent")

    lines = [
        f"{emoji} <b>XAUUSD — {verdict}</b>  |  {now_vn.strftime('%H:%M %d/%m')} (VN)",
        f"📊 Tin cậy: {confidence_bar(conf)} {conf if conf is not None else 'N/A'}%",
        f"📈 Xu hướng: {signal.get('xu_huong', 'N/A')}",
    ]
    if signal.get("vung_gia_quan_trong"):
        lines.append(f"🎯 Vùng giá: {signal.get('vung_gia_quan_trong')}")

    if verdict in ("BUY", "SELL"):
        entry, sl = signal.get("entry"), signal.get("stop_loss")
        tp1, tp2 = signal.get("take_profit_1"), signal.get("take_profit_2")
        sl_dist, tp_dist, rr = calc_risk_reward(entry, sl, tp1)

        lines.append("")
        lines.append(f"💰 Entry: <code>{_fmt_price(entry)}</code>")
        lines.append(f"🛑 SL: <code>{_fmt_price(sl)}</code>" + (f"  (~{sl_dist:.2f} pt)" if sl_dist else ""))
        lines.append(f"✅ TP1: <code>{_fmt_price(tp1)}</code>" + (f"  (~{tp_dist:.2f} pt)" if tp_dist else ""))
        if tp2 is not None:
            lines.append(f"✅ TP2: <code>{_fmt_price(tp2)}</code>")
        if rr:
            lines.append(f"⚖️ R:R: 1 : {rr}")
        lines.append(f"⏳ Giữ lệnh: {signal.get('thoi_gian_giu_lenh', 'N/A')}")
        if signal.get("tin_hieu_vao_lenh"):
            lines.append(f"🕯 Xác nhận: {signal.get('tin_hieu_vao_lenh')}")
        if signal.get("dieu_kien_vo_hieu"):
            lines.append(f"🚫 Vô hiệu nếu: {signal.get('dieu_kien_vo_hieu')}")

    if signal.get("ghi_chu"):
        lines.append("")
        lines.append(f"📝 {signal.get('ghi_chu')}")

    lines.append("")
    lines.append("⚠️ <i>AI output, không phải khuyến nghị đầu tư. Tự quản lý rủi ro.</i>")
    return "\n".join(lines)


# ---------- Main ----------
def main():
    images, dfs = {}, {}
    items = list(TIMEFRAMES.items())
    for i, (tv_interval, (label, size)) in enumerate(items):
        try:
            df = fetch_ohlc(tv_interval, size)
            images[label] = render_chart_png(df, label)
            dfs[label] = df
            print(f"OK: {label}")
        except Exception as e:
            print(f"LỖI khung {label}: {e}", file=sys.stderr)
        if i < len(items) - 1:
            time.sleep(TWELVEDATA_REQUEST_DELAY_SEC)

    if len(images) < 3:
        send_telegram("⚠️ Bot lỗi: không lấy đủ dữ liệu chart lần này.")
        sys.exit(1)

    atr_h1 = calc_atr(dfs.get("H1"))
    now_utc = datetime.now(timezone.utc)
    raw_calendar = fetch_economic_calendar()
    news_context, news_veto, _ = build_news_context(raw_calendar, now_utc)

    try:
        signal = call_gemini(images, news_context, atr_h1=atr_h1)
    except Exception as e:
        print(f"LỖI Gemini: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini: {e}")
        sys.exit(1)

    # Ép cứng bằng code (không chỉ dựa vào AI tuân thủ prompt) nếu có tin High Impact sắp tới.
    if news_veto and signal.get("final_verdict") != "NO TRADE":
        signal["final_verdict"] = "NO TRADE"
        signal["ghi_chu"] = (signal.get("ghi_chu", "") + " [Hệ thống ép NO TRADE do sắp có tin High Impact.]").strip()

    message = format_message(signal)

    chart_for_photo = images.get("H1") or next(iter(images.values()), None)
    try:
        if chart_for_photo:
            verdict = signal.get("final_verdict", "NO TRADE")
            emoji = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪️"}.get(verdict, "⚪️")
            send_telegram_photo(chart_for_photo, caption=f"{emoji} XAUUSD H1 — {verdict}")
    except Exception as e:
        print(f"CẢNH BÁO: gửi ảnh thất bại: {e}", file=sys.stderr)

    send_telegram(message)
    print("Đã gửi Telegram thành công.")


if __name__ == "__main__":
    main()
