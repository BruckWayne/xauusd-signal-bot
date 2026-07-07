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


def calc_atr(df: pd.DataFrame, period: int = 14):
    """ATR (Average True Range) — dùng làm mốc khách quan cho biên độ thực tế,
    để TP/SL không bị đặt vượt quá khả năng di chuyển giá trong phiên."""
    if df is None or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 2) if pd.notna(atr) else None


# ---------- Bước 3: Gọi Gemini Vision ----------
PROMPT_TEMPLATE = """
Bạn là một trader chuyên nghiệp phân tích price action (KHÔNG dùng chỉ báo máy móc).
Bạn được cung cấp 6 ảnh chart nến của XAUUSD theo thứ tự: M1, M5, M15, M30, H1, H4.

Hãy phân tích đa khung thời gian theo phương pháp top-down (H4 -> H1 -> M30 -> M15 -> M5 -> M1):
- Xác định xu hướng chính trên khung lớn (H4, H1)
- Xác định vùng hỗ trợ/kháng cự, order block, vùng thanh khoản quan trọng
- Tìm điểm vào lệnh (entry) hợp lý trên khung nhỏ (M15/M5/M1) theo xu hướng khung lớn
- Chỉ đưa tín hiệu BUY/SELL khi có setup rõ ràng, nếu không có setup tốt thì trả lời "WAIT"

QUY TẮC BẮT BUỘC — điểm entry và điểm chốt lời (TP) phải cùng một phương pháp, không được tách rời:
- Nếu entry được xác định dựa trên cấu trúc M1/M5 (ví dụ: quét thanh khoản, phá vỡ cấu trúc nhỏ, order block M5),
  thì TP phải là vùng thanh khoản / kháng cự-hỗ trợ / order block TIẾP THEO cùng logic cấu trúc đó trên khung lớn hơn
  (M15/M30/H1) — KHÔNG được chọn TP theo số điểm tùy ý hoặc theo tỷ lệ R:R cố định.
- SL đặt sau vùng cấu trúc vừa dùng để entry (dưới order block/đáy gần nhất khi BUY, trên order block/đỉnh gần nhất khi SELL).
{atr_context}
Dữ liệu ATR ở trên phản ánh biên độ trung bình thực tế — nếu TP dự kiến vượt quá khả năng di chuyển của giá trong
khung thời gian giữ lệnh đề xuất, hãy chọn lại vùng TP gần hơn (thanh khoản/kháng cự-hỗ trợ gần nhất phù hợp) thay vì
giữ nguyên TP xa.

Về thời gian giữ lệnh: ưu tiên đề xuất THỜI GIAN VỪA PHẢI, không quá dài (tránh lệnh bị sideway/tin tức ăn mòn lợi
nhuận, tránh phí qua đêm/gap cuối tuần), và không quá ngắn kiểu scalp giây/phút. Khung tham chiếu hợp lý: khoảng
1–6 giờ cho một lệnh intraday dựa trên entry M1/M5 xác nhận theo xu hướng H1/H4.

CHỈ trả lời bằng JSON hợp lệ, không thêm markdown, không thêm giải thích ngoài JSON:
{{
  "tin_hieu": "BUY" hoặc "SELL" hoặc "WAIT",
  "do_tin_cay": "cao/trung binh/thap",
  "xu_huong_chinh": "mô tả ngắn gọn xu huong H4/H1",
  "cau_truc_gia": "mô tả cấu trúc giá hiện tại (higher high/higher low, sideway, break of structure...)",
  "vung_gia_quan_trong": "liệt kê ngắn gọn các vùng hỗ trợ/kháng cự hoặc order block quan trọng đang theo dõi, kèm mức giá",
  "khung_gia_vao_lenh": "khung thời gian dùng để xác nhận điểm vào lệnh, ví dụ M5/M1",
  "phuong_phap_tp": "giải thích ngắn gọn vì sao TP được chọn tại vùng đó, và nó liên hệ thế nào với logic đã dùng để xác định entry",
  "entry": số hoặc null,
  "stop_loss": số hoặc null,
  "take_profit": số hoặc null,
  "thoi_gian_giu_lenh_de_xuat": "khoảng thời gian đề xuất giữ lệnh, ví dụ '2-4 giờ' (ưu tiên vừa phải, không quá dài)",
  "dieu_kien_thoat_som": "điều kiện nên thoát lệnh sớm nếu setup mất hiệu lực trước khi tới TP/hết thời gian đề xuất",
  "kich_ban_khac": "kịch bản thay thế nếu giá đi ngược setup chính, hoặc null",
  "ly_do": "giải thích ngắn gọn bằng tiếng Việt, tối đa 4 câu"
}}
"""


def build_prompt(atr_h1, atr_h4) -> str:
    if atr_h1 or atr_h4:
        atr_context = (
            f"\nBiên độ tham khảo (ATR 14 kỳ): H1 ≈ {atr_h1 if atr_h1 else 'N/A'} USD, "
            f"H4 ≈ {atr_h4 if atr_h4 else 'N/A'} USD.\n"
        )
    else:
        atr_context = "\n"
    return PROMPT_TEMPLATE.format(atr_context=atr_context)


def call_gemini(images: dict, atr_h1=None, atr_h4=None) -> dict:
    prompt = build_prompt(atr_h1, atr_h4)
    parts = [{"text": prompt}]
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
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()


def send_telegram_photo(png_bytes: bytes, caption: str = ""):
    """Gửi kèm 1 ảnh chart (mặc định dùng H1) để trader có ngữ cảnh trực quan."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", png_bytes, "image/png")}
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "HTML",
    }
    r = requests.post(url, data=payload, files=files, timeout=30)
    r.raise_for_status()


def _fmt_price(v):
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "N/A"


def calc_risk_reward(entry, sl, tp):
    """Tự tính khoảng cách SL/TP (USD) và tỷ lệ R:R thay vì phụ thuộc AI, để số liệu luôn chính xác."""
    if not all(isinstance(x, (int, float)) for x in (entry, sl, tp)):
        return None, None, None
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
    return sl_dist, tp_dist, rr


def get_session_label(now_utc: datetime) -> str:
    """Xác định phiên giao dịch hiện tại theo giờ UTC (khung giờ tương đối)."""
    h = now_utc.hour
    sessions = []
    if 0 <= h < 8:
        sessions.append("Á (Tokyo/Sydney)")
    if 7 <= h < 16:
        sessions.append("Âu (London)")
    if 12 <= h < 21:
        sessions.append("Mỹ (New York)")
    if not sessions:
        sessions.append("Giao thời (thanh khoản thấp)")
    return " + ".join(sessions)


def confidence_bar(level: str):
    lvl = (level or "").strip().lower()
    if "cao" in lvl:
        return "🟩🟩🟩", "Cao"
    if "trung" in lvl:
        return "🟨🟨⬜", "Trung bình"
    if "thấp" in lvl or "thap" in lvl:
        return "🟥⬜⬜", "Thấp"
    return "⬜⬜⬜", level or "N/A"


def format_message(signal: dict) -> str:
    now_utc = datetime.now(timezone.utc)
    now_vn = now_utc.astimezone(timezone(timedelta(hours=7)))

    tin_hieu = signal.get("tin_hieu", "N/A")
    emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪️"}.get(tin_hieu, "⚪️")
    bar, conf_label = confidence_bar(signal.get("do_tin_cay"))

    entry = signal.get("entry")
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit")
    sl_dist, tp_dist, rr = calc_risk_reward(entry, sl, tp)

    lines = [
        f"{emoji} <b>XAUUSD — TÍN HIỆU {tin_hieu}</b>",
        "━━━━━━━━━━━━━━━━━━━",
        f"🕒 <b>Giờ VN:</b> {now_vn.strftime('%d/%m/%Y %H:%M')}  |  <b>UTC:</b> {now_utc.strftime('%H:%M')}",
        f"🌍 <b>Phiên:</b> {get_session_label(now_utc)}",
        f"📊 <b>Độ tin cậy:</b> {bar} {conf_label}",
        "",
        f"📈 <b>Xu hướng chính (H4/H1):</b> {signal.get('xu_huong_chinh', 'N/A')}",
    ]

    if signal.get("cau_truc_gia"):
        lines.append(f"🧱 <b>Cấu trúc giá:</b> {signal.get('cau_truc_gia')}")
    if signal.get("vung_gia_quan_trong"):
        lines.append(f"🎯 <b>Vùng giá quan trọng:</b> {signal.get('vung_gia_quan_trong')}")

    if tin_hieu in ("BUY", "SELL"):
        lines.append("")
        lines.append("━━━ <b>KẾ HOẠCH VÀO LỆNH</b> ━━━")
        lines.append(f"🔑 <b>Khung xác nhận entry:</b> {signal.get('khung_gia_vao_lenh', 'N/A')}")
        lines.append(f"💰 <b>Entry:</b>  <code>{_fmt_price(entry)}</code>")
        lines.append(f"🛑 <b>Stop Loss:</b>  <code>{_fmt_price(sl)}</code>" + (f"  (~{sl_dist:.2f} pt)" if sl_dist else ""))
        lines.append(f"✅ <b>Take Profit:</b>  <code>{_fmt_price(tp)}</code>" + (f"  (~{tp_dist:.2f} pt)" if tp_dist else ""))
        lines.append(f"⚖️ <b>Tỷ lệ R:R:</b> {f'1 : {rr}' if rr else 'N/A'}")
        if signal.get("phuong_phap_tp"):
            lines.append(f"📐 <b>Căn cứ chọn TP:</b> {signal.get('phuong_phap_tp')}")
        lines.append("")
        lines.append(f"⏳ <b>Thời gian giữ lệnh đề xuất:</b> {signal.get('thoi_gian_giu_lenh_de_xuat', 'N/A')}")
        if signal.get("dieu_kien_thoat_som"):
            lines.append(f"🚪 <b>Thoát sớm nếu:</b> {signal.get('dieu_kien_thoat_som')}")
        if signal.get("kich_ban_khac"):
            lines.append(f"🔄 <b>Kịch bản khác:</b> {signal.get('kich_ban_khac')}")

    lines.append("")
    lines.append(f"📝 <b>Lý do:</b> {signal.get('ly_do', '')}")
    lines.append("")
    lines.append("⚠️ <i>Đây là output từ AI, không phải khuyến nghị đầu tư. Tự quản lý rủi ro và vốn.</i>")
    return "\n".join(lines)


# ---------- Main ----------
def main():
    images = {}
    dfs = {}
    for tv_interval, (label, size) in TIMEFRAMES.items():
        try:
            df = fetch_ohlc(tv_interval, size)
            images[label] = render_chart_png(df, label)
            dfs[label] = df
            print(f"OK: lấy dữ liệu và vẽ chart {label}")
        except Exception as e:
            print(f"LỖI khi xử lý khung {label}: {e}", file=sys.stderr)

    if len(images) < 3:
        send_telegram("⚠️ Bot lỗi: không lấy đủ dữ liệu chart để phân tích lần này.")
        sys.exit(1)

    atr_h1 = calc_atr(dfs.get("H1"))
    atr_h4 = calc_atr(dfs.get("H4"))

    try:
        signal = call_gemini(images, atr_h1=atr_h1, atr_h4=atr_h4)
    except Exception as e:
        print(f"LỖI khi gọi Gemini: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini: {e}")
        sys.exit(1)

    message = format_message(signal)

    # Gửi kèm ảnh chart H1 (nếu có) để trader thấy trực quan vùng entry/SL/TP,
    # sau đó gửi bảng phân tích chi tiết dạng text.
    chart_for_photo = images.get("H1") or next(iter(images.values()), None)
    try:
        if chart_for_photo:
            tin_hieu = signal.get("tin_hieu", "N/A")
            emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪️"}.get(tin_hieu, "⚪️")
            send_telegram_photo(chart_for_photo, caption=f"{emoji} XAUUSD H1 — tín hiệu {tin_hieu}")
    except Exception as e:
        print(f"CẢNH BÁO: gửi ảnh chart thất bại (bỏ qua, vẫn gửi text): {e}", file=sys.stderr)

    send_telegram(message)
    print("Đã gửi tín hiệu lên Telegram thành công.")


if __name__ == "__main__":
    main()
