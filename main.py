"""
XAUUSD Multi-Timeframe Vision Signal Bot (Institutional SMC Edition)
---------------------------------------------------------------------
Luồng chạy:
1. Lấy dữ liệu OHLC XAU/USD từ Twelve Data (M5, M15, M30, H1, H4, D1, W1)
2. Vẽ chart nến cho từng khung bằng mplfinance
3. Tính ATR (H1/H4) làm mốc biên độ khách quan
4. Gửi tất cả ảnh + prompt "Chief Market Strategist" cho Gemini Vision,
   ép buộc trả JSON đúng schema (responseSchema) thay vì tự parse văn bản
5. Gửi thông báo chi tiết lên Telegram (kèm ảnh chart H1/H4)

Biến môi trường cần có (đặt trong GitHub Secrets):
- TWELVEDATA_API_KEY
- GEMINI_API_KEY
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID

GIỚI HẠN CẦN BIẾT (đọc kỹ trước khi tin tưởng tuyệt đối vào output):
- Bot KHÔNG có dữ liệu Volume, DXY, US10Y, lịch kinh tế/tin tức thời gian thực.
  Model được yêu cầu KHÔNG bịa các dữ liệu này mà phải tự hạ điểm số ở mục
  Macro/Volume và nêu rõ "thiếu dữ liệu" — không phải là phân tích macro thật.
- Việc lọc "tin tức quan trọng trong 60 phút tới" trong bản gốc BẮT BUỘC có lịch
  kinh tế thời gian thực để làm đúng; bot này không có nguồn đó nên bước lọc
  tin tức được thay bằng "không xác minh được — coi là rủi ro chưa biết".
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

# Twelve Data interval string : (tên hiển thị, số nến lấy về)
# Bỏ M1 (không cần thiết với holding time mục tiêu 1-6h), thêm D1/W1 để có
# góc nhìn Daily/Weekly như phương pháp institutional yêu cầu.
# Thứ tự dict cũng là thứ tự top-down gửi cho AI: W1 -> D1 -> H4 -> H1 -> M30 -> M15 -> M5
TIMEFRAMES = {
    "1week": ("W1", 60),
    "1day": ("D1", 90),
    "4h": ("H4", 100),
    "1h": ("H1", 100),
    "30min": ("M30", 100),
    "15min": ("M15", 120),
    "5min": ("M5", 120),
}

# Free-tier Twelve Data thường giới hạn ~8 request/phút. Nghỉ giữa các lần gọi
# để trải đều 7 request trong khoảng thời gian an toàn, tránh bị 429.
TWELVEDATA_REQUEST_DELAY_SEC = 1.2

# Lịch kinh tế công khai (không chính thức, không SLA) — dùng để bù lại phần
# "Macro Filter" trong quy trình gốc, vì Gemini Vision không tự tra cứu được tin tức.
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_LOOKAHEAD_HOURS = 12       # đưa vào prompt các sự kiện USD trong khoảng này
HIGH_IMPACT_VETO_MINUTES = 60   # High impact trong X phút tới => bắt buộc NO TRADE
NEWS_RELEVANT_CURRENCIES = {"USD"}  # XAUUSD nhạy nhất với dữ liệu/lãi suất USD

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


# ---------- Lịch kinh tế (Macro Filter) ----------
def fetch_economic_calendar():
    """Lấy lịch kinh tế tuần này từ nguồn công khai (ForexFactory / Fair Economy feed).
    Đây là nguồn KHÔNG chính thức, không có SLA — nếu lỗi, bot vẫn chạy tiếp, chỉ là
    không có ngữ cảnh tin tức cho lần phân tích đó (được ghi rõ trong thông báo)."""
    try:
        r = requests.get(FF_CALENDAR_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"CẢNH BÁO: không lấy được lịch kinh tế: {e}", file=sys.stderr)
        return None


def parse_calendar_events(raw_events, now_utc: datetime, lookahead_hours: int = NEWS_LOOKAHEAD_HOURS):
    """Lọc các sự kiện thuộc NEWS_RELEVANT_CURRENCIES, sắp diễn ra trong lookahead_hours
    tới, trả về danh sách đã sắp xếp theo thời gian gần nhất trước."""
    if not raw_events:
        return []
    upcoming = []
    for ev in raw_events:
        try:
            ev_time_utc = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except Exception:
            continue
        if ev.get("country") not in NEWS_RELEVANT_CURRENCIES:
            continue
        delta_min = (ev_time_utc - now_utc).total_seconds() / 60
        if 0 <= delta_min <= lookahead_hours * 60:
            upcoming.append(
                {
                    "title": ev.get("title", "N/A"),
                    "country": ev.get("country", "N/A"),
                    "impact": ev.get("impact", "N/A"),
                    "minutes_until": round(delta_min),
                }
            )
    upcoming.sort(key=lambda x: x["minutes_until"])
    return upcoming


def build_news_context(upcoming_events):
    """Tạo đoạn text mô tả lịch tin tức để đưa vào prompt, và xác định có nên ép
    NO TRADE (veto cứng) theo quy tắc 'High impact trong 60 phút tới' hay không.
    Trả về (text_cho_prompt, co_veto: bool)."""
    if upcoming_events is None:
        return (
            "Không lấy được lịch kinh tế lần này (nguồn lỗi/không truy cập được) — "
            "coi rủi ro tin tức là CHƯA XÁC MINH ĐƯỢC, không phải là 'không có tin'.",
            False,
        )

    if not upcoming_events:
        return (
            f"Không có sự kiện USD nào trong {NEWS_LOOKAHEAD_HOURS} giờ tới theo lịch ForexFactory.",
            False,
        )

    veto = any(
        e["impact"] == "High" and e["minutes_until"] <= HIGH_IMPACT_VETO_MINUTES
        for e in upcoming_events
    )

    lines = [
        f"- [{e['impact']}] {e['title']} ({e['country']}) — còn {e['minutes_until']} phút"
        for e in upcoming_events[:8]
    ]
    text = f"Lịch kinh tế {NEWS_LOOKAHEAD_HOURS}h tới (nguồn ForexFactory, chỉ USD):\n" + "\n".join(lines)
    if veto:
        text += (
            f"\n⚠️ CÓ sự kiện HIGH IMPACT trong vòng {HIGH_IMPACT_VETO_MINUTES} phút tới. "
            f"Theo quy tắc bắt buộc: PHẢI kết luận final_verdict = NO TRADE lần này, "
            f"bất kể phân tích kỹ thuật cho kết quả gì."
        )
    return text, veto


# ---------- Bước 3: Gọi Gemini Vision ----------
SYSTEM_PROMPT = """
Bạn là chuyên gia phân tích kỹ thuật XAUUSD theo phương pháp Smart Money Concept (SMC)
kết hợp Top-Down Multi-Timeframe. Mục tiêu: đưa ra tín hiệu có xác suất cao dựa trên cấu
trúc giá thực tế — không đoán mò, nhưng cũng KHÔNG tê liệt chỉ vì thiếu vài dữ liệu phụ
trợ (volume, DXY...) mà hệ thống này vốn dĩ không có.

NGUYÊN TẮC:
- Bảo toàn vốn là ưu tiên, nhưng KHÔNG đồng nghĩa với việc mặc định NO TRADE mỗi khi
  thiếu dữ liệu phụ. Chỉ kết luận NO TRADE khi bản thân cấu trúc giá không đủ rõ ràng,
  hoặc khi có veto tin tức thật sự (xem bước 0) — không phải vì "không có volume/DXY".
- Khung thời gian lớn (D1/H4) quyết định bias chính; khung nhỏ (M15/M5) chỉ dùng để tìm
  điểm vào lệnh khớp bias đó, không dùng để đảo ngược bias trừ khi có tín hiệu đảo chiều
  thật sự rõ ràng (CHOCH + phá vỡ cấu trúc trên chính khung lớn).

=== DỮ LIỆU ĐƯỢC CUNG CẤP ===
{so_luong_anh} ảnh chart XAUUSD theo thứ tự top-down: {danh_sach_khung}.
{atr_context}
{news_context}

Bot KHÔNG có Volume/DXY/US10Y trong lần chạy này — đây là giới hạn đã biết từ trước,
không phải lý do để tự động hạ tiêu chuẩn hay NO TRADE. Chỉ ghi chú "không có dữ liệu
này" ở volume_note, KHÔNG dùng nó để trừ điểm hay chặn tín hiệu.

=== QUY TRÌNH PHÂN TÍCH ===
0. News Filter: đọc {news_context} ở trên. Nếu có cảnh báo veto (High impact trong
   {high_impact_veto_minutes} phút tới) → final_verdict PHẢI là "NO TRADE" ngay lập tức,
   bỏ qua các bước còn lại. Tin Medium/Low chỉ cần ghi chú trong news_risk_note, không
   cần hạ tiêu chuẩn phân tích kỹ thuật.
1. HTF Bias: xác định xu hướng D1 + H4. Đồng thuận rõ ràng → bias mạnh. Mâu thuẫn → bias
   yếu (vẫn có thể giao dịch nếu H1 xác nhận đảo chiều rõ ràng, nhưng ghi chú thận trọng).
2. Liquidity & Order Block: tìm vùng thanh khoản (equal high/low vừa bị quét) và order
   block/FVG gần nhất còn hiệu lực trên H1/H4, cùng hướng với HTF bias.
3. LTF Confirmation: trên M15/M5, tìm xác nhận cấu trúc (BOS/CHOCH, momentum candle,
   engulfing...) khớp hướng giao dịch, xảy ra đúng tại vùng order block/thanh khoản đó.
4. Volatility check: dùng ATR đã cho để xác nhận TP dự kiến khả thi trong thời gian giữ
   lệnh đề xuất.
5. Session: phiên hiện tại có thuận lợi không — đây là yếu tố ĐIỀU CHỈNH độ tin cậy
   (cộng/trừ vài %), KHÔNG phải điều kiện loại trừ.

=== CHECKLIST QUYẾT ĐỊNH (core_checklist, đánh giá true/false) ===
- htf_bias_ro_rang: D1+H4 đồng thuận, HOẶC có đảo chiều H1 được xác nhận rõ ràng.
- vung_thanh_khoan_ob_hop_le: có order block/vùng thanh khoản cụ thể, chưa bị phá vỡ.
- xac_nhan_ltf: có xác nhận cấu trúc rõ ràng trên M15/M5 tại đúng vùng đó.
- rr_hop_ly: R:R tới TP1 ước tính ≥ 1:1.2 và khả thi theo ATR.

QUY TẮC RA QUYẾT ĐỊNH (bắt buộc tuân thủ đúng như sau, không tự thêm điều kiện khác):
- Cả 4 mục đều true → final_verdict = "BUY"/"SELL" theo hướng bias. confidence_percent
  phản ánh đúng mức độ rõ ràng thực tế quan sát được (thường rơi vào khoảng 65-90%).
  Session thuận lợi có thể cộng thêm vài %, không có ngưỡng tối thiểu nào khác.
- Đúng 3/4 mục true (không thiếu htf_bias_ro_rang hoặc xac_nhan_ltf) → vẫn có thể ra
  BUY/SELL nhưng confidence thấp hơn (45-60%), nêu rõ mục nào yếu trong executive_summary.
- Thiếu từ 2 mục trở lên, HOẶC htf_bias_ro_rang=false, HOẶC xac_nhan_ltf=false →
  final_verdict = "NO TRADE".

=== OUTPUT CÒN LẠI ===
- market_regime + regime_reason: mô tả ngắn gọn (Trending/Range/Expansion/Compression...).
- mtf_summary, smc_summary (thanh khoản/OB/FVG), price_action_summary: ngắn gọn, cụ thể,
  nêu đúng mức giá quan sát được trên chart.
- Nếu BUY/SELL: trade_plan đầy đủ — entry, SL, TP1/TP2/TP3. TP PHẢI được xác định BẰNG
  CÙNG PHƯƠNG PHÁP đã dùng để tìm entry (vùng thanh khoản/order block kế tiếp cùng logic),
  tuyệt đối KHÔNG áp đặt một tỷ lệ R:R cố định tách rời khỏi cấu trúc giá thực tế. Kèm
  break-even, trailing stop, kế hoạch chốt lời từng phần, thời gian giữ lệnh tối đa (ưu
  tiên 1-6 giờ — đủ để cấu trúc entry M5/M15 phát triển theo H1/H4, không quá dài để
  tránh sideway/tin tức/phí qua đêm ăn mòn lợi nhuận).
- risk_management: % rủi ro tối đa/lệnh, khối lượng đề xuất, lỗ tối đa ngày/tuần,
  scale-in/scale-out.
- invalidation: hành động giá cụ thể nào sẽ vô hiệu hóa setup.
- executive_summary: 1 đoạn tóm tắt ngắn gọn, rõ ràng bằng tiếng Việt, nêu rõ checklist
  nào đạt/không đạt.

Chỉ trả lời theo đúng JSON schema đã cấu hình, không thêm văn bản ngoài JSON.
"""


def build_prompt(image_labels, atr_h1, atr_h4, news_context: str) -> str:
    if atr_h1 or atr_h4:
        atr_context = (
            f"Biên độ tham khảo (ATR 14 kỳ): H1 ≈ {atr_h1 if atr_h1 else 'N/A'} USD, "
            f"H4 ≈ {atr_h4 if atr_h4 else 'N/A'} USD. Dùng để kiểm tra TP có khả thi "
            f"trong khung thời gian giữ lệnh đề xuất hay không.\n"
        )
    else:
        atr_context = "Không tính được ATR lần này (thiếu dữ liệu) — đánh giá volatility dựa trên quan sát biên độ nến trực tiếp trên chart.\n"
    return SYSTEM_PROMPT.format(
        so_luong_anh=len(image_labels),
        danh_sach_khung=", ".join(image_labels),
        atr_context=atr_context,
        news_context=news_context,
        high_impact_veto_minutes=HIGH_IMPACT_VETO_MINUTES,
    )


# responseSchema ép Gemini trả đúng cấu trúc JSON, thay vì tự parse văn bản tự do
# (đáng tin cậy hơn nhiều so với yêu cầu "chỉ trả JSON" bằng lời).
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "final_verdict": {"type": "STRING", "enum": ["BUY", "SELL", "NO TRADE"]},
        "confidence_percent": {"type": "NUMBER"},
        "market_regime": {"type": "STRING"},
        "regime_reason": {"type": "STRING"},
        "mtf_alignment_percent": {"type": "NUMBER"},
        "mtf_summary": {"type": "STRING"},
        "smc_summary": {"type": "STRING"},
        "supply_demand_summary": {"type": "STRING"},
        "price_action_summary": {"type": "STRING"},
        "volume_note": {"type": "STRING"},
        "volatility_note": {"type": "STRING"},
        "session_note": {"type": "STRING"},
        "news_risk_note": {"type": "STRING"},
        "core_checklist": {
            "type": "OBJECT",
            "properties": {
                "htf_bias_ro_rang": {"type": "BOOLEAN"},
                "vung_thanh_khoan_ob_hop_le": {"type": "BOOLEAN"},
                "xac_nhan_ltf": {"type": "BOOLEAN"},
                "rr_hop_ly": {"type": "BOOLEAN"},
            },
        },
        "trade_plan": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "bias": {"type": "STRING"},
                "entry": {"type": "NUMBER"},
                "stop_loss": {"type": "NUMBER"},
                "tp1": {"type": "NUMBER"},
                "tp2": {"type": "NUMBER", "nullable": True},
                "tp3": {"type": "NUMBER", "nullable": True},
                "risk_reward": {"type": "STRING"},
                "probability_percent": {"type": "NUMBER"},
                "break_even_at": {"type": "STRING"},
                "trailing_stop_plan": {"type": "STRING"},
                "partial_close_plan": {"type": "STRING"},
                "max_holding_time": {"type": "STRING"},
                "tp_method_note": {"type": "STRING"},
                "alternative_scenario": {"type": "STRING"},
            },
        },
        "risk_management": {
            "type": "OBJECT",
            "properties": {
                "max_risk_percent": {"type": "STRING"},
                "suggested_position_size": {"type": "STRING"},
                "max_daily_loss": {"type": "STRING"},
                "max_weekly_loss": {"type": "STRING"},
                "scaling_in": {"type": "STRING"},
                "scaling_out": {"type": "STRING"},
            },
        },
        "invalidation": {"type": "STRING"},
        "key_liquidity": {"type": "STRING"},
        "key_order_block": {"type": "STRING"},
        "key_risk": {"type": "STRING"},
        "missing_data_note": {"type": "STRING"},
        "executive_summary": {"type": "STRING"},
    },
    "required": ["final_verdict", "confidence_percent", "market_regime", "executive_summary"],
}


def call_gemini(images: dict, news_context: str, atr_h1=None, atr_h4=None) -> dict:
    prompt = build_prompt(list(images.keys()), atr_h1, atr_h4, news_context)
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
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    r = requests.post(GEMINI_URL, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    # Phòng hờ: dù đã ép responseSchema, vẫn giữ bước dọn markdown fence để an toàn.
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


def calc_risk_reward(entry, sl, tp1):
    """Tự tính khoảng cách SL/TP1 (USD) và tỷ lệ R:R thay vì phụ thuộc AI."""
    if not all(isinstance(x, (int, float)) for x in (entry, sl, tp1)):
        return None, None, None
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp1)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
    return sl_dist, tp_dist, rr


def get_session_label(now_utc: datetime) -> str:
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


def confidence_visual(percent):
    if not isinstance(percent, (int, float)):
        return "⬜⬜⬜⬜⬜", "N/A"
    filled = min(5, max(0, round(percent / 20)))
    bar = "🟩" * filled + "⬜" * (5 - filled)
    if percent >= 75:
        label = "Cao"
    elif percent >= 60:
        label = "Trung bình"
    elif percent >= 45:
        label = "Thấp"
    else:
        label = "Rất thấp / NO TRADE"
    return bar, label


def format_message(signal: dict) -> str:
    now_utc = datetime.now(timezone.utc)
    now_vn = now_utc.astimezone(timezone(timedelta(hours=7)))

    verdict = signal.get("final_verdict", "NO TRADE")
    emoji = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪️"}.get(verdict, "⚪️")
    conf_pct = signal.get("confidence_percent")
    bar, conf_label = confidence_visual(conf_pct)

    tp = signal.get("trade_plan") or {}
    entry, sl, tp1 = tp.get("entry"), tp.get("stop_loss"), tp.get("tp1")
    sl_dist, tp_dist, rr_calc = calc_risk_reward(entry, sl, tp1)

    lines = [
        f"{emoji} <b>XAUUSD — {verdict}</b>",
        "━━━━━━━━━━━━━━━━━━━",
        f"🕒 <b>Giờ VN:</b> {now_vn.strftime('%d/%m/%Y %H:%M')}  |  <b>UTC:</b> {now_utc.strftime('%H:%M')}",
        f"🌍 <b>Phiên:</b> {get_session_label(now_utc)}",
        f"📊 <b>Độ tin cậy:</b> {bar} {conf_pct if conf_pct is not None else 'N/A'}% ({conf_label})",
        "",
        f"🏛 <b>Market Regime:</b> {signal.get('market_regime', 'N/A')}",
    ]
    if signal.get("regime_reason"):
        lines.append(f"↳ {signal.get('regime_reason')}")
    if signal.get("mtf_alignment_percent") is not None:
        lines.append(f"📐 <b>Đồng thuận đa khung:</b> {signal.get('mtf_alignment_percent')}%")
    if signal.get("mtf_summary"):
        lines.append(f"📈 <b>MTF:</b> {signal.get('mtf_summary')}")
    if signal.get("smc_summary"):
        lines.append(f"💧 <b>Smart Money:</b> {signal.get('smc_summary')}")
    if signal.get("supply_demand_summary"):
        lines.append(f"🧱 <b>Supply/Demand:</b> {signal.get('supply_demand_summary')}")
    if signal.get("price_action_summary"):
        lines.append(f"🕯 <b>Price Action:</b> {signal.get('price_action_summary')}")

    checklist = signal.get("core_checklist") or {}
    if checklist:
        label_map = {
            "htf_bias_ro_rang": "HTF Bias rõ ràng",
            "vung_thanh_khoan_ob_hop_le": "Thanh khoản/OB hợp lệ",
            "xac_nhan_ltf": "Xác nhận LTF (M15/M5)",
            "rr_hop_ly": "R:R hợp lý theo ATR",
        }
        lines.append("")
        lines.append("🧮 <b>Checklist:</b>")
        for k, label in label_map.items():
            v = checklist.get(k)
            mark = "✅" if v else ("❌" if v is False else "➖")
            lines.append(f"{mark} {label}")

    if verdict in ("BUY", "SELL") and tp:
        lines.append("")
        lines.append("━━━ <b>KẾ HOẠCH VÀO LỆNH</b> ━━━")
        if tp.get("bias"):
            lines.append(f"🎯 <b>Bias:</b> {tp.get('bias')}")
        lines.append(f"💰 <b>Entry:</b>  <code>{_fmt_price(entry)}</code>")
        lines.append(f"🛑 <b>Stop Loss:</b>  <code>{_fmt_price(sl)}</code>" + (f"  (~{sl_dist:.2f} pt)" if sl_dist else ""))
        lines.append(f"✅ <b>TP1:</b>  <code>{_fmt_price(tp1)}</code>" + (f"  (~{tp_dist:.2f} pt)" if tp_dist else ""))
        if tp.get("tp2") is not None:
            lines.append(f"✅ <b>TP2:</b>  <code>{_fmt_price(tp.get('tp2'))}</code>")
        if tp.get("tp3") is not None:
            lines.append(f"✅ <b>TP3:</b>  <code>{_fmt_price(tp.get('tp3'))}</code>")
        rr_display = tp.get("risk_reward") or (f"1 : {rr_calc}" if rr_calc else "N/A")
        lines.append(f"⚖️ <b>R:R (tới TP1):</b> {rr_display}")
        if tp.get("probability_percent") is not None:
            lines.append(f"🎲 <b>Xác suất kịch bản:</b> {tp.get('probability_percent')}%")
        if tp.get("tp_method_note"):
            lines.append(f"📐 <b>Căn cứ chọn TP:</b> {tp.get('tp_method_note')}")

        lines.append("")
        lines.append("━━━ <b>QUẢN LÝ LỆNH</b> ━━━")
        lines.append(f"⏳ <b>Thời gian giữ lệnh tối đa:</b> {tp.get('max_holding_time', 'N/A')}")
        if tp.get("break_even_at"):
            lines.append(f"⚖️ <b>Dời SL về hòa vốn tại:</b> {tp.get('break_even_at')}")
        if tp.get("trailing_stop_plan"):
            lines.append(f"📉 <b>Trailing stop:</b> {tp.get('trailing_stop_plan')}")
        if tp.get("partial_close_plan"):
            lines.append(f"✂️ <b>Chốt lời từng phần:</b> {tp.get('partial_close_plan')}")
        if tp.get("alternative_scenario"):
            lines.append(f"🔄 <b>Kịch bản thay thế:</b> {tp.get('alternative_scenario')}")

    rm = signal.get("risk_management") or {}
    if rm:
        lines.append("")
        lines.append("━━━ <b>RISK MANAGEMENT</b> ━━━")
        if rm.get("max_risk_percent"):
            lines.append(f"⚠️ <b>Rủi ro tối đa/lệnh:</b> {rm.get('max_risk_percent')}")
        if rm.get("suggested_position_size"):
            lines.append(f"📏 <b>Khối lượng đề xuất:</b> {rm.get('suggested_position_size')}")
        if rm.get("max_daily_loss"):
            lines.append(f"📅 <b>Lỗ tối đa/ngày:</b> {rm.get('max_daily_loss')}")
        if rm.get("max_weekly_loss"):
            lines.append(f"🗓 <b>Lỗ tối đa/tuần:</b> {rm.get('max_weekly_loss')}")
        if rm.get("scaling_in"):
            lines.append(f"➕ <b>Scale-in:</b> {rm.get('scaling_in')}")
        if rm.get("scaling_out"):
            lines.append(f"➖ <b>Scale-out:</b> {rm.get('scaling_out')}")

    if signal.get("invalidation"):
        lines.append("")
        lines.append(f"🚫 <b>Invalidation:</b> {signal.get('invalidation')}")

    key_bits = []
    if signal.get("key_liquidity"):
        key_bits.append(f"💧 Thanh khoản chính: {signal.get('key_liquidity')}")
    if signal.get("key_order_block"):
        key_bits.append(f"🧱 Order Block chính: {signal.get('key_order_block')}")
    if signal.get("key_risk"):
        key_bits.append(f"⚠️ Rủi ro chính: {signal.get('key_risk')}")
    if key_bits:
        lines.append("")
        lines.extend(key_bits)

    if signal.get("volume_note"):
        lines.append("")
        lines.append(f"🔇 <b>Volume:</b> {signal.get('volume_note')}")
    if signal.get("missing_data_note"):
        lines.append(f"❓ <b>Dữ liệu còn thiếu:</b> {signal.get('missing_data_note')}")

    upcoming_news = signal.get("_upcoming_news") or []
    if upcoming_news or signal.get("news_risk_note"):
        lines.append("")
        lines.append("━━━ <b>LỊCH TIN TỨC (USD)</b> ━━━")
        if upcoming_news:
            for e in upcoming_news[:6]:
                impact_emoji = {"High": "🔴", "Medium": "🟠", "Low": "⚪️", "Holiday": "🏖"}.get(e["impact"], "⚪️")
                lines.append(f"{impact_emoji} {e['title']} — còn {e['minutes_until']} phút")
        if signal.get("news_risk_note"):
            lines.append(f"📰 <b>Đánh giá rủi ro tin tức:</b> {signal.get('news_risk_note')}")

    lines.append("")
    lines.append(f"📝 <b>Executive Summary:</b> {signal.get('executive_summary', '')}")
    lines.append("")
    lines.append(
        "⚠️ <i>Đây là output từ AI dựa trên dữ liệu giá + chart (không có volume/macro/tin "
        "tức thời gian thực). Không phải khuyến nghị đầu tư. Tự quản lý rủi ro và vốn.</i>"
    )
    return "\n".join(lines)


# ---------- Main ----------
def main():
    images = {}
    dfs = {}
    for i, (tv_interval, (label, size)) in enumerate(TIMEFRAMES.items()):
        try:
            df = fetch_ohlc(tv_interval, size)
            images[label] = render_chart_png(df, label)
            dfs[label] = df
            print(f"OK: lấy dữ liệu và vẽ chart {label}")
        except Exception as e:
            print(f"LỖI khi xử lý khung {label}: {e}", file=sys.stderr)
        if i < len(TIMEFRAMES) - 1:
            time.sleep(TWELVEDATA_REQUEST_DELAY_SEC)

    if len(images) < 3:
        send_telegram("⚠️ Bot lỗi: không lấy đủ dữ liệu chart để phân tích lần này.")
        sys.exit(1)

    atr_h1 = calc_atr(dfs.get("H1"))
    atr_h4 = calc_atr(dfs.get("H4"))

    now_utc = datetime.now(timezone.utc)
    raw_calendar = fetch_economic_calendar()
    upcoming_events = parse_calendar_events(raw_calendar, now_utc)
    news_context, news_veto = build_news_context(upcoming_events)
    if news_veto:
        print("CẢNH BÁO: có tin tức High Impact trong khung giờ veto — sẽ ép NO TRADE.")

    try:
        signal = call_gemini(images, news_context, atr_h1=atr_h1, atr_h4=atr_h4)
    except Exception as e:
        print(f"LỖI khi gọi Gemini: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini: {e}")
        sys.exit(1)

    # Ép cứng bằng code, không chỉ dựa vào việc model tuân thủ prompt: nếu có sự kiện
    # High Impact trong khung giờ veto, luôn buộc NO TRADE bất kể Gemini trả về gì.
    if news_veto and signal.get("final_verdict") != "NO TRADE":
        signal["final_verdict"] = "NO TRADE"
        signal["trade_plan"] = None
        signal["news_risk_note"] = (
            (signal.get("news_risk_note") or "")
            + " [Hệ thống tự động ép NO TRADE do có tin tức High Impact sắp diễn ra, "
            "bất kể model đề xuất gì.]"
        ).strip()

    signal["_upcoming_news"] = upcoming_events or []

    message = format_message(signal)

    chart_for_photo = images.get("H1") or next(iter(images.values()), None)
    try:
        if chart_for_photo:
            verdict = signal.get("final_verdict", "NO TRADE")
            emoji = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪️"}.get(verdict, "⚪️")
            send_telegram_photo(chart_for_photo, caption=f"{emoji} XAUUSD H1 — {verdict}")
    except Exception as e:
        print(f"CẢNH BÁO: gửi ảnh chart thất bại (bỏ qua, vẫn gửi text): {e}", file=sys.stderr)

    send_telegram(message)
    print("Đã gửi tín hiệu lên Telegram thành công.")


if __name__ == "__main__":
    main()
