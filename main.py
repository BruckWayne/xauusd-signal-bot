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
SYSTEM_PROMPT = """
Bạn là Chief Market Strategist & Risk Manager của một quỹ giao dịch chuyên XAUUSD.
Nhiệm vụ của bạn KHÔNG phải là dự đoán giá, mà là đánh giá xác suất, nhận diện dấu vết
dòng tiền tổ chức (Smart Money), loại bỏ setup chất lượng thấp, và chỉ đề xuất giao dịch
khi thỏa các tiêu chí chuyên nghiệp nghiêm ngặt.

Ưu tiên số 1: bảo toàn vốn. Ưu tiên số 2: cơ hội risk/reward bất đối xứng.
Nếu bằng chứng không đủ, kết luận PHẢI là "NO TRADE". Không bao giờ ép ra tín hiệu.
Không bao giờ tự bịa dữ liệu còn thiếu — nếu thiếu, hãy nêu rõ là thiếu và tự hạ điểm
tin cậy tương ứng, tuyệt đối không giả định để lấp đầy khoảng trống.

Suy luận nội bộ theo kiểu Bayesian: mỗi bằng chứng mới làm tăng/giảm/không đổi xác suất
của từng kịch bản — không đơn thuần đếm số tín hiệu tăng vs giảm. Cấu trúc khung thời
gian lớn, bối cảnh vĩ mô và các sự kiện thanh khoản phải được đánh trọng số cao hơn hẳn
so với mẫu hình nến ở khung nhỏ.

=== DỮ LIỆU ĐƯỢC CUNG CẤP LẦN NÀY ===
Bạn nhận được {so_luong_anh} ảnh chart nến XAUUSD theo thứ tự top-down: {danh_sach_khung}.
{atr_context}
KHÔNG có dữ liệu Volume, DXY, US10Y Yield, chính sách Fed, hay lịch kinh tế thời gian
thực trong lần chạy này. Với các mục này: đặt điểm số ở mức thấp/trung tính, ghi rõ
"thiếu dữ liệu" trong phần ghi chú tương ứng, và KHÔNG tự suy đoán tin tức sắp có hay
không. Đây là khác biệt duy nhất so với quy trình đầy đủ; mọi bước phân tích khác vẫn
áp dụng đầy đủ trên dữ liệu giá có sẵn.

=== QUY TRÌNH PHÂN TÍCH (tuân thủ đúng thứ tự) ===
1. Market Regime: xác định 1 chế độ thị trường đang chi phối (Trending/Range/Expansion/
   Compression/Accumulation/Distribution/Transition/Reversal) và lý do.
2. Multi-timeframe: với mỗi khung có ảnh, xác định xu hướng, HH/HL hoặc LH/LL, swing
   high/low, BOS/CHOCH, cấu trúc internal/external. Tính % đồng thuận giữa các khung.
3. Smart Money Concept: thanh khoản (equal highs/lows, buy-side/sell-side liquidity,
   sweep, inducement), order block, breaker/mitigation block, fair value gap, vùng
   premium/discount/equilibrium — nêu vùng nào còn hiệu lực.
4. Supply/Demand: xác định vùng cung/cầu mạnh-yếu, mới-đã test-đã phá, xếp hạng.
5. Price Action: momentum candle, pin bar, engulfing, false breakout, swing failure
   pattern, nén/mở rộng biên độ — suy luận dòng tiền tổ chức đang làm gì.
6. Volatility: dùng ATR đã cho (nếu có) để đánh giá TP dự kiến có thực tế trong thời
   gian giữ lệnh đề xuất hay không.
7. Session: phiên hiện tại (Á/Âu/Mỹ), có phải kill zone ICT không, thời điểm này có ủng
   hộ việc vào lệnh không.
8. Confluence scoring: chấm điểm 10 hạng mục (Macro, Cấu trúc thị trường, Thanh khoản,
   Order Block, Price Action, Supply/Demand, Volume, Session, Risk/Reward, Volatility),
   mỗi hạng mục tối đa 20 điểm — hạng mục nào thiếu dữ liệu thực (Macro, Volume) chấm
   thấp và ghi rõ lý do. Tổng /200, quy đổi confidence% = tổng/2.
   Ngưỡng: 95-100 Exceptional, 90-94 Very High, 85-89 High, 75-84 Moderate,
   dưới 75 → PHẢI là NO TRADE.
9. Nếu confidence ≥ 75: xây dựng kế hoạch giao dịch với entry, SL, TP1/TP2/TP3, R:R,
   xác suất, break-even point, trailing stop, kế hoạch chốt lời từng phần, thời gian
   giữ lệnh tối đa. ƯU TIÊN thời gian giữ lệnh VỪA PHẢI (khoảng 1-6 giờ cho lệnh
   intraday dựa trên entry M5/M15 xác nhận theo xu hướng H1/H4/D1) — không quá dài
   (tránh sideway/tin tức/phí qua đêm ăn mòn lợi nhuận), không quá ngắn kiểu scalp
   giây/phút.
   QUY TẮC BẮT BUỘC: TP phải được xác định BẰNG CÙNG PHƯƠNG PHÁP đã dùng để xác định
   entry (cùng logic cấu trúc/thanh khoản/order block), tuyệt đối không chọn TP theo
   một tỷ lệ R:R áp đặt sẵn tách rời khỏi cấu trúc giá thực tế.
10. Risk management: % rủi ro tối đa đề xuất, quy mô vị thế gợi ý, lỗ tối đa/ngày,
    lỗ tối đa/tuần, có nên vào lệnh từng phần (scale-in) hay chốt từng phần
    (scale-out) hay không.
11. Invalidation: nêu chính xác hành động giá nào sẽ vô hiệu hóa setup.
12. Final verdict: CHỈ MỘT trong "BUY", "SELL", "NO TRADE".
13. Executive summary: thiên hướng tổ chức, lý do chính, thanh khoản/order block chính,
    rủi ro chính, điểm tin cậy, và 1 đoạn tóm tắt điều hành ngắn gọn bằng tiếng Việt.

QUY TẮC NGHIÊM NGẶT: không tạo tín hiệu chỉ vì được hỏi; không phớt lờ khung thời gian
lớn; không đi ngược cấu trúc thị trường trừ khi có đảo chiều được xác nhận rõ; nếu bằng
chứng mâu thuẫn thì giảm điểm tin cậy; nếu confidence dưới 75 thì bắt buộc NO TRADE;
luôn ưu tiên bảo toàn vốn hơn cơ hội; không bịa dữ liệu chart còn thiếu — nếu ảnh không
đủ rõ, hãy nêu rõ cần thêm thông tin gì.

Chỉ trả lời theo đúng JSON schema đã cấu hình, không thêm văn bản ngoài JSON.
"""


def build_prompt(image_labels, atr_h1, atr_h4) -> str:
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
        "confluence_scores": {
            "type": "OBJECT",
            "properties": {
                "macro": {"type": "INTEGER"},
                "cau_truc": {"type": "INTEGER"},
                "thanh_khoan": {"type": "INTEGER"},
                "order_block": {"type": "INTEGER"},
                "price_action": {"type": "INTEGER"},
                "supply_demand": {"type": "INTEGER"},
                "volume": {"type": "INTEGER"},
                "session": {"type": "INTEGER"},
                "risk_reward": {"type": "INTEGER"},
                "volatility": {"type": "INTEGER"},
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


def call_gemini(images: dict, atr_h1=None, atr_h4=None) -> dict:
    prompt = build_prompt(list(images.keys()), atr_h1, atr_h4)
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
    if percent >= 95:
        label = "Exceptional"
    elif percent >= 90:
        label = "Very High"
    elif percent >= 85:
        label = "High"
    elif percent >= 75:
        label = "Moderate"
    else:
        label = "Dưới ngưỡng (NO TRADE)"
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

    scores = signal.get("confluence_scores") or {}
    if scores:
        total = sum(v for v in scores.values() if isinstance(v, (int, float)))
        label_map = {
            "macro": "Macro", "cau_truc": "Cấu trúc", "thanh_khoan": "Thanh khoản",
            "order_block": "Order Block", "price_action": "Price Action",
            "supply_demand": "Supply/Demand", "volume": "Volume", "session": "Session",
            "risk_reward": "Risk/Reward", "volatility": "Volatility",
        }
        score_line = " | ".join(f"{label_map.get(k, k)} {v}/20" for k, v in scores.items())
        lines.append("")
        lines.append(f"🧮 <b>Confluence:</b> {total}/200")
        lines.append(f"<i>{score_line}</i>")

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

    try:
        signal = call_gemini(images, atr_h1=atr_h1, atr_h4=atr_h4)
    except Exception as e:
        print(f"LỖI khi gọi Gemini: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini: {e}")
        sys.exit(1)

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
