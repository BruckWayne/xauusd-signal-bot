"""
XAUUSD Vision Signal Bot — bản nâng cấp
----------------------------------------
Nâng cấp chính:
1. Lấy chart H4/H1/M15/M5 từ Twelve Data.
2. Tính ATR, EMA, swing high/low, market bias để tạo lớp dữ liệu số.
3. Kiểm tra lịch tin tức USD High Impact và ép NO TRADE nếu nằm trong vùng cấm.
4. Gửi ảnh chart có EMA + prompt top-down cho Gemini Vision.
5. Chạy validator sau Gemini: hướng lệnh, entry/SL/TP, R:R, confidence, ATR, news.
6. Ghi log JSON/CSV để đánh giá lịch sử tín hiệu.
7. Gửi Telegram ngắn gọn, có lý do bị chặn nếu không đủ điều kiện.

Biến môi trường bắt buộc:
- TWELVEDATA_API_KEY
- GEMINI_API_KEY
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID

Biến môi trường tùy chọn:
- MIN_CONFIDENCE=70
- MIN_RR=1.2
- MAX_ENTRY_DISTANCE_ATR=0.45
- NEWS_LOOKAHEAD_HOURS=6
- HIGH_IMPACT_VETO_BEFORE_MINUTES=60
- HIGH_IMPACT_VETO_AFTER_MINUTES=30
- SEND_NO_TRADE=true
- LOG_DIR=logs
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mplfinance as mpf
import pandas as pd
import requests

# ---------- Cấu hình ----------
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = os.getenv("SYMBOL", "XAU/USD")
TIMEFRAMES = {
    "4h": ("H4", 160),
    "1h": ("H1", 180),
    "15min": ("M15", 180),
    "5min": ("M5", 180),
}
TWELVEDATA_REQUEST_DELAY_SEC = float(os.getenv("TWELVEDATA_REQUEST_DELAY_SEC", "1.0"))

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_LOOKAHEAD_HOURS = int(os.getenv("NEWS_LOOKAHEAD_HOURS", "6"))
HIGH_IMPACT_VETO_BEFORE_MINUTES = int(os.getenv("HIGH_IMPACT_VETO_BEFORE_MINUTES", "60"))
HIGH_IMPACT_VETO_AFTER_MINUTES = int(os.getenv("HIGH_IMPACT_VETO_AFTER_MINUTES", "30"))
NEWS_RELEVANT_CURRENCIES = {"USD"}
CRITICAL_NEWS_KEYWORDS = ("CPI", "PPI", "NFP", "Non-Farm", "FOMC", "Federal Funds", "Fed Chair", "Powell", "Unemployment", "GDP")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "70"))
MIN_RR = float(os.getenv("MIN_RR", "1.2"))
MAX_ENTRY_DISTANCE_ATR = float(os.getenv("MAX_ENTRY_DISTANCE_ATR", "0.45"))
MAX_SL_ATR_MULTIPLIER = float(os.getenv("MAX_SL_ATR_MULTIPLIER", "1.35"))
MIN_SL_ATR_MULTIPLIER = float(os.getenv("MIN_SL_ATR_MULTIPLIER", "0.08"))
SEND_NO_TRADE = os.getenv("SEND_NO_TRADE", "true").lower() in {"1", "true", "yes", "y"}
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

VN_TZ = timezone(timedelta(hours=7))


@dataclass
class NewsStatus:
    text: str
    veto: bool
    event: dict[str, Any] | None
    calendar_available: bool


# ---------- Dữ liệu & chỉ báo ----------
def fetch_ohlc(interval: str, outputsize: int) -> pd.DataFrame:
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": SYMBOL,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVEDATA_API_KEY,
            "format": "JSON",
            "order": "ASC",
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
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    return out


def calc_atr(df: pd.DataFrame | None, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 2) if pd.notna(atr) else None


def detect_swings(df: pd.DataFrame, lookback: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    if len(df) < lookback * 2 + 3:
        return highs, lows
    for i in range(lookback, len(df) - lookback):
        window = df.iloc[i - lookback : i + lookback + 1]
        row = df.iloc[i]
        dt = df.index[i]
        if row["high"] == window["high"].max():
            highs.append({"time": str(dt), "price": round(float(row["high"]), 2)})
        if row["low"] == window["low"].min():
            lows.append({"time": str(dt), "price": round(float(row["low"]), 2)})
    return highs[-5:], lows[-5:]


def infer_market_bias(df: pd.DataFrame) -> str:
    if len(df) < 55:
        return "unknown"
    last = df.iloc[-1]
    ema20, ema50 = last.get("ema20"), last.get("ema50")
    if pd.isna(ema20) or pd.isna(ema50):
        return "unknown"
    close = last["close"]
    if close > ema20 > ema50:
        return "bullish"
    if close < ema20 < ema50:
        return "bearish"
    return "sideway/mixed"


def build_market_context(dfs: dict[str, pd.DataFrame], atr_h1: float | None) -> dict[str, Any]:
    context: dict[str, Any] = {"symbol": SYMBOL, "atr_h1": atr_h1, "timeframes": {}}
    for label, df in dfs.items():
        highs, lows = detect_swings(df)
        last = df.iloc[-1]
        context["timeframes"][label] = {
            "last_close": round(float(last["close"]), 2),
            "ema20": round(float(last["ema20"]), 2) if pd.notna(last.get("ema20")) else None,
            "ema50": round(float(last["ema50"]), 2) if pd.notna(last.get("ema50")) else None,
            "ema200": round(float(last["ema200"]), 2) if pd.notna(last.get("ema200")) else None,
            "bias": infer_market_bias(df),
            "recent_swing_highs": highs,
            "recent_swing_lows": lows,
        }
    return context


def render_chart_png(df: pd.DataFrame, label: str) -> bytes:
    buf = io.BytesIO()
    apds = []
    for col in ["ema20", "ema50", "ema200"]:
        if col in df and df[col].notna().sum() > 5:
            apds.append(mpf.make_addplot(df[col], width=0.8))
    hline = float(df["close"].iloc[-1])
    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"XAUUSD - {label} | last {hline:.2f}",
        volume=False,
        addplot=apds if apds else None,
        hlines=dict(hlines=[hline], linewidths=0.7),
        savefig=dict(fname=buf, dpi=130, bbox_inches="tight"),
        warn_too_much_data=10000,
    )
    buf.seek(0)
    return buf.read()


# ---------- Lịch tin tức ----------
def fetch_economic_calendar() -> list[dict[str, Any]] | None:
    try:
        r = requests.get(FF_CALENDAR_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"CẢNH BÁO: không lấy được lịch kinh tế: {e}", file=sys.stderr)
        return None


def _is_critical_news(title: str) -> bool:
    title_upper = title.upper()
    return any(k.upper() in title_upper for k in CRITICAL_NEWS_KEYWORDS)


def get_next_high_impact_event(raw_events: list[dict[str, Any]] | None, now_utc: datetime) -> dict[str, Any] | None:
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
        if -HIGH_IMPACT_VETO_AFTER_MINUTES <= delta_min <= NEWS_LOOKAHEAD_HOURS * 60:
            title = ev.get("title", "N/A")
            candidates.append(
                {
                    "title": title,
                    "minutes_until": round(delta_min),
                    "is_critical": _is_critical_news(title),
                    "event_time_utc": ev_time.isoformat(),
                }
            )
    return min(candidates, key=lambda x: abs(x["minutes_until"])) if candidates else None


def build_news_status(raw_events: list[dict[str, Any]] | None, now_utc: datetime) -> NewsStatus:
    if raw_events is None:
        return NewsStatus(
            text="Lịch tin tức: không lấy được lần này — rủi ro tin tức chưa xác minh.",
            veto=False,
            event=None,
            calendar_available=False,
        )

    ev = get_next_high_impact_event(raw_events, now_utc)
    if ev is None:
        return NewsStatus(
            text=f"Lịch tin tức: không có USD High Impact trong {NEWS_LOOKAHEAD_HOURS}h tới và không nằm trong {HIGH_IMPACT_VETO_AFTER_MINUTES} phút sau tin.",
            veto=False,
            event=None,
            calendar_available=True,
        )

    minutes = ev["minutes_until"]
    before_veto = 0 <= minutes <= HIGH_IMPACT_VETO_BEFORE_MINUTES
    after_veto = -HIGH_IMPACT_VETO_AFTER_MINUTES <= minutes < 0
    critical_veto = ev.get("is_critical") and 0 <= minutes <= max(HIGH_IMPACT_VETO_BEFORE_MINUTES, 120)
    veto = bool(before_veto or after_veto or critical_veto)

    if minutes >= 0:
        time_text = f"còn {minutes} phút"
    else:
        time_text = f"vừa qua {abs(minutes)} phút"
    text = f"Lịch tin tức: {ev['title']} (USD, High Impact) — {time_text}."
    if veto:
        text += " ⚠️ Thuộc vùng cấm giao dịch → BẮT BUỘC NO TRADE."
    return NewsStatus(text=text, veto=veto, event=ev, calendar_available=True)


# ---------- Prompt & schema ----------
SYSTEM_PROMPT = """
Bạn là trader chuyên nghiệp phân tích XAUUSD theo Price Action + Order Block đa khung thời gian.
Mục tiêu: chỉ đưa tín hiệu intraday chất lượng cao, không cố ép lệnh.

Bạn nhận {so_luong_anh} ảnh chart theo thứ tự lớn → nhỏ: {danh_sach_khung}.
Dữ liệu định lượng do hệ thống tính trước:
{market_context_json}

{news_context}

QUY TRÌNH BẮT BUỘC:
1. Xác định bias H4/H1: bullish, bearish, sideway/mixed. Nếu H4 và H1 xung đột mạnh → ưu tiên NO TRADE.
2. Xác định vùng giá quan trọng gần giá hiện tại: order block, supply/demand, hỗ trợ/kháng cự, liquidity sweep.
3. Chỉ xác nhận entry trên M15/M5 khi có phản ứng rõ tại đúng vùng: BOS/CHOCH, engulfing, pin bar, rejection hoặc retest rõ.
4. BUY/SELL chỉ hợp lệ khi xu hướng lớn, vùng giá và xác nhận nhỏ cùng hướng.
5. Nếu lịch tin tức báo vùng cấm hoặc thiếu xác nhận → NO TRADE.

Nếu BUY/SELL:
- Entry phải sát giá hiện tại hoặc vùng retest hợp lý, không chase khi giá đã chạy xa.
- SL đặt sau cấu trúc bị phá, không đặt tùy tiện theo số tròn.
- TP1/TP2 đặt ở vùng thanh khoản hoặc hỗ trợ/kháng cự kế tiếp.
- Không bịa volume, DXY, US10Y vì hệ thống không cung cấp các dữ liệu đó.

Trả lời đúng JSON schema. Không thêm markdown, không thêm chữ ngoài JSON.
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
    "required": [
        "final_verdict",
        "confidence_percent",
        "xu_huong",
        "vung_gia_quan_trong",
        "tin_hieu_vao_lenh",
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "ly_do_tp",
        "thoi_gian_giu_lenh",
        "dieu_kien_vo_hieu",
        "ghi_chu",
    ],
}


def build_prompt(image_labels: list[str], market_context: dict[str, Any], news_status: NewsStatus) -> str:
    market_context_json = json.dumps(market_context, ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT.format(
        so_luong_anh=len(image_labels),
        danh_sach_khung=", ".join(image_labels),
        market_context_json=market_context_json,
        news_context=news_status.text,
    )


def call_gemini(images: dict[str, bytes], market_context: dict[str, Any], news_status: NewsStatus) -> dict[str, Any]:
    prompt = build_prompt(list(images.keys()), market_context, news_status)
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for _, png_bytes in images.items():
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
            "temperature": 0.15,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    r = requests.post(GEMINI_URL, json=body, timeout=120)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json\n", "", 1).replace("json", "", 1).strip()
    return json.loads(text)


# ---------- Validator tín hiệu ----------
def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def calc_risk_reward(entry: Any, sl: Any, tp1: Any) -> tuple[float | None, float | None, float | None]:
    entry_f, sl_f, tp1_f = _as_float(entry), _as_float(sl), _as_float(tp1)
    if not all(x is not None for x in (entry_f, sl_f, tp1_f)):
        return None, None, None
    sl_dist = abs(entry_f - sl_f)
    tp_dist = abs(entry_f - tp1_f)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
    return round(sl_dist, 2), round(tp_dist, 2), rr


def normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    out = dict(signal)
    verdict = str(out.get("final_verdict", "NO TRADE")).upper().strip()
    if verdict in {"WAIT", "HOLD", "NONE"}:
        verdict = "NO TRADE"
    if verdict not in {"BUY", "SELL", "NO TRADE"}:
        verdict = "NO TRADE"
    out["final_verdict"] = verdict
    out["confidence_percent"] = _as_float(out.get("confidence_percent"))
    for key in ["entry", "stop_loss", "take_profit_1", "take_profit_2"]:
        out[key] = _as_float(out.get(key))
    return out


def force_no_trade(signal: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    out = dict(signal)
    out["final_verdict"] = "NO TRADE"
    old_note = str(out.get("ghi_chu") or "").strip()
    validator_note = "Hệ thống chặn lệnh: " + "; ".join(reasons)
    out["ghi_chu"] = f"{old_note} | {validator_note}" if old_note else validator_note
    return out


def validate_signal(signal: dict[str, Any], market_context: dict[str, Any], news_status: NewsStatus) -> dict[str, Any]:
    signal = normalize_signal(signal)
    reasons: list[str] = []
    verdict = signal.get("final_verdict")
    atr_h1 = _as_float(market_context.get("atr_h1"))
    h1_close = _as_float(market_context.get("timeframes", {}).get("H1", {}).get("last_close"))
    h4_bias = market_context.get("timeframes", {}).get("H4", {}).get("bias")
    h1_bias = market_context.get("timeframes", {}).get("H1", {}).get("bias")

    if news_status.veto:
        reasons.append("có tin USD High Impact trong vùng cấm")

    if not news_status.calendar_available:
        reasons.append("không xác minh được lịch tin tức")

    if verdict in {"BUY", "SELL"}:
        conf = _as_float(signal.get("confidence_percent"))
        entry = _as_float(signal.get("entry"))
        sl = _as_float(signal.get("stop_loss"))
        tp1 = _as_float(signal.get("take_profit_1"))

        if conf is None or conf < MIN_CONFIDENCE:
            reasons.append(f"confidence thấp hơn ngưỡng {MIN_CONFIDENCE:.0f}%")

        if not all(x is not None for x in (entry, sl, tp1)):
            reasons.append("thiếu entry/SL/TP1")
        else:
            if verdict == "BUY":
                if not (sl < entry < tp1):
                    reasons.append("cấu trúc giá BUY sai: cần SL < entry < TP1")
                if h4_bias == "bearish" and h1_bias == "bearish":
                    reasons.append("BUY ngược bias H4/H1 đều bearish")
            elif verdict == "SELL":
                if not (tp1 < entry < sl):
                    reasons.append("cấu trúc giá SELL sai: cần TP1 < entry < SL")
                if h4_bias == "bullish" and h1_bias == "bullish":
                    reasons.append("SELL ngược bias H4/H1 đều bullish")

            sl_dist, _, rr = calc_risk_reward(entry, sl, tp1)
            if rr is None or rr < MIN_RR:
                reasons.append(f"R:R thấp hơn ngưỡng 1:{MIN_RR}")

            if atr_h1 and sl_dist:
                if sl_dist > atr_h1 * MAX_SL_ATR_MULTIPLIER:
                    reasons.append("SL quá rộng so với ATR H1")
                if sl_dist < atr_h1 * MIN_SL_ATR_MULTIPLIER:
                    reasons.append("SL quá sát so với ATR H1")

            if atr_h1 and h1_close and entry:
                entry_distance = abs(entry - h1_close)
                if entry_distance > atr_h1 * MAX_ENTRY_DISTANCE_ATR:
                    reasons.append("entry quá xa giá hiện tại, có nguy cơ chase")

        if reasons:
            return force_no_trade(signal, reasons)

    elif reasons:
        old_note = str(signal.get("ghi_chu") or "").strip()
        extra = "Bộ lọc hệ thống: " + "; ".join(reasons)
        signal["ghi_chu"] = f"{old_note} | {extra}" if old_note else extra

    return signal


# ---------- Telegram ----------
def send_telegram(text: str) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    r.raise_for_status()


def send_telegram_photo(png_bytes: bytes, caption: str = "") -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", png_bytes, "image/png")},
        timeout=30,
    )
    r.raise_for_status()


def _fmt_price(v: Any) -> str:
    f = _as_float(v)
    return f"{f:,.2f}" if f is not None else "N/A"


def confidence_bar(percent: Any) -> str:
    p = _as_float(percent)
    if p is None:
        return "⬜⬜⬜⬜⬜"
    filled = min(5, max(0, round(p / 20)))
    return "🟩" * filled + "⬜" * (5 - filled)


def format_message(signal: dict[str, Any], market_context: dict[str, Any], news_status: NewsStatus) -> str:
    now_vn = datetime.now(timezone.utc).astimezone(VN_TZ)
    verdict = signal.get("final_verdict", "NO TRADE")
    emoji = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪️"}.get(verdict, "⚪️")
    conf = signal.get("confidence_percent")
    atr = market_context.get("atr_h1")
    h4_bias = market_context.get("timeframes", {}).get("H4", {}).get("bias", "N/A")
    h1_bias = market_context.get("timeframes", {}).get("H1", {}).get("bias", "N/A")

    lines = [
        f"{emoji} <b>XAUUSD — {verdict}</b> | {now_vn.strftime('%H:%M %d/%m')} (VN)",
        f"📊 Tin cậy: {confidence_bar(conf)} {conf if conf is not None else 'N/A'}%",
        f"🧭 Bias: H4 {h4_bias} | H1 {h1_bias}" + (f" | ATR H1 ≈ {atr}" if atr else ""),
        f"📈 Xu hướng AI: {signal.get('xu_huong', 'N/A')}",
    ]

    if signal.get("vung_gia_quan_trong"):
        lines.append(f"🎯 Vùng giá: {signal.get('vung_gia_quan_trong')}")

    if verdict in {"BUY", "SELL"}:
        entry, sl = signal.get("entry"), signal.get("stop_loss")
        tp1, tp2 = signal.get("take_profit_1"), signal.get("take_profit_2")
        sl_dist, tp_dist, rr = calc_risk_reward(entry, sl, tp1)
        lines.extend(
            [
                "",
                f"💰 Entry: <code>{_fmt_price(entry)}</code>",
                f"🛑 SL: <code>{_fmt_price(sl)}</code>" + (f" (~{sl_dist:.2f} pt)" if sl_dist else ""),
                f"✅ TP1: <code>{_fmt_price(tp1)}</code>" + (f" (~{tp_dist:.2f} pt)" if tp_dist else ""),
            ]
        )
        if tp2 is not None:
            lines.append(f"✅ TP2: <code>{_fmt_price(tp2)}</code>")
        if rr:
            lines.append(f"⚖️ R:R: 1 : {rr}")
        lines.append(f"⏳ Giữ lệnh: {signal.get('thoi_gian_giu_lenh', 'N/A')}")
        if signal.get("tin_hieu_vao_lenh"):
            lines.append(f"🕯 Xác nhận: {signal.get('tin_hieu_vao_lenh')}")
        if signal.get("dieu_kien_vo_hieu"):
            lines.append(f"🚫 Vô hiệu nếu: {signal.get('dieu_kien_vo_hieu')}")
    else:
        if signal.get("tin_hieu_vao_lenh"):
            lines.append(f"🕯 Xác nhận: {signal.get('tin_hieu_vao_lenh')}")

    lines.append(f"📰 News: {news_status.text}")

    if signal.get("ghi_chu"):
        lines.extend(["", f"📝 {signal.get('ghi_chu')}"])

    lines.extend(["", "⚠️ <i>AI output, không phải khuyến nghị đầu tư. Luôn tự quản lý rủi ro.</i>"])
    return "\n".join(lines)


# ---------- Logging ----------
def write_signal_log(signal: dict[str, Any], market_context: dict[str, Any], news_status: NewsStatus) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    record = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_vn": now_utc.astimezone(VN_TZ).isoformat(),
        "symbol": SYMBOL,
        "final_verdict": signal.get("final_verdict"),
        "confidence_percent": signal.get("confidence_percent"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit_1": signal.get("take_profit_1"),
        "take_profit_2": signal.get("take_profit_2"),
        "risk_reward_tp1": calc_risk_reward(signal.get("entry"), signal.get("stop_loss"), signal.get("take_profit_1"))[2],
        "atr_h1": market_context.get("atr_h1"),
        "h4_bias": market_context.get("timeframes", {}).get("H4", {}).get("bias"),
        "h1_bias": market_context.get("timeframes", {}).get("H1", {}).get("bias"),
        "news_veto": news_status.veto,
        "news_text": news_status.text,
        "xu_huong": signal.get("xu_huong"),
        "vung_gia_quan_trong": signal.get("vung_gia_quan_trong"),
        "tin_hieu_vao_lenh": signal.get("tin_hieu_vao_lenh"),
        "ghi_chu": signal.get("ghi_chu"),
    }

    jsonl_path = LOG_DIR / "signals.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = LOG_DIR / "signals.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ---------- Main ----------
def main() -> None:
    images: dict[str, bytes] = {}
    dfs: dict[str, pd.DataFrame] = {}
    items = list(TIMEFRAMES.items())

    for i, (tv_interval, (label, size)) in enumerate(items):
        try:
            df = add_indicators(fetch_ohlc(tv_interval, size))
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
    market_context = build_market_context(dfs, atr_h1)
    now_utc = datetime.now(timezone.utc)
    news_status = build_news_status(fetch_economic_calendar(), now_utc)

    try:
        raw_signal = call_gemini(images, market_context, news_status)
        signal = validate_signal(raw_signal, market_context, news_status)
    except Exception as e:
        print(f"LỖI Gemini/validator: {e}", file=sys.stderr)
        send_telegram(f"⚠️ Bot lỗi khi gọi Gemini hoặc kiểm duyệt tín hiệu: {e}")
        sys.exit(1)

    write_signal_log(signal, market_context, news_status)

    if signal.get("final_verdict") == "NO TRADE" and not SEND_NO_TRADE:
        print("NO TRADE, SEND_NO_TRADE=false nên không gửi Telegram.")
        return

    message = format_message(signal, market_context, news_status)

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
