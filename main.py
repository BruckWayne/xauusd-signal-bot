"""
XAUUSD Regime-Adaptive Vision Signal Bot
-----------------------------------------
Phương pháp cố định:
REGIME -> TREND -> LOCATION -> PULLBACK -> LIQUIDITY -> CONFIRMATION -> RISK

Chức năng:
1. Lấy chart D1/H4/H1/M15/M5 từ Twelve Data.
2. Tính ATR, EMA, swing high/low và dữ liệu định lượng.
3. Lọc tin USD High Impact.
4. Gửi ảnh + context cho Gemini Vision.
5. Gemini trả JSON có market regime, setup model, điểm chất lượng và kế hoạch lệnh.
6. Validator bằng code chặn tín hiệu sai cấu trúc, RR thấp, entry xa, tin mạnh,
   thiếu xác nhận hoặc không đồng thuận H4/H1.
7. Gửi kết quả lên Telegram và ghi log JSONL/CSV.

Biến môi trường bắt buộc:
- TWELVEDATA_API_KEY
- GEMINI_API_KEY
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID

Biến môi trường tùy chọn:
- SYMBOL=XAU/USD
- GEMINI_MODEL=gemini-2.5-flash
- MIN_SETUP_SCORE=75
- MIN_RR=2.0
- MAX_ENTRY_DISTANCE_ATR=0.50
- MAX_SL_ATR_MULTIPLIER=1.60
- MIN_SL_ATR_MULTIPLIER=0.10
- NEWS_LOOKAHEAD_HOURS=6
- HIGH_IMPACT_VETO_BEFORE_MINUTES=30
- HIGH_IMPACT_VETO_AFTER_MINUTES=20
- SEND_NO_TRADE=true
- SEND_WAIT=true
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


# ============================================================
# CẤU HÌNH
# ============================================================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = os.getenv("SYMBOL", "XAU/USD")

# Thứ tự lớn -> nhỏ là bắt buộc để Gemini nhận đúng hierarchy.
TIMEFRAMES = {
    "1day": ("D1", 220),
    "4h": ("H4", 220),
    "1h": ("H1", 220),
    "15min": ("M15", 220),
    "5min": ("M5", 220),
}

TWELVEDATA_REQUEST_DELAY_SEC = float(
    os.getenv("TWELVEDATA_REQUEST_DELAY_SEC", "1.0")
)

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_LOOKAHEAD_HOURS = int(os.getenv("NEWS_LOOKAHEAD_HOURS", "6"))
HIGH_IMPACT_VETO_BEFORE_MINUTES = int(
    os.getenv("HIGH_IMPACT_VETO_BEFORE_MINUTES", "30")
)
HIGH_IMPACT_VETO_AFTER_MINUTES = int(
    os.getenv("HIGH_IMPACT_VETO_AFTER_MINUTES", "20")
)
NEWS_RELEVANT_CURRENCIES = {"USD"}
CRITICAL_NEWS_KEYWORDS = (
    "CPI",
    "PPI",
    "NFP",
    "NON-FARM",
    "FOMC",
    "FEDERAL FUNDS",
    "FED CHAIR",
    "POWELL",
    "CORE PCE",
    "UNEMPLOYMENT",
    "GDP",
    "RETAIL SALES",
    "ISM",
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

MIN_SETUP_SCORE = float(os.getenv("MIN_SETUP_SCORE", "75"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_ENTRY_DISTANCE_ATR = float(os.getenv("MAX_ENTRY_DISTANCE_ATR", "0.50"))
MAX_SL_ATR_MULTIPLIER = float(os.getenv("MAX_SL_ATR_MULTIPLIER", "1.60"))
MIN_SL_ATR_MULTIPLIER = float(os.getenv("MIN_SL_ATR_MULTIPLIER", "0.10"))

SEND_NO_TRADE = os.getenv("SEND_NO_TRADE", "true").lower() in {
    "1", "true", "yes", "y"
}
SEND_WAIT = os.getenv("SEND_WAIT", "true").lower() in {
    "1", "true", "yes", "y"
}
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
VN_TZ = timezone(timedelta(hours=7))


@dataclass
class NewsStatus:
    text: str
    veto: bool
    event: dict[str, Any] | None
    calendar_available: bool


# ============================================================
# DỮ LIỆU VÀ CHỈ BÁO
# ============================================================

def fetch_ohlc(interval: str, outputsize: int) -> pd.DataFrame:
    response = requests.get(
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
    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        raise RuntimeError(f"Twelve Data lỗi cho {interval}: {payload}")

    df = pd.DataFrame(payload["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.set_index("datetime", inplace=True)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if df.empty:
        raise RuntimeError(f"Không có OHLC hợp lệ cho {interval}")

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()

    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr14"] = true_range.rolling(14).mean()
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    return out


def calc_atr(df: pd.DataFrame | None, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None

    series = df.get(f"atr{period}")
    if series is None:
        return None

    value = series.iloc[-1]
    return round(float(value), 2) if pd.notna(value) else None


def detect_swings(
    df: pd.DataFrame,
    lookback: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []

    if len(df) < lookback * 2 + 3:
        return highs, lows

    for i in range(lookback, len(df) - lookback):
        window = df.iloc[i - lookback : i + lookback + 1]
        row = df.iloc[i]
        candle_time = df.index[i]

        if row["high"] == window["high"].max():
            highs.append({
                "time": str(candle_time),
                "price": round(float(row["high"]), 2),
            })

        if row["low"] == window["low"].min():
            lows.append({
                "time": str(candle_time),
                "price": round(float(row["low"]), 2),
            })

    return highs[-6:], lows[-6:]


def infer_market_bias(df: pd.DataFrame) -> str:
    if len(df) < 55:
        return "unknown"

    last = df.iloc[-1]
    close = last["close"]
    ema20 = last.get("ema20")
    ema50 = last.get("ema50")

    if pd.isna(ema20) or pd.isna(ema50):
        return "unknown"

    if close > ema20 > ema50:
        return "bullish"
    if close < ema20 < ema50:
        return "bearish"
    return "sideway/mixed"


def infer_regime_quant(df: pd.DataFrame) -> str:
    """Bộ lọc định lượng đơn giản; Gemini vẫn phải xác nhận bằng chart."""
    if len(df) < 55:
        return "unknown"

    recent = df.tail(20)
    last = df.iloc[-1]
    atr = last.get("atr14")
    ema20 = last.get("ema20")
    ema50 = last.get("ema50")

    if pd.isna(atr) or atr <= 0 or pd.isna(ema20) or pd.isna(ema50):
        return "unknown"

    ema_gap_atr = abs(float(ema20 - ema50)) / float(atr)
    directional_move = abs(float(recent["close"].iloc[-1] - recent["close"].iloc[0]))
    path = float(recent["close"].diff().abs().sum())
    efficiency = directional_move / path if path > 0 else 0.0

    short_atr = recent["range"].tail(5).mean()
    long_atr = recent["range"].mean()
    compression_ratio = (
        float(short_atr / long_atr)
        if pd.notna(long_atr) and long_atr > 0
        else 1.0
    )

    if compression_ratio < 0.65:
        return "compression"
    if ema_gap_atr >= 0.75 and efficiency >= 0.35:
        return "trending"
    if efficiency < 0.20:
        return "ranging"
    return "transition/mixed"


def build_market_context(
    dfs: dict[str, pd.DataFrame],
    atr_h1: float | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "symbol": SYMBOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "atr_h1": atr_h1,
        "method": (
            "REGIME -> TREND -> LOCATION -> PULLBACK -> "
            "LIQUIDITY -> CONFIRMATION -> RISK"
        ),
        "timeframes": {},
    }

    for label, df in dfs.items():
        highs, lows = detect_swings(df)
        last = df.iloc[-1]

        context["timeframes"][label] = {
            "last_candle_time": str(df.index[-1]),
            "last_close": round(float(last["close"]), 2),
            "ema20": (
                round(float(last["ema20"]), 2)
                if pd.notna(last.get("ema20"))
                else None
            ),
            "ema50": (
                round(float(last["ema50"]), 2)
                if pd.notna(last.get("ema50"))
                else None
            ),
            "ema200": (
                round(float(last["ema200"]), 2)
                if pd.notna(last.get("ema200"))
                else None
            ),
            "atr14": (
                round(float(last["atr14"]), 2)
                if pd.notna(last.get("atr14"))
                else None
            ),
            "ema_bias": infer_market_bias(df),
            "quant_regime": infer_regime_quant(df),
            "recent_swing_highs": highs,
            "recent_swing_lows": lows,
        }

    return context


def render_chart_png(df: pd.DataFrame, label: str) -> bytes:
    buffer = io.BytesIO()
    overlays = []

    for col in ("ema20", "ema50", "ema200"):
        if col in df and df[col].notna().sum() > 5:
            overlays.append(mpf.make_addplot(df[col], width=0.8))

    last_close = float(df["close"].iloc[-1])

    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"XAUUSD - {label} | Last {last_close:.2f}",
        volume=False,
        addplot=overlays if overlays else None,
        hlines=dict(hlines=[last_close], linewidths=0.7),
        savefig=dict(fname=buffer, dpi=135, bbox_inches="tight"),
        warn_too_much_data=10000,
    )

    buffer.seek(0)
    return buffer.read()


# ============================================================
# LỊCH KINH TẾ
# ============================================================

def fetch_economic_calendar() -> list[dict[str, Any]] | None:
    try:
        response = requests.get(FF_CALENDAR_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(
            f"CẢNH BÁO: không lấy được lịch kinh tế: {exc}",
            file=sys.stderr,
        )
        return None


def _is_critical_news(title: str) -> bool:
    upper = title.upper()
    return any(keyword in upper for keyword in CRITICAL_NEWS_KEYWORDS)


def get_nearest_high_impact_event(
    raw_events: list[dict[str, Any]] | None,
    now_utc: datetime,
) -> dict[str, Any] | None:
    if not raw_events:
        return None

    candidates: list[dict[str, Any]] = []

    for event in raw_events:
        if (
            event.get("country") not in NEWS_RELEVANT_CURRENCIES
            or event.get("impact") != "High"
        ):
            continue

        try:
            event_time = datetime.fromisoformat(event["date"]).astimezone(
                timezone.utc
            )
        except Exception:
            continue

        delta_min = (event_time - now_utc).total_seconds() / 60

        if (
            -HIGH_IMPACT_VETO_AFTER_MINUTES
            <= delta_min
            <= NEWS_LOOKAHEAD_HOURS * 60
        ):
            title = event.get("title", "N/A")
            candidates.append({
                "title": title,
                "minutes_until": round(delta_min),
                "is_critical": _is_critical_news(title),
                "event_time_utc": event_time.isoformat(),
            })

    return (
        min(candidates, key=lambda item: abs(item["minutes_until"]))
        if candidates
        else None
    )


def build_news_status(
    raw_events: list[dict[str, Any]] | None,
    now_utc: datetime,
) -> NewsStatus:
    if raw_events is None:
        return NewsStatus(
            text=(
                "Không lấy được lịch kinh tế; rủi ro tin tức chưa được "
                "xác minh."
            ),
            veto=False,
            event=None,
            calendar_available=False,
        )

    event = get_nearest_high_impact_event(raw_events, now_utc)

    if event is None:
        return NewsStatus(
            text=(
                f"Không có tin USD High Impact trong "
                f"{NEWS_LOOKAHEAD_HOURS} giờ tới."
            ),
            veto=False,
            event=None,
            calendar_available=True,
        )

    minutes = event["minutes_until"]
    before_veto = 0 <= minutes <= HIGH_IMPACT_VETO_BEFORE_MINUTES
    after_veto = -HIGH_IMPACT_VETO_AFTER_MINUTES <= minutes < 0
    critical_extended = (
        event.get("is_critical")
        and 0 <= minutes <= max(HIGH_IMPACT_VETO_BEFORE_MINUTES, 60)
    )
    veto = bool(before_veto or after_veto or critical_extended)

    timing = (
        f"còn {minutes} phút"
        if minutes >= 0
        else f"vừa qua {abs(minutes)} phút"
    )

    text = (
        f"{event['title']} (USD, High Impact) — {timing}."
    )
    if veto:
        text += " Thuộc vùng cấm giao dịch."

    return NewsStatus(
        text=text,
        veto=veto,
        event=event,
        calendar_available=True,
    )


# ============================================================
# PROMPT VÀ JSON SCHEMA
# ============================================================

SYSTEM_PROMPT = """
Bạn là Risk Manager và Market Analyst chuyên phân tích XAUUSD intraday.

Bạn KHÔNG có nhiệm vụ dự đoán chắc chắn hay cố tạo tín hiệu.
Bạn phải bảo toàn vốn, loại bỏ setup kém và duy trì đúng một phương pháp:

REGIME -> TREND -> LOCATION -> PULLBACK -> LIQUIDITY -> CONFIRMATION -> RISK

Bạn nhận {so_luong_anh} ảnh theo thứ tự khung lớn đến nhỏ:
{danh_sach_khung}

Dữ liệu định lượng do hệ thống tính:
{market_context_json}

Trạng thái tin tức:
{news_context}

NGUYÊN TẮC DỮ LIỆU:
- Chỉ phân tích những gì nhìn thấy hoặc có trong dữ liệu định lượng.
- Không bịa DXY, US10Y, volume, tin tức, giá hoặc mô hình.
- Confidence là điểm chất lượng setup, KHÔNG phải tỷ lệ thắng.
- Nếu khung thiếu, ảnh mờ, dữ liệu xung đột hoặc nến xác nhận chưa đóng,
  phải giảm điểm hoặc chọn WAIT/NO TRADE.
- D1/H4 quyết định bối cảnh; H1 quyết định setup; M15/M5 chỉ kích hoạt.
- Một tín hiệu M5 không được đảo ngược cấu trúc H4/H1.
- Không gọi mọi phá đỉnh/đáy nhỏ là BOS.
- Không dùng một pin bar, engulfing, EMA, FVG hay Order Block đơn lẻ để vào lệnh.

BƯỚC 1 — DATA QUALITY
Kiểm tra khung đã nhận, đồng bộ giá, độ rõ ảnh và tính hợp lệ.
Nếu dữ liệu xung đột nghiêm trọng: NO TRADE.

BƯỚC 2 — MARKET REGIME
Chọn đúng một:
TRENDING, RANGING, BREAKOUT_EXPANSION, COMPRESSION,
TRANSITION, DISORDERED_HIGH_RISK.

- TRENDING: cấu trúc và displacement có tính tiếp diễn.
- RANGING: giá ở hai biên rõ, breakout thất bại, nến chồng lấn.
- BREAKOUT_EXPANSION: đóng ngoài range và có follow-through.
- COMPRESSION: biên độ thu hẹp; phải chờ breakout xác nhận.
- TRANSITION: xu hướng cũ yếu nhưng xu hướng mới chưa hoàn chỉnh.
- DISORDERED_HIGH_RISK: nến hai chiều lớn, wick dài, cấu trúc nhiễu.

Nếu RANGING ở giữa biên, TRANSITION hoặc DISORDERED_HIGH_RISK:
ưu tiên NO TRADE.

BƯỚC 3 — TREND HIERARCHY
Xác định bias D1, H4, H1, M15, M5.
Thứ tự ưu tiên D1 > H4 > H1 > M15 > M5.
BUY/SELL chỉ hợp lệ khi H4/H1 đồng thuận hoặc H1 là pullback có kiểm soát
trong cấu trúc H4.

BƯỚC 4 — LOCATION
Đánh giá giá đang ở đâu:
- swing high/low;
- hỗ trợ/kháng cự;
- vùng breakout/retest;
- supply/demand;
- premium/equilibrium/discount;
- thanh khoản phía trên/dưới.

Không BUY sát kháng cự mạnh.
Không SELL sát hỗ trợ mạnh.
Không chase sau nến expansion lớn.

BƯỚC 5 — PULLBACK
Pullback tốt:
- chậm hơn nhịp xu hướng;
- nến chồng lấn hoặc thu hẹp;
- không phá protected swing;
- hồi về vùng giá trị.

Pullback xấu:
- displacement mạnh ngược bias;
- phá protected swing;
- H1 tạo cấu trúc đảo chiều hoàn chỉnh.

BƯỚC 6 — LIQUIDITY
Phân biệt chính xác:
- sweep;
- breakout;
- fakeout;
- buy-side/sell-side liquidity;
- equal highs/lows.

Không gán nhãn “thao túng” nếu chỉ quan sát được một râu nến.

BƯỚC 7 — ENTRY MODEL
Chỉ dùng một trong ba model:

MODEL_A_TREND_PULLBACK:
H4/H1 thuận xu hướng, pullback về vùng giá trị, protected swing còn giữ,
sau đó M15/M5 xác nhận tiếp diễn.

MODEL_B_BREAKOUT_RETEST:
range/compression rõ, nến đóng ngoài biên bằng displacement,
có follow-through và retest giữ được.

MODEL_C_CONFIRMED_REVERSAL:
sweep tại vùng D1/H4, xu hướng cũ suy yếu, H1 structure shift rõ,
retest thất bại theo hướng cũ và M15/M5 xác nhận hướng mới.
Model C có ưu tiên thấp hơn Model A.

BƯỚC 8 — TRIGGER
Cần ít nhất hai xác nhận:
- M15/M5 structure shift thuận bias;
- displacement đóng qua cấu trúc nhỏ;
- retest giữ được;
- rejection rõ đúng vùng;
- breakout-retest hoàn chỉnh.

Nếu bối cảnh tốt nhưng trigger chưa hoàn tất: WAIT.
WAIT không phải lệnh chờ tự động.

BƯỚC 9 — VOLATILITY VÀ RISK
- SL nằm sau điểm vô hiệu cấu trúc, có buffer biến động.
- TP tại cấu trúc/thanh khoản kế tiếp.
- Không thu SL chỉ để làm đẹp RR.
- Không đặt TP vượt phạm vi thực tế.
- RR tới TP1 phải tối thiểu 1:2 cho lệnh thông thường.

BƯỚC 10 — SCORING
Chấm đúng trọng số:
- regime_score: 0-15
- trend_alignment_score: 0-20
- location_score: 0-15
- pullback_score: 0-15
- liquidity_score: 0-10
- confirmation_score: 0-10
- rr_score: 0-10
- execution_score: 0-5

Tổng setup_score phải bằng tổng tám điểm trên và nằm trong 0-100.

Xếp hạng:
- A: 85-100
- B: 75-84
- WATCHLIST: 65-74
- REJECTED: dưới 65

QUY TẮC VERDICT:
- BUY/SELL: setup_score >= 75, trigger hoàn tất, RR hợp lệ,
  không vi phạm điều kiện loại trừ.
- WAIT: context tốt nhưng trigger chưa hoàn tất hoặc nến chưa đóng.
- NO TRADE: không có edge, dữ liệu xung đột, regime không phù hợp,
  RR thấp, entry muộn, tin mạnh hoặc rủi ro cao.

ĐIỀU KIỆN NO TRADE BẮT BUỘC:
- tin tức đang thuộc vùng cấm;
- giá ở giữa range;
- H4/H1 xung đột mạnh;
- không xác định được invalidation;
- entry quá xa setup;
- RR dưới 1:2;
- M5 là bằng chứng duy nhất;
- nến trigger chưa đóng nhưng AI lại đề xuất MARKET;
- dữ liệu thiếu nghiêm trọng hoặc không đồng bộ.

KHI BUY/SELL:
- entry, stop_loss, take_profit_1 phải là số.
- BUY: stop_loss < entry < take_profit_1.
- SELL: take_profit_1 < entry < stop_loss.
- take_profit_2 chỉ cung cấp nếu có target hợp lý.
- Nêu rõ điều kiện kích hoạt và invalidation.

KHI WAIT/NO TRADE:
- entry, stop_loss, take_profit_1, take_profit_2 phải là null.
- Nêu rõ cần chờ điều gì hoặc vì sao loại bỏ setup.

Trả về đúng JSON schema, không markdown, không thêm chữ ngoài JSON.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "final_verdict": {
            "type": "STRING",
            "enum": ["BUY", "SELL", "WAIT", "NO TRADE"],
        },
        "market_regime": {"type": "STRING"},
        "setup_model": {
            "type": "STRING",
            "enum": [
                "MODEL_A_TREND_PULLBACK",
                "MODEL_B_BREAKOUT_RETEST",
                "MODEL_C_CONFIRMED_REVERSAL",
                "NONE",
            ],
        },
        "setup_grade": {
            "type": "STRING",
            "enum": ["A", "B", "WATCHLIST", "REJECTED"],
        },
        "setup_score": {"type": "NUMBER"},
        "regime_score": {"type": "NUMBER"},
        "trend_alignment_score": {"type": "NUMBER"},
        "location_score": {"type": "NUMBER"},
        "pullback_score": {"type": "NUMBER"},
        "liquidity_score": {"type": "NUMBER"},
        "confirmation_score": {"type": "NUMBER"},
        "rr_score": {"type": "NUMBER"},
        "execution_score": {"type": "NUMBER"},
        "data_quality": {"type": "STRING"},
        "xu_huong": {"type": "STRING"},
        "bias_chinh": {"type": "STRING"},
        "vung_gia_quan_trong": {"type": "STRING"},
        "thanh_khoan": {"type": "STRING"},
        "pullback_danh_gia": {"type": "STRING"},
        "tin_hieu_vao_lenh": {"type": "STRING"},
        "entry_type": {"type": "STRING"},
        "entry": {"type": "NUMBER", "nullable": True},
        "stop_loss": {"type": "NUMBER", "nullable": True},
        "take_profit_1": {"type": "NUMBER", "nullable": True},
        "take_profit_2": {"type": "NUMBER", "nullable": True},
        "ly_do_tp": {"type": "STRING"},
        "thoi_gian_giu_lenh": {"type": "STRING"},
        "dieu_kien_vo_hieu": {"type": "STRING"},
        "dieu_kien_cho": {"type": "STRING"},
        "rui_ro_lon_nhat": {"type": "STRING"},
        "ghi_chu": {"type": "STRING"},
    },
    "required": [
        "final_verdict",
        "market_regime",
        "setup_model",
        "setup_grade",
        "setup_score",
        "regime_score",
        "trend_alignment_score",
        "location_score",
        "pullback_score",
        "liquidity_score",
        "confirmation_score",
        "rr_score",
        "execution_score",
        "data_quality",
        "xu_huong",
        "bias_chinh",
        "vung_gia_quan_trong",
        "thanh_khoan",
        "pullback_danh_gia",
        "tin_hieu_vao_lenh",
        "entry_type",
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "ly_do_tp",
        "thoi_gian_giu_lenh",
        "dieu_kien_vo_hieu",
        "dieu_kien_cho",
        "rui_ro_lon_nhat",
        "ghi_chu",
    ],
}


def build_prompt(
    image_labels: list[str],
    market_context: dict[str, Any],
    news_status: NewsStatus,
) -> str:
    context_json = json.dumps(
        market_context,
        ensure_ascii=False,
        indent=2,
    )

    return SYSTEM_PROMPT.format(
        so_luong_anh=len(image_labels),
        danh_sach_khung=", ".join(image_labels),
        market_context_json=context_json,
        news_context=news_status.text,
    )


def call_gemini(
    images: dict[str, bytes],
    market_context: dict[str, Any],
    news_status: NewsStatus,
) -> dict[str, Any]:
    prompt = build_prompt(
        list(images.keys()),
        market_context,
        news_status,
    )

    parts: list[dict[str, Any]] = [{"text": prompt}]

    for label, png_bytes in images.items():
        parts.append({"text": f"Ảnh tiếp theo là chart {label}."})
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("utf-8"),
            }
        })

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.10,
            "topP": 0.80,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    response = requests.post(
        GEMINI_URL,
        json=body,
        timeout=150,
    )
    response.raise_for_status()

    payload = response.json()
    candidates = payload.get("candidates") or []

    if not candidates:
        raise RuntimeError(f"Gemini không trả candidate: {payload}")

    text = (
        candidates[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

    if not text:
        raise RuntimeError(f"Gemini trả nội dung rỗng: {payload}")

    if text.startswith("```"):
        text = (
            text.strip("`")
            .replace("json\n", "", 1)
            .replace("json", "", 1)
            .strip()
        )

    return json.loads(text)


# ============================================================
# VALIDATOR
# ============================================================

SCORE_FIELDS = (
    "regime_score",
    "trend_alignment_score",
    "location_score",
    "pullback_score",
    "liquidity_score",
    "confirmation_score",
    "rr_score",
    "execution_score",
)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _clamp(value: float | None, minimum: float, maximum: float) -> float:
    if value is None:
        return minimum
    return min(maximum, max(minimum, value))


def calc_risk_reward(
    entry: Any,
    stop_loss: Any,
    take_profit: Any,
) -> tuple[float | None, float | None, float | None]:
    entry_f = _as_float(entry)
    sl_f = _as_float(stop_loss)
    tp_f = _as_float(take_profit)

    if not all(v is not None for v in (entry_f, sl_f, tp_f)):
        return None, None, None

    sl_distance = abs(entry_f - sl_f)
    tp_distance = abs(entry_f - tp_f)
    rr = tp_distance / sl_distance if sl_distance > 0 else None

    return (
        round(sl_distance, 2),
        round(tp_distance, 2),
        round(rr, 2) if rr is not None else None,
    )


def normalize_signal(signal: dict[str, Any]) -> dict[str, Any]:
    out = dict(signal)

    verdict = str(out.get("final_verdict", "NO TRADE")).upper().strip()
    aliases = {
        "HOLD": "WAIT",
        "WATCH": "WAIT",
        "NONE": "NO TRADE",
        "NO_TRADE": "NO TRADE",
    }
    verdict = aliases.get(verdict, verdict)

    if verdict not in {"BUY", "SELL", "WAIT", "NO TRADE"}:
        verdict = "NO TRADE"

    out["final_verdict"] = verdict

    for key in (
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "setup_score",
        *SCORE_FIELDS,
    ):
        out[key] = _as_float(out.get(key))

    max_scores = {
        "regime_score": 15,
        "trend_alignment_score": 20,
        "location_score": 15,
        "pullback_score": 15,
        "liquidity_score": 10,
        "confirmation_score": 10,
        "rr_score": 10,
        "execution_score": 5,
    }

    for key, maximum in max_scores.items():
        out[key] = _clamp(out.get(key), 0, maximum)

    calculated_score = round(sum(out[key] for key in SCORE_FIELDS), 2)
    out["setup_score"] = calculated_score

    if calculated_score >= 85:
        out["setup_grade"] = "A"
    elif calculated_score >= 75:
        out["setup_grade"] = "B"
    elif calculated_score >= 65:
        out["setup_grade"] = "WATCHLIST"
    else:
        out["setup_grade"] = "REJECTED"

    if verdict in {"WAIT", "NO TRADE"}:
        for key in (
            "entry",
            "stop_loss",
            "take_profit_1",
            "take_profit_2",
        ):
            out[key] = None

    return out


def force_verdict(
    signal: dict[str, Any],
    verdict: str,
    reasons: list[str],
) -> dict[str, Any]:
    out = dict(signal)
    out["final_verdict"] = verdict

    if verdict in {"WAIT", "NO TRADE"}:
        for key in (
            "entry",
            "stop_loss",
            "take_profit_1",
            "take_profit_2",
        ):
            out[key] = None

    existing = str(out.get("ghi_chu") or "").strip()
    validator_note = "Validator: " + "; ".join(reasons)
    out["ghi_chu"] = (
        f"{existing} | {validator_note}"
        if existing
        else validator_note
    )
    return out


def validate_signal(
    signal: dict[str, Any],
    market_context: dict[str, Any],
    news_status: NewsStatus,
) -> dict[str, Any]:
    signal = normalize_signal(signal)
    verdict = signal["final_verdict"]
    score = _as_float(signal.get("setup_score")) or 0.0

    contexts = market_context.get("timeframes", {})
    d1_bias = contexts.get("D1", {}).get("ema_bias")
    h4_bias = contexts.get("H4", {}).get("ema_bias")
    h1_bias = contexts.get("H1", {}).get("ema_bias")
    h1_close = _as_float(contexts.get("H1", {}).get("last_close"))
    atr_h1 = _as_float(market_context.get("atr_h1"))

    hard_reasons: list[str] = []
    wait_reasons: list[str] = []

    if news_status.veto:
        hard_reasons.append("tin USD High Impact thuộc vùng cấm")

    if not news_status.calendar_available:
        hard_reasons.append("không xác minh được lịch kinh tế")

    regime = str(signal.get("market_regime") or "").upper()
    if regime == "DISORDERED_HIGH_RISK":
        hard_reasons.append("market regime hỗn loạn/rủi ro cao")

    if verdict in {"BUY", "SELL"}:
        entry = _as_float(signal.get("entry"))
        sl = _as_float(signal.get("stop_loss"))
        tp1 = _as_float(signal.get("take_profit_1"))
        confirmation_score = (
            _as_float(signal.get("confirmation_score")) or 0.0
        )

        if score < MIN_SETUP_SCORE:
            wait_reasons.append(
                f"setup score {score:.0f} thấp hơn {MIN_SETUP_SCORE:.0f}"
            )

        if confirmation_score < 7:
            wait_reasons.append("xác nhận entry chưa đủ mạnh")

        if not all(v is not None for v in (entry, sl, tp1)):
            hard_reasons.append("thiếu entry/SL/TP1")
        else:
            if verdict == "BUY":
                if not (sl < entry < tp1):
                    hard_reasons.append(
                        "BUY sai cấu trúc: cần SL < entry < TP1"
                    )
                if h4_bias == "bearish" and h1_bias == "bearish":
                    hard_reasons.append(
                        "BUY ngược H4/H1 đều bearish"
                    )

            if verdict == "SELL":
                if not (tp1 < entry < sl):
                    hard_reasons.append(
                        "SELL sai cấu trúc: cần TP1 < entry < SL"
                    )
                if h4_bias == "bullish" and h1_bias == "bullish":
                    hard_reasons.append(
                        "SELL ngược H4/H1 đều bullish"
                    )

            sl_distance, _, rr = calc_risk_reward(entry, sl, tp1)

            if rr is None or rr < MIN_RR:
                hard_reasons.append(
                    f"R:R thấp hơn 1:{MIN_RR}"
                )

            if atr_h1 and sl_distance:
                if sl_distance > atr_h1 * MAX_SL_ATR_MULTIPLIER:
                    hard_reasons.append("SL quá rộng so với ATR H1")
                if sl_distance < atr_h1 * MIN_SL_ATR_MULTIPLIER:
                    hard_reasons.append("SL quá sát so với ATR H1")

            if atr_h1 and h1_close and entry:
                entry_distance = abs(entry - h1_close)
                if entry_distance > atr_h1 * MAX_ENTRY_DISTANCE_ATR:
                    hard_reasons.append(
                        "entry quá xa giá hiện tại, có nguy cơ chase"
                    )

        # D1 chỉ là lớp giảm độ tin cậy, không phủ quyết máy móc.
        if (
            verdict == "BUY"
            and d1_bias == "bearish"
            and h4_bias != "bullish"
        ):
            wait_reasons.append("BUY chưa được D1/H4 hỗ trợ")
        if (
            verdict == "SELL"
            and d1_bias == "bullish"
            and h4_bias != "bearish"
        ):
            wait_reasons.append("SELL chưa được D1/H4 hỗ trợ")

        if hard_reasons:
            return force_verdict(
                signal,
                "NO TRADE",
                hard_reasons + wait_reasons,
            )

        if wait_reasons:
            return force_verdict(signal, "WAIT", wait_reasons)

    if verdict == "WAIT" and score < 65:
        return force_verdict(
            signal,
            "NO TRADE",
            ["setup dưới ngưỡng watchlist"],
        )

    if hard_reasons:
        return force_verdict(signal, "NO TRADE", hard_reasons)

    return signal


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text: str) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    response.raise_for_status()


def send_telegram_photo(
    png_bytes: bytes,
    caption: str = "",
) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML",
        },
        files={"photo": ("chart.png", png_bytes, "image/png")},
        timeout=30,
    )
    response.raise_for_status()


def _fmt_price(value: Any) -> str:
    number = _as_float(value)
    return f"{number:,.2f}" if number is not None else "N/A"


def score_bar(score: Any) -> str:
    value = _as_float(score)
    if value is None:
        return "⬜⬜⬜⬜⬜"

    filled = min(5, max(0, round(value / 20)))
    return "🟩" * filled + "⬜" * (5 - filled)


def format_message(
    signal: dict[str, Any],
    market_context: dict[str, Any],
    news_status: NewsStatus,
) -> str:
    now_vn = datetime.now(timezone.utc).astimezone(VN_TZ)
    verdict = signal.get("final_verdict", "NO TRADE")
    emoji = {
        "BUY": "🟢",
        "SELL": "🔴",
        "WAIT": "🟡",
        "NO TRADE": "⚪️",
    }.get(verdict, "⚪️")

    score = signal.get("setup_score")
    grade = signal.get("setup_grade", "N/A")
    contexts = market_context.get("timeframes", {})
    d1_bias = contexts.get("D1", {}).get("ema_bias", "N/A")
    h4_bias = contexts.get("H4", {}).get("ema_bias", "N/A")
    h1_bias = contexts.get("H1", {}).get("ema_bias", "N/A")
    atr = market_context.get("atr_h1")

    lines = [
        (
            f"{emoji} <b>XAUUSD — {verdict}</b> | "
            f"{now_vn.strftime('%H:%M %d/%m')} (VN)"
        ),
        (
            f"📊 Setup: {score_bar(score)} "
            f"{score if score is not None else 'N/A'}/100 | Grade {grade}"
        ),
        (
            f"🌐 Regime: {signal.get('market_regime', 'N/A')} | "
            f"Model: {signal.get('setup_model', 'NONE')}"
        ),
        (
            f"🧭 Bias định lượng: D1 {d1_bias} | "
            f"H4 {h4_bias} | H1 {h1_bias}"
            + (f" | ATR H1 ≈ {atr}" if atr else "")
        ),
        f"📈 Bias AI: {signal.get('bias_chinh', 'N/A')}",
        f"🧱 Cấu trúc: {signal.get('xu_huong', 'N/A')}",
    ]

    if signal.get("vung_gia_quan_trong"):
        lines.append(
            f"🎯 Vùng giá: {signal.get('vung_gia_quan_trong')}"
        )

    if signal.get("thanh_khoan"):
        lines.append(
            f"💧 Thanh khoản: {signal.get('thanh_khoan')}"
        )

    if signal.get("pullback_danh_gia"):
        lines.append(
            f"↩️ Pullback: {signal.get('pullback_danh_gia')}"
        )

    if verdict in {"BUY", "SELL"}:
        entry = signal.get("entry")
        sl = signal.get("stop_loss")
        tp1 = signal.get("take_profit_1")
        tp2 = signal.get("take_profit_2")
        sl_distance, _, rr = calc_risk_reward(entry, sl, tp1)

        lines.extend([
            "",
            f"🎬 Entry type: {signal.get('entry_type', 'N/A')}",
            f"💰 Entry: <code>{_fmt_price(entry)}</code>",
            (
                f"🛑 SL: <code>{_fmt_price(sl)}</code>"
                + (
                    f" (~{sl_distance:.2f} USD)"
                    if sl_distance is not None
                    else ""
                )
            ),
            f"✅ TP1: <code>{_fmt_price(tp1)}</code>",
        ])

        if tp2 is not None:
            lines.append(
                f"✅ TP2: <code>{_fmt_price(tp2)}</code>"
            )

        if rr is not None:
            lines.append(f"⚖️ R:R TP1: 1:{rr}")

        lines.append(
            f"🕯 Trigger: {signal.get('tin_hieu_vao_lenh', 'N/A')}"
        )
        lines.append(
            f"🚫 Vô hiệu: {signal.get('dieu_kien_vo_hieu', 'N/A')}"
        )
        lines.append(
            f"⏳ Giữ lệnh: {signal.get('thoi_gian_giu_lenh', 'N/A')}"
        )

    elif verdict == "WAIT":
        lines.append(
            f"⏳ Cần chờ: {signal.get('dieu_kien_cho', 'N/A')}"
        )
        if signal.get("tin_hieu_vao_lenh"):
            lines.append(
                f"🕯 Trigger cần có: "
                f"{signal.get('tin_hieu_vao_lenh')}"
            )

    else:
        lines.append(
            f"🚫 Lý do loại: {signal.get('ghi_chu', 'Không có edge rõ')}"
        )

    lines.append(f"📰 News: {news_status.text}")
    lines.append(
        f"⚠️ Rủi ro lớn nhất: "
        f"{signal.get('rui_ro_lon_nhat', 'N/A')}"
    )

    if signal.get("ghi_chu") and verdict != "NO TRADE":
        lines.extend(["", f"📝 {signal.get('ghi_chu')}"])

    lines.extend([
        "",
        (
            "⚠️ <i>AI Vision chỉ hỗ trợ phân tích. "
            "Không phải cam kết lợi nhuận hoặc khuyến nghị đầu tư.</i>"
        ),
    ])

    return "\n".join(lines)


# ============================================================
# LOGGING
# ============================================================

def write_signal_log(
    signal: dict[str, Any],
    market_context: dict[str, Any],
    news_status: NewsStatus,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    contexts = market_context.get("timeframes", {})

    record = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_vn": now_utc.astimezone(VN_TZ).isoformat(),
        "symbol": SYMBOL,
        "final_verdict": signal.get("final_verdict"),
        "market_regime": signal.get("market_regime"),
        "setup_model": signal.get("setup_model"),
        "setup_grade": signal.get("setup_grade"),
        "setup_score": signal.get("setup_score"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit_1": signal.get("take_profit_1"),
        "take_profit_2": signal.get("take_profit_2"),
        "risk_reward_tp1": calc_risk_reward(
            signal.get("entry"),
            signal.get("stop_loss"),
            signal.get("take_profit_1"),
        )[2],
        "atr_h1": market_context.get("atr_h1"),
        "d1_bias": contexts.get("D1", {}).get("ema_bias"),
        "h4_bias": contexts.get("H4", {}).get("ema_bias"),
        "h1_bias": contexts.get("H1", {}).get("ema_bias"),
        "news_veto": news_status.veto,
        "news_text": news_status.text,
        "bias_chinh": signal.get("bias_chinh"),
        "xu_huong": signal.get("xu_huong"),
        "vung_gia_quan_trong": signal.get("vung_gia_quan_trong"),
        "thanh_khoan": signal.get("thanh_khoan"),
        "pullback_danh_gia": signal.get("pullback_danh_gia"),
        "tin_hieu_vao_lenh": signal.get("tin_hieu_vao_lenh"),
        "dieu_kien_vo_hieu": signal.get("dieu_kien_vo_hieu"),
        "dieu_kien_cho": signal.get("dieu_kien_cho"),
        "rui_ro_lon_nhat": signal.get("rui_ro_lon_nhat"),
        "ghi_chu": signal.get("ghi_chu"),
    }

    with (LOG_DIR / "signals.jsonl").open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = LOG_DIR / "signals.csv"
    write_header = not csv_path.exists()

    with csv_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(record.keys()),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    images: dict[str, bytes] = {}
    dfs: dict[str, pd.DataFrame] = {}
    items = list(TIMEFRAMES.items())

    for index, (interval, (label, outputsize)) in enumerate(items):
        try:
            df = add_indicators(fetch_ohlc(interval, outputsize))
            dfs[label] = df
            images[label] = render_chart_png(df, label)
            print(f"OK: {label}")
        except Exception as exc:
            print(f"LỖI khung {label}: {exc}", file=sys.stderr)

        if index < len(items) - 1:
            time.sleep(TWELVEDATA_REQUEST_DELAY_SEC)

    required_context_frames = {"H4", "H1", "M15"}
    missing_required = required_context_frames.difference(images)

    if missing_required:
        message = (
            "⚠️ Bot lỗi: thiếu khung bắt buộc "
            + ", ".join(sorted(missing_required))
        )
        send_telegram(message)
        raise RuntimeError(message)

    atr_h1 = calc_atr(dfs.get("H1"))
    market_context = build_market_context(dfs, atr_h1)

    now_utc = datetime.now(timezone.utc)
    news_status = build_news_status(
        fetch_economic_calendar(),
        now_utc,
    )

    try:
        raw_signal = call_gemini(
            images,
            market_context,
            news_status,
        )
        signal = validate_signal(
            raw_signal,
            market_context,
            news_status,
        )
    except Exception as exc:
        print(
            f"LỖI Gemini/validator: {exc}",
            file=sys.stderr,
        )
        send_telegram(
            f"⚠️ Bot lỗi khi gọi Gemini hoặc kiểm duyệt: {exc}"
        )
        raise

    write_signal_log(signal, market_context, news_status)

    verdict = signal.get("final_verdict")

    if verdict == "NO TRADE" and not SEND_NO_TRADE:
        print("NO TRADE và SEND_NO_TRADE=false.")
        return

    if verdict == "WAIT" and not SEND_WAIT:
        print("WAIT và SEND_WAIT=false.")
        return

    message = format_message(
        signal,
        market_context,
        news_status,
    )

    chart_for_photo = (
        images.get("H1")
        or images.get("H4")
        or next(iter(images.values()), None)
    )

    if chart_for_photo:
        try:
            emoji = {
                "BUY": "🟢",
                "SELL": "🔴",
                "WAIT": "🟡",
                "NO TRADE": "⚪️",
            }.get(verdict, "⚪️")

            send_telegram_photo(
                chart_for_photo,
                caption=f"{emoji} XAUUSD H1 — {verdict}",
            )
        except Exception as exc:
            print(
                f"CẢNH BÁO: gửi ảnh thất bại: {exc}",
                file=sys.stderr,
            )

    send_telegram(message)
    print("Đã gửi Telegram thành công.")


if __name__ == "__main__":
    main()
