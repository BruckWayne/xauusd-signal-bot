"""XAUUSD AI Vision scalp analyzer (paper-trading / research use only).

Method: H1 bias -> M15 setup -> M5 confirmation -> M1 trigger.
GitHub Actions may run every 5 minutes, but Gemini is only called when a
quantitative prefilter finds a plausible setup.
"""
from __future__ import annotations

import base64, csv, io, json, os, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mplfinance as mpf
import pandas as pd
import requests

TD_KEY = os.environ["TWELVEDATA_API_KEY"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
SYMBOL = os.getenv("SYMBOL", "XAU/USD")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
M1_OUTPUTSIZE = int(os.getenv("M1_OUTPUTSIZE", "5000"))
SCALP_SESSION_UTC = os.getenv("SCALP_SESSION_UTC", "06:00-21:59")
FORCE_AI = os.getenv("FORCE_AI_CALL", "false").lower() in {"1","true","yes"}
MIN_SCORE = float(os.getenv("MIN_SETUP_SCORE", "72"))
MIN_RR = float(os.getenv("MIN_RR", "1.5"))
MIN_PREFILTER = int(os.getenv("MIN_PREFILTER_SCORE", "4"))
MIN_ATR_PCT = float(os.getenv("MIN_M5_ATR_PCT", "0.025"))
MAX_ENTRY_ATR = float(os.getenv("MAX_ENTRY_DISTANCE_ATR_M5", "0.45"))
MIN_SL_ATR = float(os.getenv("MIN_SL_ATR_M5", "0.35"))
MAX_SL_ATR = float(os.getenv("MAX_SL_ATR_M5", "1.8"))
MAX_AGE = int(os.getenv("MAX_SIGNAL_AGE_MINUTES", "8"))
NEWS_BEFORE = int(os.getenv("HIGH_IMPACT_VETO_BEFORE_MINUTES", "20"))
NEWS_AFTER = int(os.getenv("HIGH_IMPACT_VETO_AFTER_MINUTES", "15"))
SEND_WAIT = os.getenv("SEND_WAIT", "false").lower() in {"1","true","yes"}
SEND_NO_TRADE = os.getenv("SEND_NO_TRADE", "false").lower() in {"1","true","yes"}
SEND_CHART = os.getenv("SEND_CHART", "true").lower() in {"1","true","yes"}
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
VN_TZ = timezone(timedelta(hours=7))
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

@dataclass
class News:
    text: str
    veto: bool
    available: bool


def num(v: Any) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(str(v).replace(",", ""))
    except Exception:
        return None


def in_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    a, b = SCALP_SESSION_UTC.split("-", 1)
    sh, sm = map(int, a.split(":")); eh, em = map(int, b.split(":"))
    cur, start, end = now.hour*60+now.minute, sh*60+sm, eh*60+em
    return start <= cur <= end if start <= end else (cur >= start or cur <= end)


def telegram(text: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=30)
    r.raise_for_status()


def telegram_photo(data: bytes, caption: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
        data={"chat_id":TG_CHAT,"caption":caption},
        files={"photo":("xauusd_m5.png",data,"image/png")}, timeout=30)
    r.raise_for_status()


def fetch_m1() -> pd.DataFrame:
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol":SYMBOL,"interval":"1min","outputsize":M1_OUTPUTSIZE,
        "timezone":"UTC","order":"ASC","format":"JSON","apikey":TD_KEY}, timeout=40)
    r.raise_for_status(); p = r.json()
    if "values" not in p: raise RuntimeError(f"Twelve Data: {p}")
    df = pd.DataFrame(p["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df.set_index("datetime", inplace=True)
    for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    return df[~df.index.duplicated(keep="last")].sort_index()


def resample(df: pd.DataFrame, rule: str, now: datetime) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open":"first","high":"max","low":"min","close":"last"})
    out.dropna(inplace=True)
    return out[(out.index + pd.Timedelta(rule)) <= pd.Timestamp(now)]


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for n in (9,20,50): d[f"ema{n}"] = d.close.ewm(span=n, adjust=False).mean()
    pc = d.close.shift(1)
    tr = pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean(); d["body"]=(d.close-d.open).abs(); d["range"]=d.high-d.low
    return d


def build_frames(m1: pd.DataFrame, now: datetime) -> dict[str,pd.DataFrame]:
    rules={"H4":"4h","H1":"1h","M15":"15min","M5":"5min","M1":"1min"}
    return {k:indicators(resample(m1,v,now)) for k,v in rules.items()}


def bias(df: pd.DataFrame) -> str:
    if len(df)<55: return "unknown"
    x=df.iloc[-1]
    if x.close>x.ema20>x.ema50:return "bullish"
    if x.close<x.ema20<x.ema50:return "bearish"
    return "mixed"


def summary(df: pd.DataFrame) -> dict[str,Any]:
    x=df.iloc[-1]
    return {"closed_time_utc":df.index[-1].isoformat(),"close":round(float(x.close),2),
            "ema9":round(float(x.ema9),2),"ema20":round(float(x.ema20),2),
            "ema50":round(float(x.ema50),2),"atr14":round(float(x.atr14),3) if pd.notna(x.atr14) else None,
            "ema_bias":bias(df),"recent_high":round(float(df.high.tail(20).max()),2),
            "recent_low":round(float(df.low.tail(20).min()),2)}


def prefilter(f: dict[str,pd.DataFrame]) -> tuple[bool,int,list[str],str]:
    score=0; why=[]
    h1,m15,m5,m1=f["H1"],f["M15"],f["M5"],f["M1"]
    b1,b15,b5=bias(h1),bias(m15),bias(m5)
    if b1 in {"bullish","bearish"}: score+=1; why.append(f"H1 {b1}")
    if b1==b15 and b1 in {"bullish","bearish"}: score+=2; why.append("H1/M15 aligned")
    elif b15 in {"bullish","bearish"}: score+=1; why.append(f"M15 {b15}")
    x=m5.iloc[-1]; atr=num(x.atr14); close=num(x.close)
    if atr and close and atr/close*100>=MIN_ATR_PCT: score+=1; why.append("M5 volatility active")
    if atr and abs(float(x.close-x.ema20))<=atr*.8: score+=1; why.append("M5 near pullback zone")
    if atr and float(x.body)>=atr*.45: score+=1; why.append("M5 displacement")
    r=m1.tail(3); up=int((r.close>r.open).sum()); down=int((r.close<r.open).sum())
    if max(up,down)>=2: score+=1; why.append("M1 short momentum")
    candidate="BUY" if b1==b15=="bullish" else "SELL" if b1==b15=="bearish" else "NEUTRAL"
    return FORCE_AI or score>=MIN_PREFILTER,score,why,candidate


def news(now: datetime) -> News:
    try:
        r=requests.get(FF_URL,timeout=15);r.raise_for_status();raw=r.json()
    except Exception as e:
        return News(f"Không xác minh được lịch tin: {e}",True,False)
    events=[]
    for ev in raw:
        if ev.get("country")!="USD" or ev.get("impact")!="High":continue
        try:t=datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except Exception:continue
        mins=(t-now).total_seconds()/60
        if -NEWS_AFTER<=mins<=240:events.append((abs(mins),mins,ev.get("title","USD High Impact")))
    if not events:return News("Không có tin USD High Impact gần.",False,True)
    _,mins,title=min(events)
    veto=(0<=mins<=NEWS_BEFORE) or (-NEWS_AFTER<=mins<0)
    timing=f"còn {round(mins)} phút" if mins>=0 else f"đã qua {abs(round(mins))} phút"
    return News(f"{title} — {timing}"+(" | VETO" if veto else ""),veto,True)


def chart(df: pd.DataFrame,label:str,n:int) -> bytes:
    d=df.tail(n); buf=io.BytesIO(); aps=[mpf.make_addplot(d.ema9),mpf.make_addplot(d.ema20),mpf.make_addplot(d.ema50)]
    mpf.plot(d,type="candle",style="charles",title=f"XAUUSD {label} | CLOSED CANDLES",addplot=aps,
             volume=False,savefig=dict(fname=buf,dpi=125,bbox_inches="tight"),warn_too_much_data=10000)
    buf.seek(0);return buf.read()

PROMPT='''Bạn là Risk Manager phân tích XAUUSD scalp cho mục đích paper-trading.
Bạn nhận H1, M15, M5, M1; tất cả là nến đã đóng.
Phương pháp bắt buộc: H1 bias -> M15 setup -> M5 confirmation -> M1 trigger -> risk.
Dự kiến giữ lệnh 5-60 phút. Không bịa dữ liệu, không ép BUY/SELL.

Context định lượng:\n{context}
Prefilter: score={ps}; candidate={cb}; reasons={pr}
News: {news}

Quy tắc:
- H1 chỉ định bias/cản; M15 định setup/location; M5 xác nhận; M1 kích hoạt.
- Ưu tiên pullback continuation hoặc breakout-retest.
- Reversal cần sweep M15 + structure shift M5 + retest M1.
- Không trade giữa range M15, không chase nến M5 lớn.
- BUY/SELL cần score >= {minscore}, RR TP1 >= 1:{minrr}, trigger đã đóng.
- Tín hiệu phải hết hạn trong tối đa {maxage} phút.
- WAIT nếu setup tốt nhưng trigger chưa đủ. NO TRADE nếu không có edge.
- Confidence là chất lượng setup, không phải tỷ lệ thắng.
Trả đúng JSON schema, không markdown.'''

SCHEMA={"type":"OBJECT","properties":{
"final_verdict":{"type":"STRING","enum":["BUY","SELL","WAIT","NO TRADE"]},
"setup_score":{"type":"NUMBER"},"setup_grade":{"type":"STRING"},
"setup_model":{"type":"STRING","enum":["SCALP_PULLBACK_CONTINUATION","SCALP_BREAKOUT_RETEST","SCALP_CONFIRMED_REVERSAL","NONE"]},
"h1_bias":{"type":"STRING"},"m15_setup":{"type":"STRING"},"m5_confirmation":{"type":"STRING"},
"m1_trigger":{"type":"STRING"},"market_state":{"type":"STRING"},"entry_type":{"type":"STRING"},
"entry":{"type":"NUMBER","nullable":True},"stop_loss":{"type":"NUMBER","nullable":True},
"take_profit_1":{"type":"NUMBER","nullable":True},"take_profit_2":{"type":"NUMBER","nullable":True},
"signal_valid_minutes":{"type":"NUMBER"},"invalidation":{"type":"STRING"},
"wait_condition":{"type":"STRING"},"main_risk":{"type":"STRING"},"reason":{"type":"STRING"}},
"required":["final_verdict","setup_score","setup_grade","setup_model","h1_bias","m15_setup","m5_confirmation",
"m1_trigger","market_state","entry_type","entry","stop_loss","take_profit_1","take_profit_2",
"signal_valid_minutes","invalidation","wait_condition","main_risk","reason"]}


def gemini(images:dict[str,bytes],context:dict[str,Any],n:News,ps:int,pr:list[str],cb:str)->dict[str,Any]:
    text=PROMPT.format(context=json.dumps(context,ensure_ascii=False,indent=2),ps=ps,cb=cb,pr='; '.join(pr),
        news=n.text,minscore=MIN_SCORE,minrr=MIN_RR,maxage=MAX_AGE)
    parts=[{"text":text}]
    for label in ("H1","M15","M5","M1"):
        parts += [{"text":f"CHART {label}"},{"inline_data":{"mime_type":"image/png","data":base64.b64encode(images[label]).decode()}}]
    r=requests.post(GEMINI_URL,json={"contents":[{"parts":parts}],"generationConfig":{
        "temperature":0.05,"topP":0.75,"responseMimeType":"application/json","responseSchema":SCHEMA}},timeout=120)
    r.raise_for_status();p=r.json();c=p.get("candidates") or []
    if not c:raise RuntimeError(f"Gemini no candidate: {p}")
    t=c[0].get("content",{}).get("parts",[{}])[0].get("text","").strip()
    if t.startswith("```"):t=t.strip("`").replace("json\n","",1).strip()
    return json.loads(t)


def calc_rr(e:Any,s:Any,t:Any)->float|None:
    e,s,t=num(e),num(s),num(t)
    if None in (e,s,t):return None
    risk=abs(e-s);return round(abs(t-e)/risk,2) if risk else None


def neutralize(sig:dict[str,Any],verdict:str,reasons:list[str])->dict[str,Any]:
    x=dict(sig);x["final_verdict"]=verdict
    for k in ("entry","stop_loss","take_profit_1","take_profit_2"):x[k]=None
    if reasons:x["reason"]=(str(x.get("reason","")).strip()+" | Validator: "+'; '.join(reasons)).strip(' |')
    return x


def validate(sig:dict[str,Any],f:dict[str,pd.DataFrame],n:News,cb:str)->dict[str,Any]:
    x=dict(sig);v=str(x.get("final_verdict","NO TRADE")).upper().strip()
    if v not in {"BUY","SELL","WAIT","NO TRADE"}:v="NO TRADE"
    x["final_verdict"]=v;score=num(x.get("setup_score")) or 0;age=num(x.get("signal_valid_minutes")) or 0
    x["setup_grade"]="A" if score>=85 else "B" if score>=MIN_SCORE else "WATCHLIST" if score>=62 else "REJECTED"
    hard=[];wait=[]
    if n.veto:hard.append("news veto")
    if not n.available:hard.append("calendar unavailable")
    if v in {"BUY","SELL"}:
        e,s,t=num(x.get("entry")),num(x.get("stop_loss")),num(x.get("take_profit_1"))
        atr=num(f["M5"].atr14.iloc[-1]);cur=float(f["M1"].close.iloc[-1])
        if score<MIN_SCORE:wait.append("score below threshold")
        if not 1<=age<=MAX_AGE:hard.append("signal expiry invalid")
        if cb not in {v,"NEUTRAL"}:hard.append("opposes H1/M15 quantitative bias")
        if None in (e,s,t):hard.append("missing entry/SL/TP1")
        else:
            if v=="BUY" and not(s<e<t):hard.append("invalid BUY price order")
            if v=="SELL" and not(t<e<s):hard.append("invalid SELL price order")
            ratio=calc_rr(e,s,t)
            if ratio is None or ratio<MIN_RR:hard.append(f"RR below {MIN_RR}")
            if atr:
                slatr=abs(e-s)/atr
                if slatr<MIN_SL_ATR:hard.append("SL too tight vs M5 ATR")
                if slatr>MAX_SL_ATR:hard.append("SL too wide vs M5 ATR")
                if abs(e-cur)>atr*MAX_ENTRY_ATR:hard.append("entry too far from M1 price")
        weak=("chưa","đợi","wait","không rõ","none")
        if any(w in str(x.get("m1_trigger","")).lower() for w in weak):wait.append("M1 trigger incomplete")
        if any(w in str(x.get("m5_confirmation","")).lower() for w in weak):wait.append("M5 confirmation incomplete")
    if hard:return neutralize(x,"NO TRADE",hard+wait)
    if wait and v in {"BUY","SELL"}:return neutralize(x,"WAIT",wait)
    if v=="WAIT" and score<62:return neutralize(x,"NO TRADE",["below watchlist score"])
    if v in {"WAIT","NO TRADE"}:return neutralize(x,v,[])
    return x


def money(v:Any)->str:
    x=num(v);return f"{x:,.2f}" if x is not None else "N/A"


def message(s:dict[str,Any],n:News,ps:int)->str:
    v=s["final_verdict"];icon={"BUY":"🟢","SELL":"🔴","WAIT":"🟡","NO TRADE":"⚪"}[v]
    now=datetime.now(timezone.utc).astimezone(VN_TZ)
    lines=[f"{icon} <b>XAUUSD SCALP — {v}</b>",f"🕒 {now:%H:%M %d/%m/%Y} (VN)",
        f"📊 Score {s.get('setup_score','N/A')}/100 | {s.get('setup_grade','N/A')} | Prefilter {ps}",
        f"🧭 H1: {s.get('h1_bias','N/A')}",f"🎯 M15: {s.get('m15_setup','N/A')}",
        f"✅ M5: {s.get('m5_confirmation','N/A')}",f"⚡ M1: {s.get('m1_trigger','N/A')}",f"📰 {n.text}"]
    if v in {"BUY","SELL"}:
        lines += ["",f"🎬 {s.get('entry_type','N/A')}",f"💰 Entry: <code>{money(s.get('entry'))}</code>",
            f"🛑 SL: <code>{money(s.get('stop_loss'))}</code>",f"✅ TP1: <code>{money(s.get('take_profit_1'))}</code>"]
        if s.get("take_profit_2") is not None:lines.append(f"✅ TP2: <code>{money(s.get('take_profit_2'))}</code>")
        lines += [f"⚖️ R:R = 1:{calc_rr(s.get('entry'),s.get('stop_loss'),s.get('take_profit_1'))}",
            f"⏱ Hiệu lực: {s.get('signal_valid_minutes')} phút",f"🚫 Vô hiệu: {s.get('invalidation','N/A')}"]
    elif v=="WAIT":lines.append(f"⏳ Chờ: {s.get('wait_condition','N/A')}")
    lines += [f"⚠️ Rủi ro: {s.get('main_risk','N/A')}",f"📝 {s.get('reason','N/A')}","",
        "<i>Chỉ dùng để nghiên cứu/paper-trading; không tự động vào lệnh.</i>"]
    return '\n'.join(lines)


def log(s:dict[str,Any],n:News,ps:int,pr:list[str],ai:bool)->None:
    LOG_DIR.mkdir(parents=True,exist_ok=True);now=datetime.now(timezone.utc)
    rec={"timestamp_utc":now.isoformat(),"timestamp_vn":now.astimezone(VN_TZ).isoformat(),"ai_called":ai,
        "prefilter_score":ps,"prefilter_reasons":'; '.join(pr),"news_veto":n.veto,"news_text":n.text,**s,
        "rr_tp1":calc_rr(s.get('entry'),s.get('stop_loss'),s.get('take_profit_1'))}
    with (LOG_DIR/'signals.jsonl').open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
    p=LOG_DIR/'signals.csv';new=not p.exists()
    with p.open('a',encoding='utf-8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rec));
        if new:w.writeheader()
        w.writerow(rec)


def main()->None:
    now=datetime.now(timezone.utc)
    if not in_session(now) and not FORCE_AI:
        print(f"Outside scalp session {SCALP_SESSION_UTC} UTC");return
    frames=build_frames(fetch_m1(),now)
    need={"H1":40,"M15":80,"M5":100,"M1":120}
    bad=[k for k,v in need.items() if len(frames[k])<v]
    if bad:raise RuntimeError('Insufficient resampled data: '+','.join(bad))
    context={"symbol":SYMBOL,"method":"H1>M15>M5>M1","timeframes":{k:summary(v) for k,v in frames.items()}}
    ok,ps,pr,cb=prefilter(frames);n=news(now)
    if n.veto:ok=False;pr.append('news veto')
    if not ok:
        s={"final_verdict":"NO TRADE","setup_score":0,"setup_grade":"REJECTED","setup_model":"NONE",
           "h1_bias":bias(frames['H1']),"m15_setup":"Prefilter skip","m5_confirmation":"AI not called",
           "m1_trigger":"AI not called","market_state":"PREFILTER_SKIP","entry_type":"NONE","entry":None,
           "stop_loss":None,"take_profit_1":None,"take_profit_2":None,"signal_valid_minutes":0,
           "invalidation":"","wait_condition":"","main_risk":n.text,"reason":'; '.join(pr)}
        log(s,n,ps,pr,False);print('Prefilter skip:',s['reason']);return
    sizes={"H1":90,"M15":120,"M5":150,"M1":180}
    images={k:chart(frames[k],k,sizes[k]) for k in sizes}
    s=validate(gemini(images,context,n,ps,pr,cb),frames,n,cb);log(s,n,ps,pr,True)
    v=s['final_verdict'];send=v in {'BUY','SELL'} or (v=='WAIT' and SEND_WAIT) or (v=='NO TRADE' and SEND_NO_TRADE)
    if not send:print(v,'not sent');return
    if SEND_CHART:
        try:telegram_photo(images['M5'],f"XAUUSD M5 — {v}")
        except Exception as e:print('Photo warning:',e,file=sys.stderr)
    telegram(message(s,n,ps));print('Sent:',v)

if __name__=='__main__':main()
