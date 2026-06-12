# -*- coding: utf-8 -*-
"""
MA Signal Monitor (SET)  —  เฝ้าสัญญาณอย่างเดียว ไม่ยิงคำสั่ง
=============================================================

ดึงราคาหุ้นไทยจริงจาก Yahoo Finance (ใส่ .BK ท้ายชื่อหุ้นให้อัตโนมัติ)
เฝ้าสัญญาณ MA แล้วเด้งเตือน — เห็นสัญญาณแล้วไปกดซื้อ/ขายเองในแอปโบรก

ตรรกะ (เช็คเฉพาะแท่งที่ปิดแล้ว):
  สัญญาณซื้อ — 2 แท่งติด ปิดแตะหรือสูงกว่า MA(ซื้อ)
  สัญญาณขาย — แท่งแรกปิดแตะ MA(ขาย), แท่งสองปิดแตะหรือต่ำกว่า MA(ขาย)

วิธีรันบนคอม:
    py -m pip install streamlit pandas numpy yfinance
    py -m streamlit run monitor.py

เอาขึ้น Streamlit Cloud (เปิดบนมือถือได้):
    1. อัปไฟล์นี้ + requirements.txt ขึ้น GitHub
    2. เชื่อมที่ share.streamlit.io -> ได้ลิงก์เปิดจากมือถือทุกที่
    (ตัวนี้ไม่มี credential/เงินจริง จึงปลอดภัยที่จะวางบน cloud)

หมายเหตุ: เครื่องมือช่วยเฝ้าสัญญาณ ไม่ใช่คำแนะนำการลงทุน
ราคาจาก Yahoo อาจดีเลย์ ~15 นาที ใช้ดูแนวโน้ม/สัญญาณ ไม่ใช่ราคา realtime เป๊ะ
"""

import io
import wave
import struct
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ===========================================================================
# SIGNAL LOGIC
# ===========================================================================
def add_ma(df, period, col):
    df[col] = df["close"].rolling(period).mean()
    return df


def check_buy(df, tol):
    if len(df) < 2:
        return False
    c1, c2 = df.iloc[-2], df.iloc[-1]
    if pd.isna(c1.ma_buy) or pd.isna(c2.ma_buy):
        return False
    return (c1.close >= c1.ma_buy * (1 - tol)) and (c2.close >= c2.ma_buy * (1 - tol))


def check_sell(df, tol):
    if len(df) < 2:
        return False
    c1, c2 = df.iloc[-2], df.iloc[-1]
    if pd.isna(c1.ma_sell) or pd.isna(c2.ma_sell):
        return False
    cond1 = abs(c1.close - c1.ma_sell) <= c1.ma_sell * tol
    cond2 = c2.close <= c2.ma_sell
    return bool(cond1 and cond2)


def bars_above_below(df, ma_col):
    sub = df.dropna(subset=[ma_col])
    if sub.empty:
        return "—"
    above = sub["close"].iloc[-1] >= sub[ma_col].iloc[-1]
    count = 0
    for _, row in sub.iloc[::-1].iterrows():
        if (row["close"] >= row[ma_col]) == above:
            count += 1
        else:
            break
    return f"ยืน{'เหนือเส้น' if above else 'ใต้เส้น'} {count} แท่ง"


# ===========================================================================
# DATA (Yahoo Finance)
# ===========================================================================
@st.cache_data(ttl=60)
def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """ดึงราคาจาก Yahoo Finance. หุ้นไทยใส่ .BK ท้ายชื่อ"""
    import yfinance as yf

    yf_symbol = symbol if "." in symbol else f"{symbol}.BK"
    period = "2y" if interval == "1d" else "3mo"

    raw = yf.download(yf_symbol, period=period, interval=interval,
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # บาง version คืน column แบบ MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = pd.DataFrame({
        "time": raw.index,
        "open": pd.to_numeric(raw["Open"], errors="coerce"),
        "high": pd.to_numeric(raw["High"], errors="coerce"),
        "low": pd.to_numeric(raw["Low"], errors="coerce"),
        "close": pd.to_numeric(raw["Close"], errors="coerce"),
    }).dropna().reset_index(drop=True)
    return out


def make_beep() -> bytes:
    """สร้างเสียง beep สั้นๆ (wav) ไว้เตือน"""
    rate, dur, freq = 44100, 0.4, 880
    n = int(rate * dur)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(n):
            val = int(32767 * 0.4 * np.sin(2 * np.pi * freq * i / rate))
            w.writeframes(struct.pack("<h", val))
    return buf.getvalue()


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="MA Signal Monitor", page_icon="📡", layout="wide")
st.title("📡 MA Signal Monitor")
st.caption("เฝ้าสัญญาณอย่างเดียว — เห็นสัญญาณแล้วไปกดซื้อ/ขายเองในแอปโบรก")

INTERVALS = {"15 นาที": "15m", "30 นาที": "30m", "60 นาที": "60m", "รายวัน": "1d"}

with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    def row(label, ratio=(5, 4)):
        """แบ่ง label ซ้าย + ช่องค่าขวา บรรทัดเดียวกัน (ประหยัดพื้นที่)"""
        a, b = st.columns(ratio)
        a.markdown(f"<div style='padding-top:8px;font-size:0.9rem'>{label}</div>",
                   unsafe_allow_html=True)
        return b

    symbol = row("Symbol").text_input(
        "s", value="SINGER", label_visibility="collapsed").strip().upper()
    tf_label = row("Timeframe").selectbox(
        "tf", list(INTERVALS.keys()), index=2, label_visibility="collapsed")
    interval = INTERVALS[tf_label]

    st.divider()
    ma_buy = row("MA ซื้อ").number_input(
        "mb", 2, 200, 13, 1, label_visibility="collapsed")
    ma_sell = row("MA ขาย").number_input(
        "ms", 2, 200, 13, 1, label_visibility="collapsed")
    touch_tol = row("Tolerance %").number_input(
        "tt", 0.0, 2.0, 0.1, 0.05, label_visibility="collapsed") / 100
    chart_days = row("กราฟย้อนหลัง (วัน)").number_input(
        "cd", 1, 365, 3, 1, label_visibility="collapsed")
    ignore_forming = st.checkbox("ไม่นับแท่งที่กำลังวิ่ง (แนะนำเปิด)", value=True)

    st.divider()
    price_filter_on = st.checkbox(
        "กรองด้วยช่วงราคา (เฉพาะสัญญาณซื้อ)", value=True,
        help="เตือนซื้อเฉพาะตอนราคาอยู่ในช่วงที่ตั้งไว้ "
             "เช่น 6–10 บาท ถ้าเข้าสัญญาณซื้อแต่ราคา 12 บาท จะไม่เตือน "
             "(สัญญาณขายยังเตือนทุกราคา)")
    price_min = row("ราคาต่ำสุด").number_input(
        "pmin", 0.0, 10000.0, 6.0, 0.5, label_visibility="collapsed")
    price_max = row("ราคาสูงสุด").number_input(
        "pmax", 0.0, 10000.0, 10.0, 0.5, label_visibility="collapsed")

    st.divider()
    sound_on = st.checkbox("🔔 เปิดเสียงเตือน", value=True)
    auto_refresh_sec = row("รีเฟรช (วิ) 0=ปิด").number_input(
        "ar", 0, 3600, 60, 10, label_visibility="collapsed")
    st.button("🔄 ดึงข้อมูลใหม่", use_container_width=True,
              on_click=st.cache_data.clear)

# auto-refresh
if auto_refresh_sec and auto_refresh_sec > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=auto_refresh_sec * 1000, key="tick")
    except ImportError:
        st.sidebar.info("อยากให้รีเฟรชเอง: `pip install streamlit-autorefresh`")

# ---- โหลดข้อมูล ----
df = get_data(symbol, interval)
if df.empty:
    st.error(f"ดึงข้อมูล {symbol} ไม่ได้ — เช็คชื่อหุ้น (เช่น SINGER, CBG, PTT) "
             f"หรือลองใหม่อีกครั้ง")
    st.stop()

# คำนวณ MA จากข้อมูลเต็ม (รวมแท่งล่าสุด) เพื่อให้กราฟแสดงครบทุกแท่ง
df = add_ma(df, ma_buy, "ma_buy")
df = add_ma(df, ma_sell, "ma_sell")

need = max(ma_buy, ma_sell) + 2
if len(df) < need:
    st.error(f"ข้อมูลไม่พอ: ต้องการ {need} แท่ง (มี {len(df)}) ลองลดค่า MA")
    st.stop()

# สัญญาณใช้เฉพาะแท่งที่ปิดแล้ว (ตัดแท่งที่กำลังวิ่งออก) — กราฟยังโชว์ครบ
df_eval = df.iloc[:-1].copy() if ignore_forming and len(df) > 1 else df.copy()

# ตรวจทั้งซื้อและขาย (โหมดเฝ้าสัญญาณ ดูทั้งสองทาง พี่ตัดสินใจเอง)
buy_sig = check_buy(df_eval, touch_tol)
sell_sig = check_sell(df_eval, touch_tol)

last = df_eval.iloc[-1]
price = float(last.close)
bar_id = str(last["time"])

# กรองสัญญาณซื้อด้วยช่วงราคา (ซื้อเฉพาะตอนราคาอยู่ในโซนที่ตั้งไว้)
in_band = (price_min <= price <= price_max)
buy_blocked = buy_sig and price_filter_on and not in_band
if buy_blocked:
    buy_sig = False   # นอกช่วงราคา -> ไม่เตือนซื้อ

# ---- ตัวเลขสำคัญ ----
c1, c2, c3 = st.columns(3)
c1.metric("ราคา (ปิดล่าสุด)", f"{price:.2f}")
c2.metric(f"MA{ma_buy} (ซื้อ)", f"{last.ma_buy:.2f}")
c3.metric(f"MA{ma_sell} (ขาย)", f"{last.ma_sell:.2f}")

s1, s2 = st.columns(2)
s1.info(f"vs MA ซื้อ: {bars_above_below(df_eval, 'ma_buy')}")
s2.info(f"vs MA ขาย: {bars_above_below(df_eval, 'ma_sell')}")

# ---- แบนเนอร์สัญญาณ + เตือน ----
fired = None
if buy_sig:
    st.success(f"### 🟢 สัญญาณ **ซื้อ** {symbol} @ {price:.2f}\nเข้าเงื่อนไข 2 แท่งปิดแตะ/เหนือ MA ซื้อ")
    fired = "ซื้อ"
elif sell_sig:
    st.error(f"### 🔴 สัญญาณ **ขาย** {symbol} @ {price:.2f}\nเข้าเงื่อนไขแตะ แล้วแตะ/ต่ำกว่า MA ขาย")
    fired = "ขาย"
else:
    st.warning("⏳ รอสัญญาณ — ยังไม่เข้าเงื่อนไข")

if buy_blocked:
    st.caption(f"ℹ️ เข้าเงื่อนไขซื้อแล้ว แต่ราคา {price:.2f} อยู่นอกช่วง "
               f"{price_min:.2f}–{price_max:.2f} จึงไม่เตือนซื้อ")

# เตือนครั้งเดียวต่อแท่ง (toast + เสียง)
st.session_state.setdefault("last_alert_bar", None)
if fired and st.session_state.last_alert_bar != bar_id:
    st.session_state.last_alert_bar = bar_id
    st.toast(f"สัญญาณ{fired} {symbol} @ {price:.2f}", icon="🔔")
    # เก็บ log
    st.session_state.setdefault("signal_log", [])
    st.session_state.signal_log.insert(
        0, {"เวลา": dt.datetime.now().strftime("%d/%m %H:%M"),
            "หุ้น": symbol, "สัญญาณ": fired, "ราคา": round(price, 2)})
    if sound_on:
        st.audio(make_beep(), format="audio/wav", autoplay=True)

# ---- กราฟแท่งเทียน (เรียงชิด ไม่มีช่องว่างเวลาปิดตลาด แบบ Finansia) ----
cutoff = df["time"].iloc[-1] - pd.Timedelta(days=int(chart_days))
plot_df = df[df["time"] >= cutoff].copy()
if plot_df.empty:
    plot_df = df.tail(20).copy()

plot_df["label"] = plot_df["time"].dt.strftime("%d/%m %H:%M")
plot_df["order"] = range(len(plot_df))          # ใช้ลำดับเป็นแกน X -> แท่งเรียงชิด
plot_df["up"] = plot_df["close"] >= plot_df["open"]
UP, DOWN = "#26a69a", "#ef5350"

x_enc = alt.X("label:N", sort=alt.SortField("order"),
              axis=alt.Axis(title=None, labelAngle=-90, labelOverlap=True))
base = alt.Chart(plot_df)

# ไส้เทียน (high-low)
wick = base.mark_rule().encode(
    x=x_enc,
    y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="ราคา (บาท)"),
    y2="high:Q",
    color=alt.condition("datum.up", alt.value(UP), alt.value(DOWN)))
# ตัวเทียน (open-close)
body = base.mark_bar(size=6).encode(
    x=x_enc, y="open:Q", y2="close:Q",
    color=alt.condition("datum.up", alt.value(UP), alt.value(DOWN)))
# เส้น MA
ma_b = base.mark_line(color="#2196f3", strokeWidth=1.5).encode(x=x_enc, y="ma_buy:Q")
ma_s = base.mark_line(color="#ff9800", strokeWidth=1.5).encode(x=x_enc, y="ma_sell:Q")

st.altair_chart((wick + body + ma_b + ma_s).properties(height=400),
                use_container_width=True)
st.caption(f"🕯️ แท่งเทียน {symbol} | เส้นน้ำเงิน = MA{ma_buy}(ซื้อ) | "
           f"เส้นส้ม = MA{ma_sell}(ขาย) | เขียว=ขึ้น แดง=ลง")

# ---- ประวัติสัญญาณ ----
if st.session_state.get("signal_log"):
    st.subheader("📜 ประวัติสัญญาณ (รอบนี้)")
    st.dataframe(pd.DataFrame(st.session_state.signal_log[:20]),
                 use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"ราคาจาก Yahoo Finance ({symbol}.BK) อาจดีเลย์ ~15 นาที • "
    "เครื่องมือช่วยเฝ้าสัญญาณ ไม่ใช่คำแนะนำการลงทุน • "
    "เห็นสัญญาณแล้วไปกดซื้อ/ขายเองในแอปโบรก"
)
