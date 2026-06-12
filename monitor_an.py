# -*- coding: utf-8 -*-
"""
MA Signal Monitor (SET) — ตัวจอมือถือ/แท็บเล็ต + เตือนซ้ำจนกดดับ
================================================================

ปรับจากตัวเดิม 2 อย่าง:
  1) จัดหน้าจอกระชับสำหรับมือถือ/แท็บเล็ต (ราคา/MA รวมบรรทัดเดียว)
  2) เตือนซ้ำต่อเนื่อง:
       - สัญญาณซื้อ : ตี๊ด ทุก 1 วินาที วนไปเรื่อยๆ
       - สัญญาณขาย : ตี๊ดๆ (สองครั้ง) วนไปเรื่อยๆ
     ดังจนกว่าจะกดปุ่ม "🔕 ดับเสียง" จึงหยุด (เงียบจนกว่าจะมีสัญญาณใหม่)

วิธีรันบนคอม:
    py -m pip install streamlit pandas numpy yfinance streamlit-autorefresh
    py -m streamlit run monitor_an.py

เอาขึ้น Streamlit Cloud: อัปไฟล์นี้ + requirements.txt ขึ้น GitHub แล้ว deploy
(ตั้ง Main file path เป็น monitor_an.py)

หมายเหตุ: เครื่องมือช่วยเฝ้าสัญญาณ ไม่ใช่คำแนะนำการลงทุน
ราคาจาก Yahoo อาจดีเลย์ ~15 นาที
ข้อจำกัดมือถือ: ถ้าล็อกจอ/สลับแอป เบราว์เซอร์อาจหยุดเสียงชั่วคราว (เป็นข้อจำกัดของเว็บ)
"""

import io
import wave
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
def _to_df(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return pd.DataFrame({
        "time": raw.index,
        "open": pd.to_numeric(raw["Open"], errors="coerce"),
        "high": pd.to_numeric(raw["High"], errors="coerce"),
        "low": pd.to_numeric(raw["Low"], errors="coerce"),
        "close": pd.to_numeric(raw["Close"], errors="coerce"),
    }).dropna().reset_index(drop=True)


def _resample_intraday(df, rule):
    """รวมแท่งย่อยเป็นแท่งใหญ่ (เช่น 15m -> 45m) แยกตามวัน ให้ตรงเวลาเปิดตลาด"""
    if df.empty:
        return df
    d = df.set_index("time")
    frames = []
    for _, g in d.groupby(d.index.normalize()):
        r = g.resample(rule, origin="start").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        frames.append(r)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).reset_index()


@st.cache_data(ttl=45)
def get_data(symbol: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    yf_symbol = symbol if "." in symbol else f"{symbol}.BK"

    # 45 นาที: Yahoo ไม่มีให้ตรงๆ -> ดึง 15 นาทีมารวมเป็น 45 นาที
    if interval == "45m":
        raw = yf.download(yf_symbol, period="1mo", interval="15m",
                          progress=False, auto_adjust=False)
        return _resample_intraday(_to_df(raw), "45min")

    # เลือกช่วงเวลาตามขีดจำกัด Yahoo (15/30 นาที ย้อนได้แค่ ~60 วัน)
    if interval == "1d":
        period = "2y"
    elif interval in ("60m", "90m"):
        period = "3mo"
    else:                       # 15m, 30m
        period = "1mo"

    raw = yf.download(yf_symbol, period=period, interval=interval,
                      progress=False, auto_adjust=False)
    return _to_df(raw)


# ===========================================================================
# เสียงเตือน — สร้าง wav เป็นแพทเทิร์น 1 วินาที แล้ววน (loop) ในเบราว์เซอร์
# ===========================================================================
def _wav(segments, rate=44100):
    """segments = list ของ (freq|None, dur_seconds) ; None = เงียบ"""
    parts = []
    for freq, dur in segments:
        t = np.arange(int(rate * dur))
        parts.append(0.45 * np.sin(2 * np.pi * freq * t / rate) if freq
                     else np.zeros(len(t)))
    sig = np.concatenate(parts) if parts else np.zeros(1)
    pcm = (sig * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ซื้อ: ตี๊ด 1 ครั้ง/วินาที | ขาย: ตี๊ดๆ 2 ครั้ง/วินาที
BEEP_BUY = _wav([(880, 0.15), (None, 0.85)])
BEEP_SELL = _wav([(990, 0.12), (None, 0.08), (990, 0.12), (None, 0.68)])


def play_loop(data: bytes):
    """เล่นเสียงวนต่อเนื่อง (รองรับ loop ถ้า Streamlit เวอร์ชันใหม่)"""
    try:
        st.audio(data, format="audio/wav", autoplay=True, loop=True)
    except TypeError:
        st.audio(data, format="audio/wav", autoplay=True)


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="MA Signal Monitor", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

INTERVALS = {"15 นาที": "15m", "30 นาที": "30m", "45 นาที": "45m",
             "60 นาที": "60m", "รายวัน": "1d"}

with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    def row(label, ratio=(5, 4)):
        a, b = st.columns(ratio)
        a.markdown(f"<div style='padding-top:8px;font-size:0.9rem'>{label}</div>",
                   unsafe_allow_html=True)
        return b

    st.session_state.setdefault("symbol_input", "SINGER")
    symbol = row("Symbol").text_input(
        "s", key="symbol_input", label_visibility="collapsed").strip().upper()

    recent = st.session_state.setdefault("recent", [])
    if recent:
        def _pick_recent():
            v = st.session_state.get("recent_pick")
            if v and v != "—":
                st.session_state.symbol_input = v
        row("เคยดู").selectbox("rp", ["—"] + recent, key="recent_pick",
                              label_visibility="collapsed", on_change=_pick_recent)
    tf_label = row("Timeframe").selectbox(
        "tf", list(INTERVALS.keys()), index=3, label_visibility="collapsed")
    interval = INTERVALS[tf_label]

    st.divider()
    ma_buy = row("MA ซื้อ").number_input("mb", 2, 200, 13, 1,
                                         label_visibility="collapsed")
    ma_sell = row("MA ขาย").number_input("ms", 2, 200, 13, 1,
                                         label_visibility="collapsed")
    touch_tol = row("Tolerance %").number_input(
        "tt", 0.0, 2.0, 0.1, 0.05, label_visibility="collapsed") / 100
    chart_days = row("กราฟย้อนหลัง (วัน)").number_input(
        "cd", 1, 365, 3, 1, label_visibility="collapsed")
    ignore_forming = st.checkbox("ไม่นับแท่งที่กำลังวิ่ง (แนะนำเปิด)", value=True)

    st.divider()
    price_filter_on = st.checkbox("กรองด้วยช่วงราคา (เฉพาะสัญญาณซื้อ)", value=True)
    auto_band = st.checkbox(
        "ช่วงราคาอัตโนมัติ (±% จากราคาปัจจุบัน)", value=True,
        help="ตั้งช่วงเป็น ราคาปัจจุบัน ±% ให้เอง เปลี่ยนหุ้นแล้วช่วงปรับตามอัตโนมัติ")
    band_pct = row("± %").number_input(
        "bp", 1.0, 90.0, 20.0, 1.0, label_visibility="collapsed")
    price_min = row("ต่ำสุด (แมนนวล)").number_input(
        "pmin", 0.0, 10000.0, 6.0, 0.5, label_visibility="collapsed")
    price_max = row("สูงสุด (แมนนวล)").number_input(
        "pmax", 0.0, 10000.0, 10.0, 0.5, label_visibility="collapsed")

    st.divider()
    sound_on = st.checkbox("🔔 เปิดเสียงเตือน", value=True)
    auto_refresh_sec = row("รีเฟรช (วิ) 0=ปิด").number_input(
        "ar", 0, 3600, 60, 10, label_visibility="collapsed")
    st.button("🔄 ดึงข้อมูลใหม่", use_container_width=True,
              on_click=st.cache_data.clear)

# ---- โหลดข้อมูล ----
df = get_data(symbol, interval)
if df.empty:
    st.error(f"ดึงข้อมูล {symbol} ({tf_label}) ไม่ได้ — เช็คชื่อหุ้น (เช่น SINGER, "
             f"CBG, PTT) หรือลองเปลี่ยน timeframe / กดดึงข้อมูลใหม่อีกครั้ง")
    st.stop()

# เก็บประวัติหุ้นที่เคยดู (ล่าสุดอยู่บน สูงสุด 10 ตัว)
_rec = st.session_state.recent
if symbol in _rec:
    _rec.remove(symbol)
_rec.insert(0, symbol)
del _rec[10:]

df = add_ma(df, ma_buy, "ma_buy")
df = add_ma(df, ma_sell, "ma_sell")

need = max(ma_buy, ma_sell) + 2
if len(df) < need:
    st.error(f"ข้อมูลไม่พอ: ต้องการ {need} แท่ง (มี {len(df)}) ลองลดค่า MA")
    st.stop()

df_eval = df.iloc[:-1].copy() if ignore_forming and len(df) > 1 else df.copy()
buy_sig = check_buy(df_eval, touch_tol)
sell_sig = check_sell(df_eval, touch_tol)

last = df_eval.iloc[-1]
price = float(last.close)          # ราคาแท่งที่ปิดแล้ว (ใช้ตัดสินสัญญาณ)
bar_id = str(last["time"])
price_latest = float(df.iloc[-1]["close"])             # ราคาล่าสุด (แท่งปัจจุบัน)
last_time = pd.Timestamp(df.iloc[-1]["time"])

if auto_band:
    band_lo = round(price_latest * (1 - band_pct / 100), 2)
    band_hi = round(price_latest * (1 + band_pct / 100), 2)
else:
    band_lo, band_hi = price_min, price_max

in_band = (band_lo <= price <= band_hi)
buy_blocked = buy_sig and price_filter_on and not in_band
if buy_blocked:
    buy_sig = False

fired = "ซื้อ" if buy_sig else ("ขาย" if sell_sig else None)

# ===========================================================================
# ระบบเตือนซ้ำจนกดดับ (รีเซ็ตเมื่อมีสัญญาณ "ใหม่")
# ===========================================================================
# ถือว่าเป็นสัญญาณใหม่เมื่อชนิดสัญญาณเปลี่ยน (None->ซื้อ, None->ขาย, ซื้อ->ขาย ฯลฯ)
if st.session_state.get("prev_fired") != fired:
    st.session_state.prev_fired = fired
    st.session_state.alarm_dismissed = False   # มีสัญญาณใหม่ -> เริ่มเตือนใหม่

alarm_on = bool(fired) and sound_on and not st.session_state.get("alarm_dismissed", False)

# ---- หัวจอแบบกระชับ (มือถือ/แท็บเล็ต) ----
st.markdown("#### 📡 MA Signal Monitor")
st.markdown(
    f"<div style='font-size:1.05rem'>"
    f"<b>{symbol}</b>　ราคา <b style='font-size:1.3rem'>{price_latest:.2f}</b>　·　"
    f"MA{ma_buy}(ซื้อ) {last.ma_buy:.2f}　·　MA{ma_sell}(ขาย) {last.ma_sell:.2f}"
    f"</div>", unsafe_allow_html=True)
st.caption(f"vs MA ซื้อ: {bars_above_below(df_eval, 'ma_buy')}　|　"
           f"vs MA ขาย: {bars_above_below(df_eval, 'ma_sell')}")
st.caption(f"⏱️ แท่งล่าสุด: {last_time.strftime('%d/%m %H:%M')} "
           f"(Yahoo ดีเลย์ ~15 นาที • สัญญาณคิดจากแท่งที่ปิดแล้ว @ {price:.2f})")

# ---- แบนเนอร์สัญญาณ ----
if buy_sig:
    st.success(f"🟢 **สัญญาณซื้อ {symbol} @ {price:.2f}** — 2 แท่งปิดแตะ/เหนือ MA ซื้อ")
elif sell_sig:
    st.error(f"🔴 **สัญญาณขาย {symbol} @ {price:.2f}** — แตะ แล้วแตะ/ต่ำกว่า MA ขาย")
else:
    st.warning("⏳ รอสัญญาณ — ยังไม่เข้าเงื่อนไข")

if price_filter_on:
    tag = f"±{band_pct:.0f}% อัตโนมัติ" if auto_band else "แมนนวล"
    st.caption(f"🔎 ช่วงราคาที่เฝ้าซื้อ: {band_lo:.2f}–{band_hi:.2f} ({tag})")
if buy_blocked:
    st.caption(f"ℹ️ เข้าเงื่อนไขซื้อ แต่ราคา {price:.2f} อยู่นอกช่วง "
               f"{band_lo:.2f}–{band_hi:.2f} จึงไม่เตือน")

# ---- เสียงเตือน + ปุ่มดับ ----
if alarm_on:
    play_loop(BEEP_BUY if fired == "ซื้อ" else BEEP_SELL)
    st.button(f"🔕 ดับเสียง (สัญญาณ{fired})", use_container_width=True, type="primary",
              on_click=lambda: st.session_state.update(alarm_dismissed=True))

# บันทึกประวัติ (ครั้งเดียวต่อสัญญาณใหม่)
if fired and st.session_state.get("logged_bar") != f"{fired}-{bar_id}":
    st.session_state.logged_bar = f"{fired}-{bar_id}"
    st.toast(f"สัญญาณ{fired} {symbol} @ {price:.2f}", icon="🔔")
    st.session_state.setdefault("signal_log", [])
    st.session_state.signal_log.insert(
        0, {"เวลา": dt.datetime.now().strftime("%d/%m %H:%M"),
            "หุ้น": symbol, "สัญญาณ": fired, "ราคา": round(price, 2)})

# ---- กราฟแท่งเทียน ----
cutoff = df["time"].iloc[-1] - pd.Timedelta(days=int(chart_days))
plot_df = df[df["time"] >= cutoff].copy()
if plot_df.empty:
    plot_df = df.tail(20).copy()

plot_df["label"] = plot_df["time"].dt.strftime("%d/%m %H:%M")
plot_df["order"] = range(len(plot_df))
plot_df["up"] = plot_df["close"] >= plot_df["open"]
UP, DOWN = "#26a69a", "#ef5350"

x_enc = alt.X("label:N", sort=alt.SortField("order"),
              axis=alt.Axis(title=None, labelAngle=-90, labelOverlap=True))
base = alt.Chart(plot_df)
wick = base.mark_rule().encode(
    x=x_enc, y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="ราคา (บาท)"),
    y2="high:Q", color=alt.condition("datum.up", alt.value(UP), alt.value(DOWN)))
body = base.mark_bar(size=6).encode(
    x=x_enc, y="open:Q", y2="close:Q",
    color=alt.condition("datum.up", alt.value(UP), alt.value(DOWN)))

# เส้น MA แบบมี legend บนกราฟ: ซื้อ=น้ำเงินทึบ / ขาย=ส้มเส้นประ
buy_lbl, sell_lbl = f"MA ซื้อ ({ma_buy})", f"MA ขาย ({ma_sell})"
ma_long = plot_df.melt(id_vars=["label", "order"],
                       value_vars=["ma_buy", "ma_sell"],
                       var_name="ma_type", value_name="ma_val")
ma_long["ma_type"] = ma_long["ma_type"].map({"ma_buy": buy_lbl, "ma_sell": sell_lbl})
ma_line = alt.Chart(ma_long).mark_line(strokeWidth=2.2).encode(
    x=x_enc,
    y=alt.Y("ma_val:Q", scale=alt.Scale(zero=False)),
    color=alt.Color("ma_type:N",
                    scale=alt.Scale(domain=[buy_lbl, sell_lbl],
                                    range=["#2196f3", "#ff9800"]),
                    legend=alt.Legend(title=None, orient="top", labelFontSize=13)),
    strokeDash=alt.StrokeDash("ma_type:N",
                              scale=alt.Scale(domain=[buy_lbl, sell_lbl],
                                              range=[[1, 0], [6, 4]]),
                              legend=None))

chart = (wick + body + ma_line).properties(height=340).resolve_scale(color="independent")
st.altair_chart(chart, use_container_width=True)
st.caption(f"🕯️ {symbol} | เขียว=แท่งขึ้น  แดง=แท่งลง  "
           f"(น้ำเงินทึบ=MA ซื้อ, ส้มเส้นประ=MA ขาย)")

# ---- ประวัติสัญญาณ ----
if st.session_state.get("signal_log"):
    with st.expander("📜 ประวัติสัญญาณ (รอบนี้)"):
        st.dataframe(pd.DataFrame(st.session_state.signal_log[:20]),
                     use_container_width=True, hide_index=True)

st.caption(f"ราคาจาก Yahoo ({symbol}.BK) อาจดีเลย์ ~15 นาที • ไม่ใช่คำแนะนำการลงทุน")

# ===========================================================================
# auto-refresh — ตอนมีเสียงเตือนให้รีเฟรชถี่ (เสียงจะได้ตี๊ดต่อเนื่อง)
# ===========================================================================
refresh = 2 if alarm_on else int(auto_refresh_sec)
if refresh and refresh > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh * 1000, key="tick")
    except ImportError:
        st.sidebar.info("อยากให้รีเฟรชเอง: `pip install streamlit-autorefresh`")
