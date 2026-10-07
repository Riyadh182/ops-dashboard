import html
import os

import numpy as np
import pandas as pd
import streamlit as st

from charts import CONFIG, SPECS, Acc, compare_chart, kpi_values, make_figs, spec_for, t5_fig, t6_breakdown
from utils import MONTHS, UNITS, ai_context, build_index, gemini, read_tables, topic_context, topic_sort_key

st.set_page_config(page_title="Ops Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look & feel
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; -webkit-text-size-adjust: 100%; touch-action: manipulation; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(139,92,246,.34), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(34,211,238,.22), transparent 55%),
    radial-gradient(800px 500px at 50% 110%, rgba(244,114,182,.14), transparent 60%), #070b1a;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1400px; padding: 1rem 1.2rem 4rem; }
@media (max-width: 640px) { .block-container { padding: .6rem .6rem 3rem; } }
.stApp, .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] { color:#e2e8f0; }
.stApp [data-testid="stCaptionContainer"] { color:#94a3b8; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,.07) !important; border-color: rgba(255,255,255,.16) !important; color:#fff !important; }
/* phones: 16px inputs stop iOS from zooming on tap; charts only scroll the page, never zoom/pan */
input, select, textarea, div[data-baseweb="select"] * { font-size: 16px !important; }
.stPlotlyChart, .js-plotly-plot, .plotly, .main-svg { touch-action: pan-y !important; }
button { touch-action: manipulation; }

.hero { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
  padding: 18px 20px; border-radius: 20px; margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(139,92,246,.28), rgba(34,211,238,.14) 60%, rgba(15,23,48,.6));
  border: 1px solid rgba(255,255,255,.12); box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.12);
  backdrop-filter: blur(14px); }
.hero h1 { margin:0; font-size: 1.7rem; font-weight: 800; letter-spacing:-.02em;
  background: linear-gradient(90deg,#fff,#a5f3fc 45%,#c4b5fd); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { margin:2px 0 0; color:#94a3b8; font-size:.85rem; }
.live { display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px; font-size:.75rem; font-weight:700;
  color:#a7f3d0; background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.35); letter-spacing:.06em; }
.live i { width:8px; height:8px; border-radius:50%; background:#34d399; box-shadow:0 0 10px #34d399; animation:pulse 1.6s infinite; }
@keyframes pulse { 50% { opacity:.35; } }

.sec { display:flex; align-items:center; gap:10px; margin: 26px 0 8px; font-weight:700; font-size:1.08rem; color:#f1f5f9; }
.sec .tcode { padding:3px 10px; border-radius:10px; font-size:.78rem; font-weight:800; color:#0b1020;
  background: linear-gradient(135deg,#22d3ee,#a78bfa); box-shadow:0 0 18px rgba(34,211,238,.35); }
.sec .nd { margin-left:auto; font-size:.68rem; font-weight:700; color:#fbbf24; padding:3px 9px; border-radius:999px;
  border:1px solid rgba(251,191,36,.4); background:rgba(251,191,36,.08); letter-spacing:.05em; }
.unitbar { margin: 4px 0 10px; font-weight:800; font-size:1.15rem; color:#fff; display:flex; align-items:center; gap:10px; }
.unitbar .chip { font-size:.72rem; padding:3px 10px; border-radius:999px; background:rgba(139,92,246,.25); border:1px solid rgba(167,139,250,.5); color:#ddd6fe; }

.kgrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin: 6px 0 8px; }
.kpi { padding: 12px 14px; border-radius:16px; position:relative; overflow:hidden;
  background: linear-gradient(160deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.10); box-shadow: 0 6px 24px rgba(0,0,0,.28); }
.kpi:before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:linear-gradient(#22d3ee,#8b5cf6); }
.kpi .kl { color:#94a3b8; font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.kpi .kv { font-size:1.55rem; font-weight:800; color:#f8fafc; font-variant-numeric: tabular-nums; line-height:1.25; }
.kpi .kv small { font-size:.78rem; color:#94a3b8; font-weight:600; margin-left:3px; }
.kd { font-size:.74rem; font-weight:700; font-variant-numeric: tabular-nums; }
.good { color:#34d399; } .bad { color:#fb7185; } .flat { color:#94a3b8; }

.score { display:grid; grid-template-columns: repeat(auto-fill, minmax(135px, 1fr)); gap:8px; margin: 6px 0 4px; }
.tile { padding:10px 12px; border-radius:14px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.09); }
.tile .tt { font-size:.68rem; font-weight:800; color:#a5b4fc; letter-spacing:.06em; }
.tile .tn { font-size:.7rem; color:#94a3b8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tile .tv { font-size:1.2rem; font-weight:800; color:#fff; font-variant-numeric: tabular-nums; }
.tile.good { border-color: rgba(52,211,153,.45); box-shadow: inset 0 0 20px rgba(52,211,153,.07); }
.tile.bad { border-color: rgba(251,113,133,.5); box-shadow: inset 0 0 20px rgba(251,113,133,.08); }

.rca { padding: 12px 14px; border-radius:16px; margin: 8px 0; font-size:.92rem; line-height:1.5; color:#e2e8f0; }
.rca.hit { background: linear-gradient(135deg, rgba(251,191,36,.16), rgba(244,114,182,.10)); border:1px solid rgba(251,191,36,.55);
  box-shadow: 0 0 28px rgba(251,191,36,.12); }
.rca.none { background: rgba(255,255,255,.04); border:1px dashed rgba(148,163,184,.35); color:#94a3b8; }
.rca.old { background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); color:#cbd5e1; font-size:.85rem; }
.mchip { display:inline-block; padding:2px 9px; margin-right:8px; border-radius:999px; font-size:.7rem; font-weight:800; letter-spacing:.05em;
  color:#0b1020; background:linear-gradient(135deg,#fbbf24,#f472b6); }
.mchip.dim { background:rgba(148,163,184,.25); color:#e2e8f0; }
.note { color:#94a3b8; font-size:.8rem; margin: 2px 0 6px; }
.note b { color:#fbbf24; }
.ai { padding: 14px 16px; border-radius:16px; margin: 6px 0 10px; background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(139,92,246,.14));
  border:1px solid rgba(34,211,238,.35); font-size:.92rem; line-height:1.55; color:#e2e8f0; }
.ai .aih { font-size:.7rem; font-weight:800; letter-spacing:.08em; color:#67e8f9; margin-bottom:6px; }

/* pills / segmented controls / tabs / buttons */
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button { border-radius:999px !important; border:1px solid rgba(255,255,255,.14) !important;
  background: rgba(255,255,255,.05) !important; color:#cbd5e1 !important; font-weight:600 !important; min-height:38px; }
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-segmented_controlActive"] {
  background: linear-gradient(135deg,#22d3ee,#8b5cf6) !important; color:#0b1020 !important; border-color: transparent !important;
  box-shadow: 0 0 18px rgba(34,211,238,.35) !important; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { border-radius:12px 12px 0 0; padding: 8px 18px; font-weight:700; }
.stButton > button { border-radius:12px; border:1px solid rgba(34,211,238,.5); background:rgba(34,211,238,.10); color:#a5f3fc; font-weight:700; min-height:42px; }
.stButton > button:hover { background:rgba(34,211,238,.22); border-color:#22d3ee; color:#fff; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data
def secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_data(ttl=300, show_spinner="Loading data...")  # re-reads the sheet every 5 minutes
def load(sheet_id):
    return read_tables(sheet_id)


try:
    data, rca = load(secret("SHEET_ID"))
except Exception as e:
    st.error(f"Could not read the data sheet: {e}")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def get_index(sheet_id, year):
    return build_index(data, year)


GEMINI_KEY = secret("GEMINI_API_KEY")
GEMINI_MODEL = secret("GEMINI_MODEL", "gemini-3.8-flash")
LANGS = {"Banglish": "Banglish (Bangla written in English letters, simple factory-floor words)",
         "English": "simple English", "বাংলা": "Bangla script"}

_n = [0]
def nk(prefix="k"):
    _n[0] += 1
    return f"{prefix}{_n[0]}"


# ------------------------------------------------------------------ small renderers
def fmtv(x, dec):
    return f"{x:,.{dec}f}"


def delta_html(d, dec, good, text):
    if d is None or abs(d) < 10 ** -(dec + 1):
        return f'<div class="kd flat">● no change {text}</div>'
    cls = "flat" if good is None else ("good" if (d < 0) == (good == "down") else "bad")
    return f'<div class="kd {cls}">{"▲" if d > 0 else "▼"} {d:+,.{dec}f} {text}</div>'


def kpi_html(ks):
    cards = []
    for k in ks:
        dp = None if k["prev"] is None else k["val"] - k["prev"]
        h = f'<div class="kpi"><div class="kl">{html.escape(k["label"])}</div>'
        h += f'<div class="kv">{fmtv(k["val"], k["dec"])}<small>{html.escape(k["suf"].strip())}</small></div>'
        h += delta_html(dp, k["dec"], k["good"], f'vs {k["prev_lab"]}') if k["prev"] is not None else ""
        if k["tgt"] is not None:
            dt = k["val"] - k["tgt"]
            h += delta_html(dt, k["dec"], k["good"], "vs target").replace("no change", "on target")
        cards.append(h + "</div>")
    return '<div class="kgrid">' + "".join(cards) + "</div>"


def tile_status(k):
    if k["good"] is None:
        return ""
    ref = k["tgt"] if k["tgt"] is not None else k["prev"]
    if ref is None or abs(k["val"] - ref) < 1e-9:
        return ""
    return "good" if ((k["val"] - ref) < 0) == (k["good"] == "down") else "bad"


@st.cache_data(ttl=86400, show_spinner=False)
def explain(ctx, lang, model, _key):
    system = (
        "You are a senior operations analyst for a garment factory (Ha-Meem Group, Ashulia Zone). "
        "You get facts for ONE unit, ONE topic and ONE meeting month, plus the short RCA typed by the unit team. "
        f"Write a detailed root-cause explanation in {LANGS[lang]}. Rules: use ONLY the facts given; never invent numbers, "
        "causes, names or dates; if the RCA is missing or thin, say what is unknown and what the unit should check. "
        "Use these short headers: What happened (numbers, change vs previous month, vs target) / Why it happened "
        "(expand the RCA into a clear cause chain; for Cut-to-Ship say which of Sewing, Wash, Fabric, Sample&Other is "
        "responsible for how many % of the loss) / Impact / Next actions (max 3 bullets). Max 180 words.")
    return gemini(ctx, system, _key, model)


def rca_block(idx, year, unit, topic, month):
    rr = rca[(rca.Year == year) & (rca.Unit == unit) & (rca.Topic == topic)]
    cur = rr[rr.Month == month]
    if cur.empty:
        st.markdown(f'<div class="rca none"><span class="mchip dim">{month.upper()} MEETING</span> No RCA entered for {month} yet.</div>',
                    unsafe_allow_html=True)
    else:
        for r in cur.itertuples():
            st.markdown(f'<div class="rca hit"><span class="mchip">{month.upper()} MEETING</span>✏ {html.escape(r.RCA)}</div>',
                        unsafe_allow_html=True)
    oth = rr[rr.Month != month]
    if not oth.empty:
        with st.expander(f"Other months' RCA ({len(oth)})"):
            for r in oth.sort_values("Month", key=lambda s: s.map(MONTHS.index)).itertuples():
                st.markdown(f'<div class="rca old"><span class="mchip dim">{r.Month.upper()}</span>{html.escape(r.RCA)}</div>',
                            unsafe_allow_html=True)
    key = nk("ai")
    if GEMINI_KEY:
        if st.button("✨ Explain this root cause (Gemini)", key=key):
            with st.spinner("Gemini is writing the detail..."):
                try:
                    ctx = topic_context(idx, rca, year, unit, topic, month)
                    txt = explain(ctx, (st.session_state.get("lang") or "Banglish"), GEMINI_MODEL, GEMINI_KEY)
                    st.markdown(f'<div class="ai"><div class="aih">✨ GEMINI · {unit} · {topic.split()[0]} · {month.upper()}</div>'
                                f'{html.escape(txt).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                    st.caption("AI-written from the sheet data and RCA. Please verify before sharing.")
                except Exception as e:
                    st.warning(str(e))
    else:
        st.caption("Add GEMINI_API_KEY in the app secrets to get the AI detail for this RCA.")


def topic_section(idx, year, unit, topic, sel, span, show_title=True):
    spec = spec_for(topic)
    acc = Acc(idx, unit, topic)
    ks = kpi_values(acc, spec["kpis"], sel)
    code, name = topic.split(" ", 1)
    empty = all(abs(k["arr"][sel]) < 1e-12 for k in ks)
    if show_title:
        st.markdown(f'<div class="sec"><span class="tcode">{code}</span>{html.escape(name)}'
                    f'{"<span class=nd>NO DATA · 0</span>" if empty else ""}</div>', unsafe_allow_html=True)
    st.markdown(kpi_html(ks), unsafe_allow_html=True)
    if code == "T5":
        st.plotly_chart(t5_fig(acc, sel), config=CONFIG, key=nk("t5"))
        if empty:
            have = [MONTHS[i] for i in range(12) if abs(ks[1]["arr"][i]) + abs(ks[2]["arr"][i]) > 0]
            if have:
                st.markdown(f'<div class="note">Manpower variance is entered for <b>{", ".join(have)}</b>. Pick that month to see it.</div>',
                            unsafe_allow_html=True)
    else:
        figs = make_figs(acc, spec, sel, span)
        if len(figs) == 2:
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs[0], config=CONFIG, key=nk("f"))
            c2.plotly_chart(figs[1], config=CONFIG, key=nk("f"))
        else:
            for f in figs:
                st.plotly_chart(f, config=CONFIG, key=nk("f"))
    if code == "T6":
        fig, info = t6_breakdown(acc, sel)
        if fig is None:
            st.markdown(f'<div class="note">Rejection split (Sewing / Wash / Fabric / Sample&Other) is not entered for <b>{MONTHS[sel]}</b>.</div>',
                        unsafe_allow_html=True)
        else:
            st.plotly_chart(fig, config=CONFIG, key=nk("t6"))
            sign = "over" if info["over"] > 0 else "within"
            st.markdown(f'<div class="note">Loss <b>{info["loss"]:.2f}%</b> vs allowed {info["allowed"]:.2f}% → '
                        f'<b>{abs(info["over"]):.2f} pp {sign}</b> the target. Bars add up to {info["total_parts"]:.2f}%.</div>',
                        unsafe_allow_html=True)
    rca_block(idx, year, unit, topic, MONTHS[sel])
    return ks


# ------------------------------------------------------------------ header + filters
st.markdown("""
<div class="hero"><div><h1>Operations Dashboard</h1><p>Ha-Meem Group · Ashulia Zone · 14 topics · 6 units</p></div>
<span class="live"><i></i>LIVE SHEET</span></div>""", unsafe_allow_html=True)

if GEMINI_KEY:
    st.segmented_control("AI language", list(LANGS), default="Banglish", key="lang")
tab_dash, tab_ai = st.tabs(["📊  Dashboard", "✨  Ask AI"])

years = sorted(data["Year"].unique(), reverse=True)
with tab_dash:
    c1, c2 = st.columns([1, 5])
    year = int(c1.selectbox("Year", years))
    ry = rca[rca.Year == year]
    default_m = ry["Month"].value_counts().idxmax() if not ry.empty else MONTHS[0]
    with c2:
        st.caption("Meeting month")
        m_pick = st.pills("Meeting month", MONTHS, selection_mode="single", default=default_m, key="month",
                          label_visibility="collapsed")
    month = m_pick or default_m
    sel = MONTHS.index(month)
    idx = get_index(secret("SHEET_ID"), year)

    cA, cB = st.columns([2, 3])
    with cA:
        st.caption("View")
        mode = st.segmented_control("View", ["Unit wise", "Topic wise"], default="Unit wise", key="mode",
                                    label_visibility="collapsed") or "Unit wise"
    with cB:
        st.caption("Chart range")
        span = st.segmented_control("Chart range", ["3 months", "Full year"], default="3 months", key="span",
                                    label_visibility="collapsed") or "3 months"
    topics = sorted(data["Topic"].unique(), key=topic_sort_key)
    units = [u for u in UNITS if u in set(data["Unit"])] + sorted(set(data["Unit"]) - set(UNITS))

    # ------------------------------------------------------------------ UNIT WISE
    if mode == "Unit wise":
        st.caption("Unit")
        unit = st.pills("Unit", units, selection_mode="single", default=units[0], key="unit", label_visibility="collapsed") or units[0]
        st.markdown(f'<div class="unitbar">{unit}<span class="chip">{month} {year}</span><span class="chip">all 14 topics</span></div>',
                    unsafe_allow_html=True)
        # scorecard: headline KPI of every topic
        tiles = []
        for t in topics:
            k = kpi_values(Acc(idx, unit, t), spec_for(t)["kpis"], sel)[0]
            code, name = t.split(" ", 1)
            tiles.append(f'<div class="tile {tile_status(k)}"><div class="tt">{code}</div><div class="tn">{html.escape(name)}</div>'
                         f'<div class="tv">{fmtv(k["val"], k["dec"])}<small style="font-size:.7rem;color:#94a3b8">{html.escape(k["suf"].strip())}</small></div></div>')
        st.markdown('<div class="score">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
        for t in topics:
            topic_section(idx, year, unit, t, sel, span)

    # ------------------------------------------------------------------ TOPIC WISE
    else:
        topic = st.selectbox("Topic", topics)
        code, name = topic.split(" ", 1)
        st.markdown(f'<div class="sec"><span class="tcode">{code}</span>{html.escape(name)}<span class="chip" style="margin-left:auto;font-size:.72rem;color:#ddd6fe">{month} {year}</span></div>',
                    unsafe_allow_html=True)
        # all-units comparison for the headline metric: previous month vs selected month
        k0 = [kpi_values(Acc(idx, u, topic), spec_for(topic)["kpis"], sel)[0] for u in units]
        prev_lab = MONTHS[sel - 1] if sel > 0 else "—"
        prev_vals = [k["prev"] if k["prev"] is not None else 0 for k in k0]
        tg = np.array([np.nan if k["tgt"] is None else k["tgt"] for k in k0])
        st.plotly_chart(compare_chart(f'{k0[0]["label"]} · all units', units, prev_vals, [k["val"] for k in k0], tg,
                                      prev_lab, month, k0[0]["dec"]), config=CONFIG, key=nk("cmp"))
        for i in range(0, len(units), 2):
            cols = st.columns(2)
            for col, u in zip(cols, units[i:i + 2]):
                with col:
                    with st.container(border=True):
                        st.markdown(f'<div class="unitbar">{u}</div>', unsafe_allow_html=True)
                        topic_section(idx, year, u, topic, sel, span, show_title=False)

# ------------------------------------------------------------------ ASK AI
with tab_ai:
    if not GEMINI_KEY:
        st.info("Add GEMINI_API_KEY in the app secrets to switch on the question box (free key: aistudio.google.com).")
    else:
        ay = year
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for m in st.session_state.chat:
            st.chat_message(m["role"]).write(m["content"])
        q = st.chat_input("Ask about the data, e.g. Aug e kon unit cost target er upore?")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            st.chat_message("user").write(q)
            system = ("You answer questions about a garment factory operations dashboard. Use ONLY the data below. "
                      "If the answer is not in the data, say so. Reply in the same language/style as the question "
                      "(Bangla, English or Banglish). Mention unit, month and number. Explain the root cause from the RCA text "
                      "when asked why. For Cut-to-Ship, say which section (Sewing/Wash/Fabric/Smpl&Other) is responsible for how many %. "
                      "Be short.\n\n" + ai_context(build_index(data, ay), rca, ay))
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ans = gemini(st.session_state.chat[-10:], system, GEMINI_KEY, GEMINI_MODEL)
                    except Exception as e:
                        ans = f"⚠️ {e}"
                st.write(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})
