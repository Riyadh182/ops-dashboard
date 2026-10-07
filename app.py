import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import MONTHS, ai_context, read_tables, topic_sort_key

st.set_page_config(page_title="Ops Dashboard", page_icon="📊", layout="centered")


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

st.title("📊 Operations Dashboard")
st.caption("Ha-Meem Group, Ashulia Zone")

# ---------- filters ----------
years = sorted(data["Year"].unique(), reverse=True)
c1, c2 = st.columns(2)
year = c1.selectbox("Year", years)
units = ["All units"] + sorted(data["Unit"].unique())
unit = c2.selectbox("Unit", units)
topics = sorted(data["Topic"].unique(), key=topic_sort_key)
topic = st.selectbox("Topic", topics)

tab_dash, tab_ai = st.tabs(["Dashboard", "Ask AI"])

# ---------- dashboard ----------
with tab_dash:
    d = data[(data["Year"] == year) & (data["Topic"] == topic)]
    metrics = list(d["Metric"].unique())
    if d.empty:
        st.info("No data for this selection yet.")
    elif unit == "All units":
        metric = st.selectbox("Metric", metrics)
        dm = d[d["Metric"] == metric]
        uom = dm["UoM"].dropna().iloc[0] if dm["UoM"].notna().any() else ""
        fig = go.Figure()
        for u, g in dm.groupby("Unit", observed=True):
            g = g.sort_values("Month")
            fig.add_trace(go.Scatter(x=g["Month"].astype(str), y=g["Value"], mode="lines+markers", name=u))
        tg = dm["Target"].dropna()
        if not tg.empty:
            fig.add_hline(y=tg.iloc[0], line_dash="dash", line_color="red", annotation_text="Target")
        fig.update_layout(title=f"{metric} ({uom})", height=360, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(categoryorder="array", categoryarray=MONTHS)
        st.plotly_chart(fig, use_container_width=True)
    else:
        du = d[d["Unit"] == unit]
        if du.empty:
            st.info("No data for this unit yet.")
        # KPI cards: latest month per metric vs target
        latest = du.sort_values("Month").groupby("Metric", observed=True).tail(1)
        cols = st.columns(2)
        for i, (_, r) in enumerate(latest.iterrows()):
            delta = None if pd.isna(r["Target"]) else round(r["Value"] - r["Target"], 2)
            cols[i % 2].metric(f"{r['Metric']} ({r['Month']})", f"{r['Value']:,.2f}",
                               None if delta is None else f"{delta:+,.2f} vs target", delta_color="off")
        # one chart per metric
        for m in metrics:
            g = du[du["Metric"] == m].sort_values("Month")
            if g.empty:
                continue
            fig = go.Figure(go.Scatter(x=g["Month"].astype(str), y=g["Value"], mode="lines+markers+text",
                                       text=g["Value"].round(2), textposition="top center", name="Actual"))
            tg = g["Target"].dropna()
            if not tg.empty:
                fig.add_trace(go.Scatter(x=g["Month"].astype(str), y=g["Target"], mode="lines",
                                         line=dict(dash="dash", color="red"), name="Target"))
            fig.update_layout(title=m, height=260, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            fig.update_xaxes(categoryorder="array", categoryarray=MONTHS)
            st.plotly_chart(fig, use_container_width=True)

    # month-wise root cause (selected year / topic / unit)
    st.subheader("Root cause by month")
    rr = rca[(rca["Year"] == year) & (rca["Topic"] == topic)]
    if unit != "All units":
        rr = rr[rr["Unit"] == unit]
    if rr.empty:
        st.caption("No RCA entered yet.")
    for _, r in rr.sort_values(["Month", "Unit"]).iterrows():
        st.info(f"**{r['Month']} - {r['Unit']}**\n\n{r['RCA']}")

# ---------- ask AI ----------
with tab_ai:
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        st.info("Add ANTHROPIC_API_KEY in the app secrets to switch on the question box.")
    else:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
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
                      "(Bangla, English or Banglish). Mention unit, month and number. Be short.\n\n"
                      + ai_context(data, rca, year))
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1000, system=system,
                                                  messages=st.session_state.chat[-10:])
                    ans = resp.content[0].text
                st.write(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})
