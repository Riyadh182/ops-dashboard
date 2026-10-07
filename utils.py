"""Data + Gemini helpers (no Streamlit import, so they are easy to test)."""
import numpy as np
import pandas as pd
import requests

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
UNITS = ["TISWL-01", "TISWL-02", "TISWL-03", "TISWL-04", "TISWL-4B", "RGL"]


# ---------------------------------------------------------------- loading
def sheet_csv_url(sheet_id: str, tab: str) -> str:
    # Works when the Google Sheet is shared as "Anyone with the link: Viewer"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab}"


def read_tables(sheet_id: str = "", local_xlsx: str = "Dashboard_Data_Template.xlsx"):
    if sheet_id:
        data = pd.read_csv(sheet_csv_url(sheet_id, "Data"))
        rca = pd.read_csv(sheet_csv_url(sheet_id, "RCA"))
    else:
        data = pd.read_excel(local_xlsx, sheet_name="Data")
        rca = pd.read_excel(local_xlsx, sheet_name="RCA")
    return complete_grid(clean_data(data)), clean_rca(rca)


def _month(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str[:3].str.title()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Year", "Value", "Target"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "UoM" not in df:
        df["UoM"] = ""
    df["UoM"] = df["UoM"].fillna("").astype(str)
    df = df.dropna(subset=["Year", "Month", "Unit", "Topic", "Metric"])
    df["Value"] = df["Value"].fillna(0)  # blank value = 0
    df["Year"] = df["Year"].astype(int)
    df["Month"] = _month(df["Month"])
    df = df[df["Month"].isin(MONTHS)]
    for c in ("Unit", "Topic", "Metric"):
        df[c] = df[c].astype(str).str.strip()
    return df.drop_duplicates(["Year", "Month", "Unit", "Topic", "Metric"], keep="last").reset_index(drop=True)


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Every Unit x Topic x Metric gets all 12 months; months with no row = 0."""
    keys = df[["Year", "Unit", "Topic", "Metric"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"Month": MONTHS}), how="cross")
    out = grid.merge(df, how="left", on=["Year", "Unit", "Topic", "Metric", "Month"])
    out["Value"] = out["Value"].fillna(0)
    uom = df.groupby(["Unit", "Topic", "Metric"])["UoM"].first()
    out["UoM"] = out.set_index(["Unit", "Topic", "Metric"]).index.map(uom)
    # constant targets (e.g. 18L, 1.60, 99%) carry into empty months; varying ones (forecast) do not
    const = df.dropna(subset=["Target"]).groupby(["Unit", "Topic", "Metric"])["Target"].agg(["min", "max"])
    const = const[const["min"] == const["max"]]["min"]
    fill = out.set_index(["Unit", "Topic", "Metric"]).index.map(const)
    out["Target"] = out["Target"].fillna(pd.Series(fill, index=out.index))
    return out


def clean_rca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Unit", "Topic", "RCA"])
    df["Year"] = df["Year"].astype(int)
    df["Month"] = _month(df["Month"])
    df = df[df["Month"].isin(MONTHS)]
    for c in ("Unit", "Topic"):
        df[c] = df[c].astype(str).str.strip()
    return df.reset_index(drop=True)


def topic_sort_key(topic: str) -> int:
    try:
        return int(topic.split()[0][1:])
    except Exception:
        return 999


def build_index(df: pd.DataFrame, year: int):
    """{(unit, topic, metric): (values[12], targets[12])} for one year, missing = 0 / NaN."""
    d = df[df["Year"] == year]
    idx, mi = {}, {m: i for i, m in enumerate(MONTHS)}
    for (u, t, m), g in d.groupby(["Unit", "Topic", "Metric"]):
        v, tg = np.zeros(12), np.full(12, np.nan)
        for mon, val, tar in zip(g["Month"], g["Value"], g["Target"]):
            v[mi[mon]] = val
            tg[mi[mon]] = tar
        idx[(u, t, m)] = (v, tg)
    return idx


# ---------------------------------------------------------------- Gemini
def gemini(prompt_or_msgs, system: str, api_key: str, model: str = "gemini-3.8-flash", max_tokens: int = 1500) -> str:
    """Plain REST call to the Gemini API (free tier works). prompt_or_msgs = str or [{'role','content'}]."""
    if isinstance(prompt_or_msgs, str):
        prompt_or_msgs = [{"role": "user", "content": prompt_or_msgs}]
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in prompt_or_msgs]
    body = {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens,
                                 "thinkingConfig": ({"thinkingLevel": "low"} if "gemini-3" in model else {"thinkingBudget": 0})}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=60)
    if r.status_code == 400:  # model rejected the thinking setting: retry without it
        body["generationConfig"].pop("thinkingConfig")
        r = requests.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=60)
    if r.status_code == 429:
        raise RuntimeError("Gemini free limit reached for now. Wait a minute and try again.")
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:200]}")
    j = r.json()
    try:
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]).strip()
    except Exception:
        raise RuntimeError("Gemini returned no answer (maybe blocked). Try rephrasing.")


def fmt(x, dec=2):
    return f"{x:,.{dec}f}"


def topic_context(idx, rca, year, unit, topic, month, back=3) -> str:
    """Facts about one unit/topic/month for Gemini: numbers, previous months, targets, RCA text."""
    si = MONTHS.index(month)
    lo = max(0, si - back)
    lines = [f"Unit: {unit} | Topic: {topic} | Meeting month: {month} {year}"]
    for (u, t, m), (v, tg) in sorted(idx.items(), key=lambda kv: kv[0][2]):
        if u != unit or t != topic:
            continue
        vals = ", ".join(f"{MONTHS[i]}={v[i]:g}" for i in range(lo, si + 1))
        tgt = tg[si]
        lines.append(f"- {m}: {vals}" + ("" if np.isnan(tgt) else f" (target {tgt:g})"))
    if topic.startswith("T6"):
        ratio = idx.get((unit, topic, "Cut-to-ship ratio"))
        parts = {k: idx.get((unit, topic, f"Rejection - {k}")) for k in ("Sewing", "Wash", "Fabric", "Smpl&Other")}
        if ratio is not None and ratio[0][si] > 0:
            loss = 100 - ratio[0][si]
            lines.append(f"Total cut-to-ship loss in {month} = {loss:.2f}% of cut (100 - {ratio[0][si]:.2f}). Allowed loss at 99% target = 1.00%.")
            tot = sum(p[0][si] for p in parts.values() if p is not None)
            for k, p in parts.items():
                if p is not None and tot > 0:
                    lines.append(f"  responsible: {k} {p[0][si]:.2f}% of cut = {p[0][si] / tot * 100:.0f}% of the loss")
    r = rca[(rca.Year == year) & (rca.Unit == unit) & (rca.Topic == topic)]
    for _, row in r.sort_values("Month", key=lambda s: s.map(MONTHS.index)).iterrows():
        lines.append(f"RCA entered for {row['Month']} meeting: {row['RCA']}")
    return "\n".join(lines)


def ai_context(idx, rca: pd.DataFrame, year: int, max_chars: int = 400_000) -> str:
    """Compact text of the whole year for the Q&A box (zero rows left out)."""
    rows = []
    for (u, t, m), (v, tg) in idx.items():
        for i in range(12):
            if v[i] != 0:
                rows.append(f"{MONTHS[i]},{u},{t},{m},{v[i]:g},{'' if np.isnan(tg[i]) else format(tg[i], 'g')}")
    text = f"DATA {year} (Month,Unit,Topic,Metric,Value,Target). Any month/unit/metric not listed = 0 (no data).\n" + "\n".join(rows)
    text += "\n\nRCA (Month,Unit,Topic,RCA)\n" + "\n".join(
        f"{r.Month},{r.Unit},{r.Topic},{r.RCA}" for r in rca[rca.Year == year].itertuples())
    return text[:max_chars]
