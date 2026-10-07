"""Data helpers (no Streamlit import, so they are easy to test)."""
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
    return clean_data(data), clean_rca(rca)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Unit", "Topic", "Metric", "Value"])
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(str).str.strip().str[:3].str.title()
    df = df[df["Month"].isin(MONTHS)]
    df["Month"] = pd.Categorical(df["Month"], categories=MONTHS, ordered=True)
    return df.sort_values(["Topic", "Metric", "Unit", "Year", "Month"]).reset_index(drop=True)


def clean_rca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Unit", "Topic", "RCA"])
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(str).str.strip().str[:3].str.title()
    df["Month"] = pd.Categorical(df["Month"], categories=MONTHS, ordered=True)
    return df.sort_values(["Topic", "Unit", "Year", "Month"]).reset_index(drop=True)


def topic_sort_key(topic: str) -> int:
    try:
        return int(topic.split()[0][1:])
    except Exception:
        return 999


def ai_context(data: pd.DataFrame, rca: pd.DataFrame, year: int, max_chars: int = 120_000) -> str:
    d = data[data["Year"] == year].copy()
    d["Month"] = d["Month"].astype(str)
    r = rca[rca["Year"] == year].copy()
    r["Month"] = r["Month"].astype(str)
    text = "DATA (Year,Month,Unit,Topic,Metric,Value,Target,UoM)\n" + d.drop(columns=["Year"]).to_csv(index=False)
    text += "\nRCA (Month,Unit,Topic,RCA)\n" + r.drop(columns=["Year"]).to_csv(index=False)
    return text[:max_chars]
