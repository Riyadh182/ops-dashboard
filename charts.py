"""Plotly figure builders + per-topic chart/KPI specs (mirrors the PPT charts)."""
import numpy as np
import plotly.graph_objects as go
from utils import MONTHS

C = dict(violet="#8b5cf6", cyan="#22d3ee", pink="#f472b6", amber="#fbbf24", green="#34d399",
         red="#fb7185", blue="#60a5fa", slate="#64748b", orange="#fb923c")
# static plot = no zoom / pan / drag, so touching a chart on a phone just scrolls the page
CONFIG = {"staticPlot": True, "displayModeBar": False, "responsive": True}
SECTIONS = ["Line Chief", "Supervisor", "Input Man", "Operator", "Asst. Op", "Cutting", "Finishing", "QI", "Mechanic", "Electrician"]


def window(sel: int, span: str):
    if span == "Full year":
        return list(range(12))
    lo = max(0, sel - 2)
    return list(range(lo, lo + 3))


def num(v, dec=2):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.{dec}f}"


class Acc:
    """Accessor for one unit + topic: values / targets as 12-month arrays (missing = 0)."""
    def __init__(self, idx, unit, topic):
        self.idx, self.unit, self.topic = idx, unit, topic

    def v(self, metric):
        return self.idx.get((self.unit, self.topic, metric), (np.zeros(12), None))[0]

    def t(self, metric):
        r = self.idx.get((self.unit, self.topic, metric))
        return np.full(12, np.nan) if r is None else r[1]


def V(m): return lambda a: a.v(m)
def T(m): return lambda a: a.t(m)
def SUMV(*ms): return lambda a: sum(a.v(m) for m in ms)
def SUMT(*ms): return lambda a: sum(np.nan_to_num(a.t(m)) for m in ms)


def _layout(fig, title, height, ncats=3, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.97, xanchor="left", font=dict(size=14, color="#e2e8f0")),
        height=height, margin=dict(l=6, r=6, t=44, b=44 if legend else 12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#cbd5e1"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14 if ncats <= 3 else -0.2, font=dict(size=11)),
        showlegend=legend, bargap=0.28, bargroupgap=0.06, barmode="group", dragmode=False,
        uniformtext=dict(minsize=8, mode="show"))
    return fig


def col_chart(title, win, sel, bars, target=None, marks=None, dec=2, height=310, ylo=None, targets=None):
    """bars: [(name, arr12, color, text_list12|None)]; target/targets: [(name, arr12, color)]; marks: [(name, arr12, color)]."""
    labels = [f"<b>{MONTHS[i]}</b>" if i == sel else MONTHS[i] for i in win]
    x = [MONTHS[i] for i in win]
    full = len(win) > 3
    fig = go.Figure()
    allv = []
    for name, arr, color, txt in bars:
        vals = [float(arr[i]) for i in win]
        allv += vals
        text = [txt[i] for i in win] if txt else [num(v, dec) for v in vals]
        if ylo is not None:
            text = [t if v > 0 else "" for t, v in zip(text, vals)]
        fig.add_bar(x=x, y=vals, name=name, text=text, textposition="outside", cliponaxis=False,
                    marker=dict(color=color, opacity=[1 if i == sel else 0.5 for i in win],
                                line=dict(color="rgba(255,255,255,.55)", width=[1.5 if i == sel else 0 for i in win])),
                    textfont=dict(size=9 if full else 11, color="#e2e8f0"), textangle=-90 if full and len(bars) > 1 else 0,
                    hoverinfo="skip")
        if ylo is not None:  # truncated axis: show "0" for months without data
            z = [i for i, v in enumerate(vals) if v == 0]
            if z:
                fig.add_scatter(x=[x[i] for i in z], y=[ylo] * len(z), mode="text", text=["0"] * len(z),
                                textposition="top center", showlegend=False, hoverinfo="skip",
                                textfont=dict(size=11, color="#94a3b8"))
    for name, arr, color in (marks or []):
        vals = [float(arr[i]) for i in win]
        allv += vals
        fig.add_scatter(x=x, y=vals, mode="markers+text", name=name, text=[num(v, dec) for v in vals],
                        textposition="top center", marker=dict(size=9, color=color, symbol="diamond", line=dict(color="#0b1020", width=1)),
                        textfont=dict(size=10, color=color), hoverinfo="skip")
    for name, arr, color in ([target] if target else []) + list(targets or []):
        vals = [None if np.isnan(arr[i]) else float(arr[i]) for i in win]
        allv += [v for v in vals if v is not None]
        fig.add_scatter(x=x, y=vals, mode="lines", name=name, line=dict(color=color, width=2, dash="dash"), hoverinfo="skip")
    fig.add_vrect(x0=win.index(sel) - 0.5, x1=win.index(sel) + 0.5, fillcolor="rgba(34,211,238,.10)", line_width=0) if sel in win else None
    hi = max(allv + [0]); lo = min(allv + [0])
    pad = (hi - (ylo if ylo is not None else lo)) * 0.22 or 1
    rng = [ylo, hi + pad] if ylo is not None else [lo - (pad if lo < 0 else 0), hi + pad]
    fig.update_yaxes(range=rng, showgrid=True, gridcolor="rgba(148,163,184,.12)", zeroline=True,
                     zerolinecolor="rgba(148,163,184,.45)", showticklabels=False, fixedrange=True)
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=labels, fixedrange=True, showgrid=False)
    return _layout(fig, title, height, len(win))


def hbar(title, labels, values, colors, texts, height=None, xfmt=None):
    n = len(labels)
    fig = go.Figure(go.Bar(y=labels[::-1], x=values[::-1], orientation="h", marker=dict(color=colors[::-1]),
                           text=texts[::-1], textposition="outside", cliponaxis=False,
                           textfont=dict(size=11, color="#e2e8f0"), hoverinfo="skip"))
    mx = max([abs(v) for v in values] + [1])
    lo = min(values + [0]); hi = max(values + [0])
    fig.update_xaxes(range=[lo - (mx * 0.28 if lo < 0 else 0), hi + mx * 0.30], showticklabels=False, showgrid=True,
                     gridcolor="rgba(148,163,184,.10)", zeroline=True, zerolinecolor="rgba(148,163,184,.5)", fixedrange=True)
    fig.update_yaxes(fixedrange=True, automargin=True)
    _layout(fig, title, height or 60 + 30 * n, legend=False)
    fig.update_layout(margin=dict(l=6, r=6, t=44, b=10))
    return fig


def compare_chart(title, units, prev_vals, cur_vals, tgt_vals, prev_lab, cur_lab, dec=2, height=320):
    fig = go.Figure()
    fig.add_bar(x=units, y=prev_vals, name=prev_lab, marker=dict(color=C["slate"]), text=[num(v, dec) for v in prev_vals],
                textposition="outside", cliponaxis=False, textfont=dict(size=10, color="#cbd5e1"), hoverinfo="skip")
    fig.add_bar(x=units, y=cur_vals, name=cur_lab, marker=dict(color=C["cyan"], line=dict(color="rgba(255,255,255,.6)", width=1.2)),
                text=[num(v, dec) for v in cur_vals], textposition="outside", cliponaxis=False,
                textfont=dict(size=11, color="#e2e8f0"), hoverinfo="skip")
    if tgt_vals is not None and not all(np.isnan(tgt_vals)):
        fig.add_scatter(x=units, y=list(tgt_vals), mode="markers", name="Target",
                        marker=dict(symbol="line-ew", size=26, line=dict(color=C["amber"], width=3)), hoverinfo="skip")
    allv = list(prev_vals) + list(cur_vals) + [t for t in (tgt_vals if tgt_vals is not None else []) if not np.isnan(t)]
    hi, lo = max(allv + [0]), min(allv + [0])
    pad = (hi - lo) * 0.22 or 1
    fig.update_yaxes(range=[lo - (pad if lo < 0 else 0), hi + pad], showticklabels=False, showgrid=True,
                     gridcolor="rgba(148,163,184,.12)", zeroline=True, zerolinecolor="rgba(148,163,184,.45)", fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    return _layout(fig, title, height, 6)


# ------------------------------------------------------------------ topic specs
# bar = (name, fn, colour[, mp_metric_fn]) ; kpi = (label, fn, target_fn|None, dec, good 'up'|'down'|None, suffix)
def _simple(title, metric, color, dec=0, target=False, tlabel="Target"):
    return dict(title=title, bars=[(title.split(" (")[0], V(metric), C[color])], dec=dec,
                target=(tlabel, T(metric), C["amber"]) if target else None)


SPECS = {
    "T1": dict(
        charts=[dict(title="Cost per line (Lakh Tk)", dec=2,
                     bars=[("With C&Z", V("Cost per line - With C&Z"), C["violet"]), ("Without C&Z", V("Cost per line - Without C&Z"), C["cyan"])],
                     marks=[("C&Z cost", V("C&Z cost per line"), C["pink"])], target=("Target", T("Cost per line - With C&Z"), C["amber"]))],
        kpis=[("Cost/line with C&Z", V("Cost per line - With C&Z"), T("Cost per line - With C&Z"), 2, "down", " L"),
              ("Without C&Z", V("Cost per line - Without C&Z"), None, 2, "down", " L"),
              ("C&Z cost/line", V("C&Z cost per line"), None, 2, "down", " L")]),
    "T2": dict(
        charts=[dict(title=f"MMR {k}", dec=2, mmr=True, bars=[
            ("With C&Z", V(f"MMR {k} - With C&Z"), C["violet"], V(f"MMR {k} MP - With C&Z")),
            ("Without C&Z", V(f"MMR {k} - Without C&Z"), C["cyan"], V(f"MMR {k} MP - Without C&Z")),
            ("C&Z only", V(f"MMR {k} - C&Z only"), C["pink"], V(f"MMR {k} MP - C&Z only"))],
            target=("Target 1.60", T("MMR Authorized - With C&Z"), C["amber"])) for k in ("Authorized", "Held")],
        kpis=[("Held MMR (w/o C&Z)", V("MMR Held - Without C&Z"), T("MMR Held - Without C&Z"), 2, "down", ""),
              ("Held MMR (with C&Z)", V("MMR Held - With C&Z"), T("MMR Held - With C&Z"), 2, "down", ""),
              ("Held − Auth MMR (w/o)", lambda a: a.v("MMR Held - Without C&Z") - a.v("MMR Authorized - Without C&Z"), None, 3, None, ""),
              ("Held − Auth MP (with)", lambda a: a.v("MMR Held MP - With C&Z") - a.v("MMR Authorized MP - With C&Z"), None, 0, None, " MP")]),
    "T3": dict(
        charts=[dict(title="Efficiency (%)", dec=2, bars=[("Forecast", T("Efficiency (Achieved vs Forecast)"), C["slate"]),
                                                          ("Achieved", V("Efficiency (Achieved vs Forecast)"), C["cyan"])])],
        kpis=[("Achieved efficiency", V("Efficiency (Achieved vs Forecast)"), T("Efficiency (Achieved vs Forecast)"), 2, "up", "%"),
              ("Forecast", T("Efficiency (Achieved vs Forecast)"), None, 2, None, "%")]),
    "T4": dict(
        charts=[dict(title="Production (K pcs)", dec=1, bars=[("Production", V("Production"), C["violet"])]),
                dict(title="Line working hours (hrs)", dec=0, bars=[("Line hours", V("Line working hours"), C["cyan"])])],
        kpis=[("Production", V("Production"), None, 1, "up", " K"),
              ("Line hours", V("Line working hours"), None, 0, None, " hr"),
              ("Pcs per line-hr", lambda a: np.divide(a.v("Production") * 1000, a.v("Line working hours"),
                                                      out=np.zeros(12), where=a.v("Line working hours") != 0), None, 1, "up", "")]),
    "T6": dict(
        charts=[dict(title="Cut-to-ship ratio (%)", dec=2, ylo="auto", bars=[("Cut-to-ship", V("Cut-to-ship ratio"), C["violet"])],
                     target=("Target 99%", T("Cut-to-ship ratio"), C["amber"]))],
        kpis=[("Cut-to-ship", V("Cut-to-ship ratio"), T("Cut-to-ship ratio"), 2, "up", "%"),
              ("Total loss", lambda a: np.where(a.v("Cut-to-ship ratio") > 0, 100 - a.v("Cut-to-ship ratio"), 0), None, 2, "down", "%")]),
    "T7": dict(charts=[_simple("Recheck cases", "Recheck cases", "violet")], kpis=[("Recheck cases", V("Recheck cases"), None, 0, "down", "")]),
    "T8": dict(charts=[_simple("Short shipment (pcs)", "Short shipment", "orange", 0, True, "Target 0")],
               kpis=[("Short shipment", V("Short shipment"), T("Short shipment"), 0, "down", " pcs")]),
    "T9": dict(charts=[_simple("Excess shipment (pcs)", "Excess shipment", "green")], kpis=[("Excess shipment", V("Excess shipment"), None, 0, "up", " pcs")]),
    "T10": dict(charts=[_simple("Late delivery rate (%)", "Late delivery rate", "pink", 1, True, "Target 0%")],
                kpis=[("Late delivery", V("Late delivery rate"), None, 1, "down", "%")]),
    "T11": dict(charts=[_simple("Loose thread (cones)", "Loose thread cones", "cyan")], kpis=[("Loose thread", V("Loose thread cones"), None, 0, "down", " cones")]),
    "T12": dict(
        charts=[dict(title="Spare / Needle / Labor cost (K Tk)", dec=0,
                     bars=[("Spare", V("Spare cost"), C["violet"]), ("Needle", V("Needle cost"), C["cyan"]), ("Labor", V("Labor cost"), C["pink"])],
                     targets=[("Spare target", T("Spare cost"), C["violet"]), ("Needle target", T("Needle cost"), C["cyan"]), ("Labor target", T("Labor cost"), C["pink"])])],
        kpis=[("Total cost", SUMV("Spare cost", "Needle cost", "Labor cost"), SUMT("Spare cost", "Needle cost", "Labor cost"), 1, "down", " K"),
              ("Spare", V("Spare cost"), T("Spare cost"), 1, "down", " K"),
              ("Needle", V("Needle cost"), T("Needle cost"), 1, "down", " K"),
              ("Labor", V("Labor cost"), T("Labor cost"), 1, "down", " K")]),
    "T13": dict(
        charts=[dict(title="Profit / Loss (Lakh Tk)", dec=1, bars=[("Production basis", V("Profit/Loss - Production basis"), C["violet"]),
                                                                  ("Shipment basis", V("Profit/Loss - Shipment basis"), C["cyan"])])],
        kpis=[("P/L production basis", V("Profit/Loss - Production basis"), None, 1, "up", " L"),
              ("P/L shipment basis", V("Profit/Loss - Shipment basis"), None, 1, "up", " L")]),
    "T14": dict(charts=[_simple("Jumper operators", "Jumper operators", "blue")], kpis=[("Jumper operators", V("Jumper operators"), None, 0, None, "")]),
}


def t5_arrays(acc):
    return [acc.v(f"Variance - {s}") for s in SECTIONS]


def spec_for(topic):
    code = topic.split()[0]
    if code == "T5":
        return dict(charts=[], kpis=[
            ("Net variance", lambda a: sum(t5_arrays(a)), None, 0, None, " MP"),
            ("Excess (+)", lambda a: sum(np.maximum(x, 0) for x in t5_arrays(a)), None, 0, None, " MP"),
            ("Deficit (−)", lambda a: sum(np.minimum(x, 0) for x in t5_arrays(a)), None, 0, None, " MP")])
    return SPECS[code]


def kpi_values(acc, kpis, sel):
    out = []
    for label, fn, tfn, dec, good, suf in kpis:
        arr = np.asarray(fn(acc), dtype=float)
        val = arr[sel]
        prev = arr[sel - 1] if sel > 0 else None
        tg = None
        if tfn is not None:
            tv = np.asarray(tfn(acc), dtype=float)[sel]
            tg = None if np.isnan(tv) else tv
        out.append(dict(label=label, val=val, prev=prev, tgt=tg, dec=dec, good=good, suf=suf, arr=arr,
                        prev_lab=MONTHS[sel - 1] if sel > 0 else ""))
    return out


def make_figs(acc, spec, sel, span):
    win = window(sel, span)
    figs = []
    for ch in spec["charts"]:
        bars = []
        for b in ch["bars"]:
            name, fn, color = b[0], b[1], b[2]
            arr = np.nan_to_num(np.asarray(fn(acc), dtype=float))
            txt = None
            if len(b) > 3:
                mp = b[3](acc)
                d = 3 if "only" in name else 2
                txt = [f"{arr[i]:.{d}f}<br>({mp[i]:,.0f})" if arr[i] or mp[i] else "0" for i in range(12)]
            bars.append((name, arr, color, txt))
        tgt = ch.get("target")
        tgt = (tgt[0], np.asarray(tgt[1](acc), dtype=float), tgt[2]) if tgt else None
        tgts = [(n, np.asarray(f(acc), dtype=float), c) for n, f, c in ch.get("targets", [])]
        marks = [(n, np.nan_to_num(np.asarray(f(acc), dtype=float)), c) for n, f, c in ch.get("marks", [])]
        ylo = None
        if ch.get("ylo"):
            shown = [x for _, arr, _, _ in bars for x in (arr[i] for i in win) if x > 0]
            ylo = max(0, np.floor(min(shown + [99]) - 4)) if shown else 90
        figs.append(col_chart(ch["title"], win, sel, bars, tgt, marks, ch.get("dec", 2), ylo=ylo, targets=tgts,
                              height=330 if ch.get("mmr") else 310))
    return figs


def t5_fig(acc, sel):
    vals = [float(a[sel]) for a in t5_arrays(acc)]
    cols = [C["green"] if v > 0 else C["red"] if v < 0 else C["slate"] for v in vals]
    texts = [f"{v:+.0f}" if v else "0" for v in vals]
    return hbar(f"Manpower variance by section · {MONTHS[sel]}  (+ excess / − deficit)", SECTIONS, vals, cols, texts)


T6_PARTS = [("Sewing", C["violet"]), ("Wash", C["cyan"]), ("Fabric", C["amber"]), ("Smpl&Other", C["pink"])]


def t6_breakdown(acc, sel):
    """-> (fig or None, info dict). Shows who is responsible for how much of the cut-to-ship loss."""
    ratio = acc.v("Cut-to-ship ratio")[sel]
    parts = [(n, float(acc.v(f"Rejection - {n}")[sel]), c) for n, c in T6_PARTS]
    tot = sum(p[1] for p in parts)
    if ratio <= 0 and tot <= 0:
        return None, None
    loss = 100 - ratio if ratio > 0 else tot
    allowed = 100 - (np.nanmax(acc.t("Cut-to-ship ratio")) if not np.all(np.isnan(acc.t("Cut-to-ship ratio"))) else 99)
    texts = [f"{v:.2f}%  ·  {v / tot * 100:.0f}% of loss" if tot else f"{v:.2f}%" for _, v, _ in parts]
    fig = hbar(f"Who is responsible · {MONTHS[sel]} loss {loss:.2f}% of cut", [p[0] for p in parts], [p[1] for p in parts],
               [p[2] for p in parts], texts, height=230)
    return fig, dict(loss=loss, allowed=allowed, over=loss - allowed, total_parts=tot, parts=parts)
