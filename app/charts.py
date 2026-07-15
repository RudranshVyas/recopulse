"""Server-side Plotly figures, themed to the RecoPulse design system."""
import plotly.graph_objects as go
import plotly.io as pio

INK = "#0A0B0D"
SURFACE = "#131519"
BORDER = "#232830"
PAPER = "#E8E4DC"
MUTED = "#7C838F"
SIGNAL = "#C2F53E"
WARM = "#FF6B35"
COOL = "#4A9DFF"
SERIES = [SIGNAL, "#FF6B35", "#4A9DFF", "#E8E4DC", "#8B7FD4", "#3FBFA0",
          "#D4A15E", "#C2557A", "#5FA8D3", "#9AA37C"]

FONT = "IBM Plex Mono, ui-monospace, monospace"

BASE = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, size=11, color=MUTED),
    colorway=SERIES,
    margin=dict(l=52, r=18, t=18, b=44),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
               tickfont=dict(size=10, color=MUTED), automargin=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
               tickfont=dict(size=10, color=MUTED), automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED),
                orientation="h", y=-0.22, x=0),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=SIGNAL,
                    font=dict(family=FONT, size=11, color=PAPER)),
))
pio.templates["recopulse"] = BASE
pio.templates.default = "recopulse"

CONFIG = {"displayModeBar": False, "responsive": True}


def render(fig, height=300):
    fig.update_layout(height=height)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                       default_height=height)


def line(x, y, name="", fill=True, color=SIGNAL):
    fig = go.Figure(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=2, shape="spline", smoothing=0.6),
        fill="tozeroy" if fill else None,
        fillcolor="rgba(194,245,62,0.08)" if fill and color == SIGNAL else None,
    ))
    return fig


def bar(x, y, horizontal=False, color=SIGNAL, text=None):
    if horizontal:
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h", marker_color=color,
                               text=text, textposition="auto",
                               textfont=dict(family=FONT, size=10, color=INK)))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color, text=text,
                               textposition="auto",
                               textfont=dict(family=FONT, size=10, color=INK)))
    fig.update_layout(bargap=0.45)
    return fig


def grouped_bar(x, series):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_bar(x=x, y=vals, name=name, marker_color=SERIES[i % len(SERIES)])
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    return fig


def scatter(x, y, text, size=None, color=None):
    fig = go.Figure(go.Scatter(
        x=x, y=y, text=text, mode="markers",
        marker=dict(size=size or 9, color=color or SIGNAL, opacity=0.72,
                    line=dict(width=0.5, color=INK),
                    colorscale=[[0, COOL], [0.5, SIGNAL], [1, WARM]] if color is not None else None),
        hovertemplate="%{text}<br>%{x:,} views · %{y:.2%} of them led to a sale<extra></extra>",
    ))
    return fig


def donut(labels, values):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.68,
        marker=dict(colors=SERIES, line=dict(color=INK, width=2)),
        textinfo="percent", textfont=dict(family=FONT, size=11, color=INK),
    ))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def heatmap_cm(cm):
    text = [[f"{v:,}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["said: won't buy", "said: will buy"],
        y=["really browsed", "really bought"],
        text=text, texttemplate="%{text}",
        textfont=dict(family=FONT, size=18),
        colorscale=[[0, SURFACE], [0.5, "#5A7A1F"], [1, SIGNAL]],
        showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y} · %{x}<br>%{z:,} visits<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=110, r=18, t=18, b=44))
    return fig


def roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random guessing",
                    line=dict(color=BORDER, width=1, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name="RecoPulse",
                    line=dict(color=SIGNAL, width=2.5),
                    fill="tozeroy", fillcolor="rgba(194,245,62,0.07)",
                    hovertemplate="catches %{y:.0%} of buyers<br>"
                                  "false alarms on %{x:.0%} of browsers<extra></extra>")
    fig.update_layout(xaxis_title="share of browsers wrongly flagged",
                      yaxis_title="share of real buyers caught")
    return fig


def elbow(sweep, chosen, rejected):
    ks = [s["k"] for s in sweep]
    fig = go.Figure()
    fig.add_scatter(x=ks, y=[s["silhouette"] for s in sweep], name="how clean the split is",
                    mode="lines+markers",
                    line=dict(color=SIGNAL, width=2.5), marker=dict(size=7),
                    hovertemplate="%{x} groups · score %{y:.3f}<extra></extra>")
    fig.add_vline(x=chosen, line=dict(color=SIGNAL, width=1, dash="dash"))
    fig.add_vline(x=rejected, line=dict(color=WARM, width=1, dash="dot"))
    fig.update_layout(
        xaxis_title="number of shopper groups",
        yaxis=dict(title="how cleanly they separate", gridcolor=BORDER),
    )
    return fig


def radar(categories, series):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]], name=name,
            fill="toself", opacity=0.55,
            line=dict(color=SERIES[i % len(SERIES)], width=2),
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, gridcolor=BORDER, linecolor=BORDER,
                                   tickfont=dict(size=8, color=MUTED)),
                   angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                                    tickfont=dict(size=9, color=MUTED))),
        margin=dict(l=60, r=60, t=30, b=60),
    )
    return fig
