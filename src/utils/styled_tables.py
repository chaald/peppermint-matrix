import pandas as pd

from pandas.io.formats.style import Styler
from IPython.display import HTML, display
from typing import Callable, Dict, List, Optional, Set, Tuple

DEFAULT_CURRENCY_METRICS: Set[str] = {"average_order_value", "average_user_value", "gross_market_value"}
DEFAULT_INTEGER_METRICS: Set[str] = {"date_count"}

TableStyle = List[Dict[str, object]]

TABLE_STYLE_BASE = [
    {"selector": "thead th",
     "props": [("background-color", "#E2E8F0"), ("color", "#111827"),
               ("font-weight", "700"), ("text-align", "center"), ("padding", "7px 14px"),
               ("border-bottom", "2px solid #CBD5E1")]},
    {"selector": "tbody th",
     "props": [("background-color", "#F1F5F9"), ("color", "#111827"),
               ("font-weight", "600"), ("text-align", "left"), ("padding", "6px 14px"),
               ("border-right", "1px solid #CBD5E1")]},
    {"selector": "td",
     "props": [("color", "#111827"), ("text-align", "right"),
               ("padding", "6px 14px"), ("font-family", "monospace, monospace")]},
    {"selector": "tr:nth-child(even) td",
     "props": [("background-color", "#F8FAFC")]},
    {"selector": "tr:nth-child(odd) td",
     "props": [("background-color", "#FFFFFF")]},
    {"selector": "tr:hover td",
     "props": [("box-shadow", "inset 0 0 0 1000px rgba(0, 0, 0, 0.06)")]},
    {"selector": "tr:hover th",
     "props": [("box-shadow", "inset 0 0 0 1000px rgba(0, 0, 0, 0.06)")]},
    {"selector": "caption",
     "props": [("font-size", "16px"), ("font-weight", "700"), ("text-align", "left"),
               ("padding-bottom", "10px"), ("color", "#1E293B"), ("background-color", "#FFFFFF")]},
    {"selector": "td:nth-child(2)",
     "props": [("text-align", "left"), ("min-width", "130px")]},
]

TABLE_STYLE_DENSITY = [
    d for d in TABLE_STYLE_BASE
    if d["selector"] not in ["tr:nth-child(even) td", "tr:nth-child(odd) td"]
]


def format_value(
    value: float, 
    metric: str,
    currency_metrics: Optional[Set[str]] = None,
    integer_metrics: Optional[Set[str]] = None
) -> str:
    if currency_metrics is None:
        currency_metrics = DEFAULT_CURRENCY_METRICS
    if integer_metrics is None:
        integer_metrics = DEFAULT_INTEGER_METRICS
    if metric in integer_metrics:
        return f"{int(value):,}"
    if metric in currency_metrics:
        return f"Rp {value:,.0f}"
    return f"{value:.5f}"


def format_delta(
    value: float, 
    metric: str,
    currency_metrics: Optional[Set[str]] = None,
    integer_metrics: Optional[Set[str]] = None
) -> str:
    if currency_metrics is None:
        currency_metrics = DEFAULT_CURRENCY_METRICS
    if integer_metrics is None:
        integer_metrics = DEFAULT_INTEGER_METRICS
    sign = "+" if value >= 0 else ""
    if metric in integer_metrics:
        return f"{sign}{int(value):,}"
    if metric in currency_metrics:
        return f"{sign}Rp {value:,.0f}"
    return f"{sign}{value:.5f}"


def format_delta_percent(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def make_delta_colors(
    pivot: pd.DataFrame, 
    metric_order: List[str], 
    column: str,
    green_when_zero_metrics: Optional[Set[str]] = None
) -> Callable:
    if green_when_zero_metrics is None:
        green_when_zero_metrics = DEFAULT_INTEGER_METRICS
    def colorize(col: pd.Series) -> List[str]:
        colors: List[str] = []
        for metric in metric_order:
            val = pivot.loc[metric, column]
            if (metric in green_when_zero_metrics and val == 0) or val > 0:
                colors.append("background-color: #DCFCE7 !important; color: #14532D !important")
            elif val < 0:
                colors.append("background-color: #FEE2E2 !important; color: #7F1D1D !important")
            else:
                colors.append("")
        return colors
    return colorize


def styled_metrics_pivot(
    pivot: pd.DataFrame, 
    caption: str,
    column_order: List[str], 
    metric_order: List[str],
    currency_metrics: Optional[Set[str]] = None,
    integer_metrics: Optional[Set[str]] = None,
    style: Optional[TableStyle] = None
) -> Styler:
    if style is None:
        style = TABLE_STYLE_BASE
    if currency_metrics is None:
        currency_metrics = DEFAULT_CURRENCY_METRICS
    if integer_metrics is None:
        integer_metrics = DEFAULT_INTEGER_METRICS

    display_pivot = pd.DataFrame(index=metric_order)
    for column in column_order:
        display_pivot[column] = [format_value(pivot.loc[m, column], m, currency_metrics, integer_metrics)
                                 for m in metric_order]

    if "delta" in pivot.columns:
        display_pivot["delta"] = [format_delta(pivot.loc[m, "delta"], m, currency_metrics, integer_metrics)
                                  for m in metric_order]
    if "delta_percent" in pivot.columns:
        display_pivot["delta_percent"] = [format_delta_percent(pivot.loc[m, "delta_percent"])
                                          for m in metric_order]

    styler = display_pivot.style.set_caption(caption).set_table_styles(style)  # type: ignore[arg-type]

    if "delta" in pivot.columns:
        styler = styler.apply(make_delta_colors(pivot, metric_order, "delta", integer_metrics),
                              subset=["delta"])
    if "delta_percent" in pivot.columns:
        styler = styler.apply(make_delta_colors(pivot, metric_order, "delta_percent", integer_metrics),
                              subset=["delta_percent"])

    return styler


def merged_tables_html(tables_with_headings: List[Tuple[str, Styler]]) -> None:
    html_parts: List[str] = []
    for index, (heading_text, styler) in enumerate(tables_with_headings):
        margin = "16px" if index == 0 else "20px"
        if heading_text:
            html_parts.append(f"<h2 style='margin-top: {margin}; margin-bottom: 8px;'>{heading_text}</h2>")
        html_parts.append(styler.to_html())
    display(HTML("".join(html_parts)))


def data_preview_styled(
    dataframe: pd.DataFrame,
    caption: Optional[str] = None,
    style: Optional[TableStyle] = None,
    max_rows: int = 10,
) -> Styler:
    if style is None:
        style = TABLE_STYLE_BASE

    preview = dataframe.copy().head(max_rows)
    if "date_axis" in preview.columns:
        preview["date_axis"] = preview["date_axis"].dt.strftime("%Y-%m-%d")

    if caption is None:
        total_rows = len(dataframe)
        shown = min(max_rows, total_rows)
        caption = f"Data Preview — first {shown} of {total_rows} rows, {len(preview.columns)} columns"

    styler = preview.style.set_caption(caption).set_table_styles(style)  # type: ignore[arg-type]
    float_cols = preview.select_dtypes("float").columns
    if len(float_cols):
        styler = styler.format(subset=float_cols, formatter="{:.6g}")
    return styler


def _make_gradient(cmap_name: str) -> Callable:
    """Factory returning a styler.apply-compatible function with inline bg colors."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)

    def _apply(s: pd.Series) -> List[str]:
        values = s.values.astype(float)
        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        normed = (values - vmin) / (vmax - vmin)
        colors = cmap(normed)
        return [
            "background-color: rgba(%d,%d,%d,1)" % (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in colors
        ]
    return _apply


def style_run_density(
    dataframe: pd.DataFrame,
    caption: Optional[str] = None,
    style: Optional[TableStyle] = None,
    max_rows: int = 10,
) -> Styler:
    styler = data_preview_styled(dataframe, caption=caption, style=style or TABLE_STYLE_DENSITY, max_rows=max_rows)

    # Alternating rows via apply (same css specificity as gradients below)
    def _alt_rows(df: pd.DataFrame) -> pd.DataFrame:
        even = "background-color: #F8FAFC"
        odd = "background-color: #FFFFFF"
        return pd.DataFrame(
            [[even if i % 2 == 0 else odd for _ in range(df.shape[1])] for i in range(df.shape[0])],
            index=df.index,
            columns=df.columns,
        )

    styler = styler.apply(_alt_rows, axis=None)

    # Gradient overlays — come AFTER alt-rows in the stylesheet so they win on overlapping cells
    colormap_pairs = [
        ("Greens", ["n_runs", "mean_score", "min_score", "max_score"]),
        ("Reds", ["std_score"]),
    ]
    for cmap_name, col_names in colormap_pairs:
        cols = [c for c in col_names if c in dataframe.columns]
        if cols:
            styler = styler.apply(_make_gradient(cmap_name), subset=cols)

    return styler
