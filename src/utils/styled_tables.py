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
]

TABLE_STYLE_WITH_CAPTION = TABLE_STYLE_BASE + [
    {"selector": "caption",
     "props": [("font-size", "13px"), ("font-weight", "bold"), ("text-align", "left"),
               ("padding-bottom", "10px"), ("color", "#111827")]},
]

TABLE_STYLE_WIDE_DATE = TABLE_STYLE_BASE + [
    {"selector": "td:nth-child(2)",
     "props": [("text-align", "left"), ("min-width", "130px")]},
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
        style = TABLE_STYLE_WITH_CAPTION
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
    style: Optional[TableStyle] = None
) -> Styler:
    if style is None:
        style = TABLE_STYLE_BASE

    preview = dataframe.copy().head(10)
    for column in preview.select_dtypes("float").columns:
        preview[column] = preview[column].astype("int64")
    if "date_axis" in preview.columns:
        preview["date_axis"] = preview["date_axis"].dt.strftime("%Y-%m-%d")

    if caption is None:
        total_rows = len(dataframe)
        shown = min(10, total_rows)
        caption = f"Data Preview — first {shown} of {total_rows} rows, {len(preview.columns)} columns"

    return preview.style.set_caption(caption).set_table_styles(style)  # type: ignore[arg-type]
