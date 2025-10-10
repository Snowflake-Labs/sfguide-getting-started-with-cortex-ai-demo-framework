import streamlit as st
import pandas as pd
import yaml
import altair as alt
import json
import os
from typing import Dict, Any, List

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except Exception:
    session = None


st.set_page_config(page_title="Snow Visualizer", layout="wide")

# Try alternative background approach for Snowflake
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .main > div {
        padding-top: 2rem;
        background-color: rgba(240, 248, 255, 0.8);
    }
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(240, 248, 255, 0.3);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Style constants and global CSS
PALE_BLUE = "#93c5fd"  # border highlight
LIGHT_BORDER = "#e5e7eb"
SELECTED_BG = "#eef5ff"  # bluish highlight background

st.markdown(
    f"""
    <style>
      /* Enhanced card styling */
      .sv-card {{
        border: 1px solid {LIGHT_BORDER};
        border-radius: 10px;
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.95);
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(41, 181, 232, 0.1);
      }}
      .sv-card.selected {{
        border-color: {PALE_BLUE};
        background: {SELECTED_BG};
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.2);
      }}
      /* Enhanced button styling */
      .stButton > button {{
        border: 1px solid {PALE_BLUE} !important;
        background: rgba(255, 255, 255, 0.95) !important;
        color: #111827 !important;
        box-shadow: 0 2px 4px rgba(41, 181, 232, 0.1) !important;
      }}
      .stButton > button:hover {{
        background: rgba(41, 181, 232, 0.1) !important;
        box-shadow: 0 4px 8px rgba(41, 181, 232, 0.2) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================
# Stage helpers (Project/YAML)
# =============================

# Get the current database from the Snowflake session context
def get_current_database():
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        result = session.sql("SELECT CURRENT_DATABASE()").collect()
        if result and result[0][0]:
            return result[0][0]
        else:
            raise Exception("Could not determine current database from session")
    except Exception as e:
        raise Exception(f"Failed to get current database: {str(e)}. Ensure DataOps deployment completed successfully.")

# DataOps database configuration - use current database context
db_name = get_current_database()

STAGE_BASE = f"@{db_name}.CONFIGS.VISUALIZATION_YAML_STAGE"
YAML_FORMAT = f"{db_name}.CONFIGS.YAML_CSV_FORMAT"

# Lightweight caches to avoid stage listing/query churn on UI interactions
@st.cache_data(ttl=300, show_spinner=False)
def _cached_list(stage_path: str) -> pd.DataFrame:
    try:
        return session.sql(f"LIST {stage_path}").to_pandas()
    except Exception:
        return pd.DataFrame()


def list_projects_in_stage() -> List[str]:
    if session is None:
        return []
    files_df = _cached_list(STAGE_BASE)
    if files_df.empty:
        return []
    file_col = files_df.columns[0]
    prefix = "VISUALIZATION_YAML_STAGE/"
    paths = [row[file_col] for _, row in files_df.iterrows()]
    # Keep after the stage root
    trimmed = []
    for p in paths:
        idx = p.lower().find(prefix.lower())
        if idx >= 0:
            trimmed.append(p[idx + len(prefix):])
    projects = sorted({p.split('/')[0] for p in trimmed if '/' in p})
    return projects


def list_yaml_files_in_project(project: str) -> List[str]:
    if session is None or not project:
        return []
    df = _cached_list(f"{STAGE_BASE}/{project}")
    if df.empty:
        return []
    file_col = df.columns[0]
    files = [row[file_col] for _, row in df.iterrows()]
    return [f for f in files if f.lower().endswith((".yaml", ".yml"))]


def load_yaml_from_stage(project: str, filename: str) -> Dict[str, Any]:
    if session is None:
        raise RuntimeError("No Snowflake session; cannot read from stage.")
    # Defensive: remove duplicated stage leaf if present in filename
    leaf = STAGE_BASE.replace("@", "").split(".")[-1]
    # filename might include full 'VISUALIZATION_YAML_STAGE/...' – strip it
    if filename.lower().startswith(f"{leaf.lower()}/"):
        filename = filename[len(leaf) + 1:]
    full_path = f"{STAGE_BASE}/{project}/{filename.split('/')[-1]}"
    df = session.sql(
        f"SELECT $1 FROM {full_path} (file_format => '{YAML_FORMAT}')"
    ).to_pandas()
    if df.empty:
        raise RuntimeError(f"YAML file empty or not found: {full_path}")
    content = "\n".join([row for row in df.iloc[:, 0].tolist() if row is not None])
    return yaml.safe_load(content)


def sql_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    db = cfg["app"]["data_source"]["database"]
    sc = cfg["app"]["data_source"]["schema"]
    tb = cfg["app"]["data_source"]["table"]
    # Support time config at root or under app
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})
    time_col = time_cfg.get("column")
    if not time_col:
        raise ValueError("YAML missing time.column; add under root 'time' or 'app.time'.")
    macros = cfg.get("sql_macros", {})

    return {
        "DB": db,
        "SCHEMA": sc,
        "TABLE": tb,
        "TIME_COL": time_col,
        "base_from": f"FROM {db}.{sc}.{tb}",
        "time_col": time_col,
        "filter_time_window": macros.get("filter_time_window", {}),
    }


def resolve_time_filter(macros: Dict[str, Any], window_key: str, time_col: str) -> str:
    template = macros.get("filter_time_window", {}).get(window_key, "")
    return template.replace("{TIME_COL}", time_col)


def run_sql(sql: str) -> pd.DataFrame:
    if session is None:
        st.error("No active Snowflake session. Run this in Snowflake Streamlit.")
        return pd.DataFrame()
    return session.sql(sql).to_pandas()


def build_metric_sql(metric_sql_expr: str, base_from: str, where_clause: str) -> str:
    where = f"WHERE {where_clause}" if where_clause else ""
    return f"SELECT {metric_sql_expr} AS METRIC {base_from} {where}"


def compute_period_value(metric_expr: str, base_from: str, time_col: str, grain: str, offset: int) -> str:
    if grain.lower() == "year":
        current_clause = f"YEAR({time_col}) = YEAR(DATEADD(year, {offset}, CURRENT_DATE()))"
    else:
        current_clause = (
            f"DATE_TRUNC('{grain}', {time_col}) = DATE_TRUNC('{grain}', DATEADD({grain}, {offset}, CURRENT_DATE()))"
        )
    return build_metric_sql(metric_expr, base_from, current_clause)


def compute_mom_yoy(metric_expr: str, base_from: str, time_col: str) -> Dict[str, Any]:
    # MoM (current month vs previous month)
    mom_curr_sql = compute_period_value(metric_expr, base_from, time_col, "month", 0)
    mom_prev_sql = compute_period_value(metric_expr, base_from, time_col, "month", -1)
    # YoY (current year vs previous year)
    yoy_curr_sql = compute_period_value(metric_expr, base_from, time_col, "year", 0)
    yoy_prev_sql = compute_period_value(metric_expr, base_from, time_col, "year", -1)

    mom_curr = run_sql(mom_curr_sql)
    mom_prev = run_sql(mom_prev_sql)
    yoy_curr = run_sql(yoy_curr_sql)
    yoy_prev = run_sql(yoy_prev_sql)

    def _delta(curr_df: pd.DataFrame, prev_df: pd.DataFrame) -> Dict[str, Any]:
        curr = float(curr_df.iloc[0, 0]) if not curr_df.empty else 0.0
        prev = float(prev_df.iloc[0, 0]) if not prev_df.empty else 0.0
        abs_change = curr - prev
        pct_change = (abs_change / prev) * 100 if prev != 0 else 0.0
        return {"current": curr, "previous": prev, "abs": abs_change, "pct": pct_change}

    return {
        "mom": _delta(mom_curr, mom_prev),
        "yoy": _delta(yoy_curr, yoy_prev),
    }


def render_overview(cfg: Dict[str, Any]):
    # Section header - ULTRA COMPRESSED
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0f9ff, #e0f2fe); 
                padding: 4px 10px; border-radius: 6px; margin: 5px 0 3px 0; 
                border-left: 3px solid #0ea5e9; border: 1px solid #bfdbfe;'>
        <h2 style='color: #1e40af; margin: 0; font-weight: 600; font-size: 16px;'>📊 Overview Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls section - ULTRA COMPRESSED
    st.markdown("""
    <div style='background: #f8fafc; padding: 6px 8px; border-radius: 4px; 
                border: 1px solid #e2e8f0; margin-bottom: 6px;'>
        <h4 style='color: #374151; margin: 0 0 4px 0; font-size: 12px;'>⚙️ Time Controls</h4>
    """, unsafe_allow_html=True)
    maps = sql_from_config(cfg)
    base_from = maps["base_from"]
    time_col = maps["time_col"]

    # Controls (side-by-side to save space)
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})
    windows = [time_cfg.get("default_window", "last_12_months")] + [
        w for w in (time_cfg.get("windows", []) or []) if w != time_cfg.get("default_window")
    ]
    # Allow 'LTD' as a time window (no WHERE clause)
    if not any(str(w).lower() == "ltd" for w in windows):
        windows.append("LTD")
    grains = list(time_cfg.get("supported_grains", ["month"]))
    if "ltd" not in [g.lower() for g in grains]:
        grains.append("LTD")
    default_grain = time_cfg.get("default_grain", "month")
    default_index = next((i for i, g in enumerate(grains) if str(g).lower() == default_grain.lower()), 0)

    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        window = st.selectbox("Time Window", options=windows, index=0)
    with filt_col2:
        grain = st.selectbox("Time Grain (timeseries)", options=grains, index=default_index)
    
    # Close controls container
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics section - ULTRA COMPRESSED
    st.markdown("""
    <div style='background: white; padding: 4px 6px; border-radius: 4px; 
                border: 1px solid #e2e8f0; margin-bottom: 4px;'>
        <h4 style='color: #374151; margin: 0 0 4px 0; font-size: 12px;'>📈 Key Metrics</h4>
    """, unsafe_allow_html=True)

    # LTD window => no WHERE clause
    if str(window).lower() == "ltd":
        where_time = ""
    else:
        where_time = resolve_time_filter(cfg.get("sql_macros", {}), window, time_col)

    # Metric cards (max 6)
    metrics_meta = {m["key"]: m for m in cfg.get("metrics", [])}
    card_keys = cfg.get("cards", {}).get("default_metrics", [])[:6]

    cols = st.columns(len(card_keys) or 1)

    selected_metric_key = st.session_state.get("selected_metric_key", card_keys[0] if card_keys else None)

    for i, key in enumerate(card_keys):
        meta = metrics_meta[key]
        expr = meta["sql"]
        current_sql = build_metric_sql(expr, base_from, where_time)
        current_val_df = run_sql(current_sql)
        current_val = float(current_val_df.iloc[0, 0]) if not current_val_df.empty else 0.0

        deltas = compute_mom_yoy(expr, base_from, time_col)
        mom_delta = deltas["mom"]["pct"]
        yoy_delta = deltas["yoy"]["pct"]

        with cols[i]:
            clicked = st.button(meta["label"], key=f"card_{key}")
            is_selected = (selected_metric_key == key)
            border_color = PALE_BLUE if is_selected else LIGHT_BORDER
            background = SELECTED_BG if is_selected else "#ffffff"
            mom_color = "#16a34a" if mom_delta > 0 else ("#dc2626" if mom_delta < 0 else "#6b7280")
            yoy_color = "#16a34a" if yoy_delta > 0 else ("#dc2626" if yoy_delta < 0 else "#6b7280")
            card_html = f"""
            <div style="border:1px solid {border_color}; border-radius:6px; padding:6px; 
                        background: linear-gradient(135deg, {background}, rgba(240, 248, 255, 0.9)); 
                        margin-bottom:4px; box-shadow: 0 1px 3px rgba(14, 165, 233, 0.1);
                        transition: all 0.2s ease;">
              <div style='margin-bottom: 2px;'>
                <h4 style='margin: 0; color: #1e40af; font-size: 10px; font-weight: 600;'>{meta["label"]}</h4>
              </div>
              <div style='display:flex;justify-content:space-between;align-items:flex-end;'>
                <div style='font-size:20px;font-weight:700;line-height:1;color:#0f172a;'>{current_val:,.2f}</div>
                <div style='text-align:right;font-size:9px;'>
                  <div style="color:{mom_color};font-weight:600;">MoM {mom_delta:+.1f}%</div>
                  <div style="color:{yoy_color};font-weight:600;">YoY {yoy_delta:+.1f}%</div>
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            if clicked:
                selected_metric_key = key

    if selected_metric_key:
        st.session_state["selected_metric_key"] = selected_metric_key
    
    # Close metrics container
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts section - ULTRA COMPRESSED
    st.markdown("""
    <div style='background: white; padding: 4px 6px; border-radius: 4px; 
                border: 1px solid #e2e8f0; margin: 4px 0;'>
        <h4 style='color: #374151; margin: 0 0 4px 0; font-size: 12px;'>📊 Data Visualizations</h4>
    """, unsafe_allow_html=True)

    # Side-by-side: Time Series (left) and Ranked Grid (right) with breathing room
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # Time Series - ULTRA COMPRESSED
        st.markdown("""
        <div style='background: #f8fafc; padding: 3px 6px; border-radius: 3px; 
                    border: 1px solid #e2e8f0; margin-bottom: 4px;'>
            <h5 style='color: #374151; margin: 0; font-size: 11px;'>📈 Time Series</h5>
        </div>
        """, unsafe_allow_html=True)
        if selected_metric_key:
            metric_expr = metrics_meta[selected_metric_key]["sql"]
            if str(grain).lower() == "ltd":
                sql = f"""
                WITH base AS (
                  SELECT DATE_TRUNC('month', {time_col}) AS PERIOD, {metric_expr} AS VALUE
                  {base_from}
                  {('WHERE ' + where_time) if where_time else ''}
                  GROUP BY 1
                )
                SELECT PERIOD,
                       SUM(VALUE) OVER (ORDER BY PERIOD ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS VALUE
                FROM base
                ORDER BY PERIOD
                """
            else:
                sql = f"""
                SELECT
                  DATE_TRUNC('{grain}', {time_col}) AS PERIOD,
                  {metric_expr} AS VALUE
                {base_from}
                {('WHERE ' + where_time) if where_time else ''}
                GROUP BY 1
                ORDER BY 1
                """
            df = run_sql(sql)
            if not df.empty:
                c = alt.Chart(df).mark_line(point=True).encode(
                    x="PERIOD:T",
                    y="VALUE:Q",
                    tooltip=["PERIOD:T", "VALUE:Q"],
                ).properties(height=350)
                st.altair_chart(c, use_container_width=True)
            else:
                st.info("No data returned for the selected configuration/time window.")

    with col_right:
        # Ranked Grid - ULTRA COMPRESSED
        st.markdown("""
        <div style='background: #f8fafc; padding: 3px 6px; border-radius: 3px; 
                    border: 1px solid #e2e8f0; margin-bottom: 4px;'>
            <h5 style='color: #374151; margin: 0; font-size: 11px;'>🏆 Ranked Grid</h5>
        </div>
        """, unsafe_allow_html=True)
        dim_defs = {d["key"]: d for d in cfg.get("dimensions", [])}
        grid_dims = cfg.get("tabs", [])[0].get("grid", {}).get("dimension_selector", [])
        grid_dim_key = st.selectbox("Dimension", options=grid_dims, format_func=lambda k: dim_defs[k]["label"]) if grid_dims else None

        if grid_dim_key and selected_metric_key:
            dim_col = dim_defs[grid_dim_key].get("column") or dim_defs[grid_dim_key].get("expression")
            metric_expr = metrics_meta[selected_metric_key]["sql"]
            sql = f"""
            SELECT
              {dim_col} AS DIMENSION,
              {metric_expr} AS VALUE
            {base_from}
            {('WHERE ' + where_time) if where_time else ''}
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 200
            """
            df = run_sql(sql)
            # Top 10 with fixed height + scroll
            df_top = df.sort_values("VALUE", ascending=False).head(10) if not df.empty else df
            st.dataframe(df_top, use_container_width=True, height=350)
    
    # Close charts container
    st.markdown('</div>', unsafe_allow_html=True)



def main():
    # CSS for clear visual boundaries and interactive elements
    st.markdown("""
    <style>
    /* Main container - ULTRA COMPRESSED */
    .main .block-container {
        background: #fafbfc !important;
        padding: 8px !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }
    
    /* Make sections visually distinct */
    .stSelectbox > div > div {
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        background: white !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    /* Make buttons more interactive looking */
    .stButton > button {
        border: 2px solid #3b82f6 !important;
        border-radius: 8px !important;
        background: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #eff6ff !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Visual separation for dataframes */
    .stDataFrame {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Make number inputs and text areas more defined */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    
    /* Add subtle background to radio buttons for grouping */
    .stRadio > div {
        background: #f8fafc !important;
        padding: 8px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # (Title rendered after sidebar once we know the selected folder/name)

    # Sidebar – choose project and YAML from stage, with local file fallback
    title_suffix = None
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f0f9ff, #e0f2fe); 
                    padding: 4px 8px; border-radius: 4px; margin: 4px 0;'>
            <h4 style='color: #1e40af; margin: 0; font-size: 12px;'>⚙️ Configuration Source</h4>
        </div>
        """, unsafe_allow_html=True)
        source = st.radio("Load from", ["Stage", "Local file"], index=0, horizontal=True)

        cfg = None
        if source == "Stage":
            projects = list_projects_in_stage()
            if not projects:
                st.warning(f"No projects found in @{db_name}.CONFIGS.VISUALIZATION_YAML_STAGE")
            project = st.selectbox("Project", options=["Select..."] + projects, index=0)
            selected_yaml = None
            if project != "Select...":
                files = list_yaml_files_in_project(project)
                display_files = [f.split("/")[-1] for f in files]
                idx = st.selectbox("YAML File", options=range(len(display_files)), format_func=lambda i: display_files[i] if display_files else "", disabled=not display_files)
                if files:
                    selected_yaml = files[idx]
                # Use the selected project folder as the title suffix when available
                title_suffix = project
            if project != "Select..." and selected_yaml:
                try:
                    cfg = load_yaml_from_stage(project, selected_yaml)
                except Exception as e:
                    st.error(f"Failed to load from stage: {e}")
        else:
            local_path = st.text_input("Local YAML path", value="snow_visualizer_qualtrics.yaml")
            if st.button("Load Local YAML"):
                st.session_state["local_yaml_path"] = local_path
            local_to_use = st.session_state.get("local_yaml_path", local_path)
            try:
                cfg = load_yaml_config(local_to_use)
            except Exception as e:
                st.error(f"Failed to load local YAML: {e}")
            # Derive a friendly name from local path: prefer parent folder, else base name
            lp = (local_to_use or "").strip().rstrip("/")
            parts = lp.split("/")
            if len(parts) >= 2 and parts[-2]:
                title_suffix = parts[-2]
            else:
                base = parts[-1] if parts else ""
                title_suffix = base.rsplit(".", 1)[0] if "." in base else (base or "Local")

    # Title - ULTRA COMPRESSED (now dynamic based on selected project/folder)
    effective_title = f"❄️ SnowVisualizer – {title_suffix}" if title_suffix else "❄️ SnowVisualizer"
    st.markdown(f"""
    <div style='background: linear-gradient(to right, #1e40af, #0ea5e9); 
                padding: 6px 10px; border-radius: 6px; margin-bottom: 8px;
                box-shadow: 0 1px 4px rgba(14, 165, 233, 0.3);'>
        <h1 style='color: white; margin: 0; text-align: center; font-weight: 700; font-size: 18px;'>{effective_title}</h1>
    </div>
    """, unsafe_allow_html=True)

    if not cfg:
        st.info("Select a project/YAML (or load a local file) to start.")
        return

    tabs = cfg.get("tabs", [])
    if not tabs:
        st.error("No tabs defined in YAML.")
        return

    tab_titles = [t.get("title", t.get("key")) for t in tabs]
    
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f0f9ff, #e0f2fe); 
                    padding: 4px 8px; border-radius: 4px; margin: 6px 0;'>
            <h4 style='color: #1e40af; margin: 0; font-size: 12px;'>📊 Navigation</h4>
        </div>
        """, unsafe_allow_html=True)
    
    selected_title = st.sidebar.radio("Select Page", tab_titles, index=0, key="app_active_page")

    # Render only the selected page to avoid re-running heavy queries on UI tweaks
    selected_cfg = next((t for t in tabs if t.get("title", t.get("key")) == selected_title), tabs[0])
    ttype = selected_cfg.get("type")
    if ttype == "overview":
        render_overview(cfg)
    elif ttype == "product":
        render_product_tab(cfg, selected_cfg)
    elif ttype == "self_service":
        render_self_service_tab(cfg, selected_cfg)
    elif ttype == "compare" or ttype == "vs":
        render_compare_tab(cfg, selected_cfg)
    elif ttype == "topn":
        render_topn_tab(cfg, selected_cfg)
    elif ttype == "search":
        render_search_tab(cfg, selected_cfg)
    elif ttype == "analyst":
        render_analyst_tab(cfg, selected_cfg)
    else:
        st.info("This tab type is not implemented yet.")


def render_product_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    # Controls
    dim_defs = {d["key"]: d for d in cfg.get("dimensions", [])}
    metrics_meta = {m["key"]: m for m in cfg.get("metrics", [])}

    selectable_dims = [d for d in tab_cfg.get("entity_dimensions", []) if d in dim_defs]
    metrics_allowed = [m for m in tab_cfg.get("metrics_allowed", []) if m in metrics_meta]

    if not selectable_dims or not metrics_allowed:
        st.warning("Product tab is not configured with valid dimensions/metrics.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_dim_key = st.selectbox("Dimension", options=selectable_dims, format_func=lambda k: dim_defs[k]["label"], key="product_dim") 
    with c2:
        secondary_options = [d for d in selectable_dims if d != chosen_dim_key] or selectable_dims
        other_dim_key = st.selectbox("Secondary Dimension", options=secondary_options, format_func=lambda k: dim_defs[k]["label"], key="product_dim2")
    with c3:
        chosen_metric_key = st.selectbox("Metric", options=metrics_allowed, format_func=lambda k: metrics_meta[k]["label"], key="product_metric") 

    # Load shared time/window config from Overview logic
    maps = sql_from_config(cfg)
    base_from = maps["base_from"]
    time_col = maps["time_col"]
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})

    windows = [time_cfg.get("default_window", "last_12_months")] + [
        w for w in (time_cfg.get("windows", []) or []) if w != time_cfg.get("default_window")
    ]
    if not any(str(w).lower() == "ltd" for w in windows):
        windows.append("LTD")
    grains = list(time_cfg.get("supported_grains", ["month"]))
    if "ltd" not in [g.lower() for g in grains]:
        grains.append("LTD")

    fc1, fc2 = st.columns(2)
    with fc1:
        window = st.selectbox("Time Window", options=windows, index=0, key="product_window")
    with fc2:
        grain = st.selectbox("Time Grain", options=grains, index=0, key="product_grain")

    where_time = "" if str(window).lower() == "ltd" else resolve_time_filter(cfg.get("sql_macros", {}), window, time_col)

    # Top panel: Asset + Ring chart + Bar chart (3 columns)
    top1, top2, top3 = st.columns([1,1,2])

    # 1) Asset panel
    with top1:
        st.subheader("Asset")
        asset_field = tab_cfg.get("asset_url_field", "")
        # Pick a representative value (top by metric)
        dim_col = dim_defs[chosen_dim_key].get("column") or dim_defs[chosen_dim_key].get("expression")
        metric_expr = metrics_meta[chosen_metric_key]["sql"]
        sql_top = f"""
        SELECT {dim_col} AS ENTITY, {metric_expr} AS VAL
        {base_from}
        {('WHERE ' + where_time) if where_time else ''}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 1
        """
        df_top = run_sql(sql_top)
        entity = df_top.iloc[0]["ENTITY"] if not df_top.empty else "N/A"
        if asset_field:
            sql_asset = f"SELECT {asset_field} AS URL {base_from} WHERE {dim_col} = %s LIMIT 1"
            try:
                # Snowpark parameter binding not available here; do simple string literal escaping
                safe_entity = str(entity).replace("'", "''")
                sql_asset = f"SELECT {asset_field} AS URL {base_from} WHERE {dim_col} = '{safe_entity}' LIMIT 1"
                df_asset = run_sql(sql_asset)
                url = df_asset.iloc[0]["URL"] if not df_asset.empty else ""
            except Exception:
                url = ""
            if url:
                st.image(url, use_column_width=True)
            else:
                st.markdown(f"**{entity}**")
        else:
            st.markdown(f"**{entity}**")

    # 2) Ring chart (donut)
    with top2:
        st.subheader("Composition")
        sql_ring = f"""
        SELECT {dim_col} AS ENTITY, {metric_expr} AS VALUE
        {base_from}
        {('WHERE ' + where_time) if where_time else ''}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 12
        """
        df_ring = run_sql(sql_ring)
        if not df_ring.empty:
            # Donut using altair: pie with innerRadius
            ring = alt.Chart(df_ring).mark_arc(innerRadius=60).encode(
                theta="VALUE:Q",
                color=alt.Color("ENTITY:N", legend=None),
                tooltip=["ENTITY:N", "VALUE:Q"],
            ).properties(height=260)
            st.altair_chart(ring, use_container_width=True)
        else:
            st.info("No data for ring chart.")

    # 3) Time bar chart
    with top3:
        st.subheader("Metric by Time")
        if str(grain).lower() == "ltd":
            sql_bar = f"""
            WITH base AS (
              SELECT DATE_TRUNC('month', {time_col}) AS PERIOD, {metric_expr} AS VALUE
              {base_from}
              {('WHERE ' + where_time) if where_time else ''}
              GROUP BY 1
            )
            SELECT PERIOD, SUM(VALUE) OVER (ORDER BY PERIOD ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS VALUE
            FROM base
            ORDER BY PERIOD
            """
        else:
            sql_bar = f"""
            SELECT DATE_TRUNC('{grain}', {time_col}) AS PERIOD, {metric_expr} AS VALUE
            {base_from}
            {('WHERE ' + where_time) if where_time else ''}
            GROUP BY 1
            ORDER BY 1
            """
        df_bar = run_sql(sql_bar)
        if not df_bar.empty:
            chart = alt.Chart(df_bar).mark_bar().encode(x="PERIOD:T", y="VALUE:Q", tooltip=["PERIOD:T", "VALUE:Q"]).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data for bar chart.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Bottom: Heatmap + Big grid
    bottom1, bottom2 = st.columns([2,1])

    with bottom1:
        st.subheader("Heatmap")
        # Use selected secondary dimension
        other_key = other_dim_key
        other_col = dim_defs[other_key].get("column") or dim_defs[other_key].get("expression")
        sql_heat = f"""
        SELECT {dim_col} AS A, {other_col} AS B, {metric_expr} AS VALUE
        {base_from}
        {('WHERE ' + where_time) if where_time else ''}
        GROUP BY 1,2
        LIMIT 200
        """
        df_h = run_sql(sql_heat)
        if not df_h.empty:
            heat = alt.Chart(df_h).mark_rect().encode(
                x=alt.X("A:N", title=dim_defs[chosen_dim_key]["label"]),
                y=alt.Y("B:N", title=dim_defs[other_key]["label"]),
                color=alt.Color("VALUE:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["A:N", "B:N", "VALUE:Q"],
            ).properties(height=360)
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("No data for heatmap.")

    with bottom2:
        st.subheader("Detail Grid")
        # Grid by selected dimension with all key metrics from cards
        card_keys = cfg.get("cards", {}).get("default_metrics", [])
        metric_exprs = [metrics_meta[k]["sql"] + f" AS {k.upper()}" for k in card_keys if k in metrics_meta]
        select_metrics = ",\n          ".join(metric_exprs)
        sql_grid = f"""
        SELECT {dim_col} AS DIMENSION,
          {select_metrics}
        {base_from}
        {('WHERE ' + where_time) if where_time else ''}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 200
        """
        df_g = run_sql(sql_grid)
        if not df_g.empty:
            # Rename columns to friendly labels from YAML once, applied everywhere
            rename_map = {"DIMENSION": dim_defs[chosen_dim_key]["label"]}
            for k in card_keys:
                if k in metrics_meta:
                    rename_map[k.upper()] = metrics_meta[k]["label"]
            df_g = df_g.rename(columns=rename_map)

            # Narrow container encourages horizontal scroll for many columns
            st.dataframe(df_g, use_container_width=True, height=360)
        else:
            st.info("No data for detail grid.")


def render_compare_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    """VS tab: pick two entities for a configured dimension and compare metrics."""
    dim_defs = {d["key"]: d for d in cfg.get("dimensions", [])}
    metrics_meta = {m["key"]: m for m in cfg.get("metrics", [])}

    # Dimension to compare
    dim_key = tab_cfg.get("dimension")
    if not dim_key or dim_key not in dim_defs:
        st.warning("Compare tab missing a valid 'dimension' in YAML.")
        return
    dim_col = dim_defs[dim_key].get("column") or dim_defs[dim_key].get("expression")

    # Metrics to compare
    metrics_allowed = [m for m in tab_cfg.get("metrics_allowed", []) if m in metrics_meta]
    if not metrics_allowed:
        st.warning("Compare tab missing 'metrics_allowed' in YAML.")
        return

    # Time controls (reuse Overview model)
    maps = sql_from_config(cfg)
    base_from = maps["base_from"]
    time_col = maps["time_col"]
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})
    windows = [time_cfg.get("default_window", "last_12_months")] + [
        w for w in (time_cfg.get("windows", []) or []) if w != time_cfg.get("default_window")
    ]
    if not any(str(w).lower() == "ltd" for w in windows):
        windows.append("LTD")
    grains = list(time_cfg.get("supported_grains", ["month"]))
    if "ltd" not in [g.lower() for g in grains]:
        grains.append("LTD")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        window = st.selectbox("Time Window", options=windows, index=0, key="vs_window")
    with tc2:
        grain = st.selectbox("Time Grain", options=grains, index=0, key="vs_grain")
    with tc3:
        chosen_metrics = st.multiselect(
            "Metrics",
            options=metrics_allowed,
            default=metrics_allowed[:3],
            format_func=lambda k: metrics_meta[k]["label"],
            key="vs_metrics",
        )

    where_time = "" if str(window).lower() == "ltd" else resolve_time_filter(cfg.get("sql_macros", {}), window, time_col)

    # Fetch distinct dimension values for selection
    list_sql = f"""
    SELECT DISTINCT {dim_col} AS ENTITY
    {base_from}
    {('WHERE ' + where_time) if where_time else ''}
    ORDER BY 1
    LIMIT 500
    """
    entities_df = run_sql(list_sql)
    entity_options = entities_df["ENTITY"].dropna().astype(str).tolist() if not entities_df.empty else []

    s1, s2 = st.columns(2)
    with s1:
        a_val = st.selectbox("Left", options=entity_options, index=0 if entity_options else None, key="vs_a")
    with s2:
        b_val = st.selectbox("Right", options=entity_options, index=1 if len(entity_options) > 1 else 0, key="vs_b")

    if not a_val or not b_val or not chosen_metrics:
        st.info("Select two entities and at least one metric.")
        return

    # Pre-escape selected entities for SQL reuse
    a_val_esc = str(a_val).replace("'", "''")
    b_val_esc = str(b_val).replace("'", "''")

    # Comparison table
    rows = []
    for mkey in chosen_metrics:
        meta = metrics_meta[mkey]
        label = meta["label"]
        expr = meta["sql"]
        sql_cmp = f"""
        WITH base AS (
          SELECT {dim_col} AS ENTITY, {expr} AS VAL
          {base_from}
          {('WHERE ' + where_time) if where_time else ''}
          GROUP BY 1
        )
        SELECT ENTITY, VAL FROM base WHERE ENTITY IN (%s, %s)
        """
        sql_cmp = sql_cmp % (f"'{a_val_esc}'", f"'{b_val_esc}'")
        df_cmp = run_sql(sql_cmp)
        va = float(df_cmp[df_cmp["ENTITY"].astype(str) == str(a_val)].iloc[0]["VAL"]) if not df_cmp.empty and (df_cmp["ENTITY"].astype(str) == str(a_val)).any() else 0.0
        vb = float(df_cmp[df_cmp["ENTITY"].astype(str) == str(b_val)].iloc[0]["VAL"]) if not df_cmp.empty and (df_cmp["ENTITY"].astype(str) == str(b_val)).any() else 0.0
        delta = va - vb
        pct = (delta / vb * 100.0) if vb != 0 else 0.0
        if abs(va - vb) < 1e-12:
            winner_label = "Tie"
        else:
            winner_label = str(a_val) if va > vb else str(b_val)
        pct_str = f"{pct:+.1f}%"
        rows.append({
            "Metric": label,
            f"{a_val}": va,
            f"{b_val}": vb,
            "Winner": winner_label,
            "Δ": delta,
            "%Δ": pct_str,
        })
    df_rows = pd.DataFrame(rows)
    st.dataframe(df_rows, use_container_width=True)

    # Time series comparison for selected trend metric
    st.markdown("---")
    st.subheader("Trend")
    trend_metric_key = st.selectbox(
        "Trend Metric",
        options=chosen_metrics,
        format_func=lambda k: metrics_meta[k]["label"],
        index=0,
        key="vs_trend_metric"
    )
    primary_expr = metrics_meta[trend_metric_key]["sql"]
    if str(grain).lower() == "ltd":
        sql_ts = f"""
        WITH base AS (
          SELECT DATE_TRUNC('month', {time_col}) AS PERIOD, {dim_col} AS ENTITY, {primary_expr} AS VAL
          {base_from}
          {('WHERE ' + where_time) if where_time else ''}
          GROUP BY 1,2
        )
        SELECT PERIOD, ENTITY, SUM(VAL) OVER (PARTITION BY ENTITY ORDER BY PERIOD ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS VAL
        FROM base
        WHERE ENTITY IN (%s, %s)
        ORDER BY PERIOD
        """
    else:
        sql_ts = f"""
        SELECT DATE_TRUNC('{grain}', {time_col}) AS PERIOD, {dim_col} AS ENTITY, {primary_expr} AS VAL
        {base_from}
        {('WHERE ' + where_time) if where_time else ''}
        GROUP BY 1,2
        HAVING {dim_col} IN (%s, %s)
        ORDER BY 1
        """
    sql_ts = sql_ts % (f"'{a_val_esc}'", f"'{b_val_esc}'")
    df_ts = run_sql(sql_ts)
    if not df_ts.empty:
        chart = alt.Chart(df_ts).mark_line(point=True).encode(
            x="PERIOD:T",
            y="VAL:Q",
            color="ENTITY:N",
            tooltip=["PERIOD:T", "ENTITY:N", "VAL:Q"],
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data for time series.")


def render_topn_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    """Top-N tab: select a grouping dimension, a metric, and N; show values and contribution."""
    dim_defs = {d["key"]: d for d in cfg.get("dimensions", [])}
    metrics_meta = {m["key"]: m for m in cfg.get("metrics", [])}

    split_dims = [d for d in tab_cfg.get("split_dimensions", []) if d in dim_defs]
    if not split_dims:
        st.warning("Top N tab is missing 'split_dimensions' in YAML.")
        return
    n_options = tab_cfg.get("n_options", [5, 10, 25, 50])

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        dim_key = st.selectbox("Group By", options=split_dims, format_func=lambda k: dim_defs[k]["label"], key="topn_dim")
    with c2:
        # Default to first overview card metric if available
        default_metric_key = (cfg.get("cards", {}).get("default_metrics", []) or [list(metrics_meta.keys())[0]])[0]
        metric_key = st.selectbox(
            "Metric",
            options=list(metrics_meta.keys()),
            index=max(0, list(metrics_meta.keys()).index(default_metric_key)) if default_metric_key in metrics_meta else 0,
            format_func=lambda k: metrics_meta[k]["label"],
            key="topn_metric",
        )
    with c3:
        top_n = st.selectbox("Top N", options=n_options, index=0, key="topn_n")

    dim_col = dim_defs[dim_key].get("column") or dim_defs[dim_key].get("expression")
    metric_expr = metrics_meta[metric_key]["sql"]

    # Time controls reuse
    maps = sql_from_config(cfg)
    base_from = maps["base_from"]
    time_col = maps["time_col"]
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})
    windows = [time_cfg.get("default_window", "last_12_months")] + [
        w for w in (time_cfg.get("windows", []) or []) if w != time_cfg.get("default_window")
    ]
    if not any(str(w).lower() == "ltd" for w in windows):
        windows.append("LTD")
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        window = st.selectbox("Time Window", options=windows, index=0, key="topn_window")
    with wcol2:
        show_share = st.checkbox("Show Contribution %", value=True, key="topn_share")
    where_time = "" if str(window).lower() == "ltd" else resolve_time_filter(cfg.get("sql_macros", {}), window, time_col)

    # Overall total (for contribution)
    total_sql = f"SELECT {metric_expr} AS TOTAL {base_from} {('WHERE ' + where_time) if where_time else ''}"
    total_df = run_sql(total_sql)
    total_val = float(total_df.iloc[0]["TOTAL"]) if not total_df.empty else 0.0

    # Top-N by dimension
    top_sql = f"""
    SELECT {dim_col} AS DIMENSION, {metric_expr} AS VALUE
    {base_from}
    {('WHERE ' + where_time) if where_time else ''}
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT {int(top_n)}
    """
    df = run_sql(top_sql)

    if df.empty:
        st.info("No data for Top N.")
        return

    if show_share and total_val != 0:
        df["% Share"] = (df["VALUE"] / total_val * 100.0).map(lambda x: f"{x:.1f}%")

    # Rename for display
    df_disp = df.rename(columns={"DIMENSION": dim_defs[dim_key]["label"], "VALUE": metrics_meta[metric_key]["label"]})
    st.dataframe(df_disp, use_container_width=True, height=520)

    # Optional bar chart
    try:
        chart_data = df.copy()
        chart_data["DIMENSION"] = chart_data["DIMENSION"].astype(str)
        bar = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("VALUE:Q", title=metrics_meta[metric_key]["label"]),
            y=alt.Y("DIMENSION:N", sort='-x', title=dim_defs[dim_key]["label"]),
            tooltip=["DIMENSION:N", "VALUE:Q"]
        ).properties(height=min(40 * len(chart_data), 600))
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        pass

def render_self_service_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    """Flexible grid: multi-dimension + multi-metric with download."""
    dim_defs = {d["key"]: d for d in cfg.get("dimensions", [])}
    metrics_meta = {m["key"]: m for m in cfg.get("metrics", [])}

    selectable_dims = [d for d in tab_cfg.get("selectable_dimensions", []) if d in dim_defs]
    selectable_metrics = [m for m in tab_cfg.get("selectable_metrics", []) if m in metrics_meta]

    if not selectable_metrics:
        st.warning("Self Service is not configured with metrics.")
        return

    # Controls row
    c1, c2 = st.columns(2)
    with c1:
        dims_chosen = st.multiselect(
            "Dimensions",
            options=selectable_dims,
            default=selectable_dims[:1],
            format_func=lambda k: dim_defs[k]["label"],
            key="ss_dims",
        )
    with c2:
        metrics_chosen = st.multiselect(
            "Metrics",
            options=selectable_metrics,
            default=selectable_metrics[:3],
            format_func=lambda k: metrics_meta[k]["label"],
            key="ss_metrics",
        )

    # Time controls
    maps = sql_from_config(cfg)
    base_from = maps["base_from"]
    time_col = maps["time_col"]
    time_cfg = cfg.get("time") or cfg.get("app", {}).get("time", {})
    windows = [time_cfg.get("default_window", "last_12_months")] + [
        w for w in (time_cfg.get("windows", []) or []) if w != time_cfg.get("default_window")
    ]
    if not any(str(w).lower() == "ltd" for w in windows):
        windows.append("LTD")

    fc1, fc2 = st.columns(2)
    with fc1:
        window = st.selectbox("Time Window", options=windows, index=0, key="ss_window")
    with fc2:
        limit_rows = st.number_input("Row Limit", min_value=10, max_value=10000, value=1000, step=10, key="ss_limit")

    where_time = "" if str(window).lower() == "ltd" else resolve_time_filter(cfg.get("sql_macros", {}), window, time_col)

    # Build SQL
    dim_cols = [dim_defs[k].get("column") or dim_defs[k].get("expression") for k in dims_chosen]
    select_dims = ", ".join([f"{col} AS {dim_defs[k]['label'].replace(' ', '_').upper()}" for k, col in zip(dims_chosen, dim_cols)])
    metric_parts = [metrics_meta[k]["sql"] + f" AS {k.upper()}" for k in metrics_chosen]
    select_metrics = ",\n       ".join(metric_parts)

    select_list = select_metrics if not select_dims else f"{select_dims},\n       {select_metrics}"
    group_by_clause = "" if not dims_chosen else "GROUP BY " + ", ".join(str(i+1) for i in range(len(dims_chosen)))
    order_by_clause = "" if not metrics_chosen else f"ORDER BY {len(dims_chosen)+1} DESC"

    sql = f"""
    SELECT {select_list}
    {base_from}
    {('WHERE ' + where_time) if where_time else ''}
    {group_by_clause}
    {order_by_clause}
    LIMIT {int(limit_rows)}
    """

    with st.expander("SQL", expanded=False):
        st.code(sql, language="sql")

    df = run_sql(sql)
    if df.empty:
        st.info("No data returned.")
        return

    # Friendly column names: already used YAML labels for dimensions; map metrics
    rename_map = {}
    for k in metrics_chosen:
        rename_map[k.upper()] = metrics_meta[k]["label"]
    df = df.rename(columns=rename_map)

    st.dataframe(df, use_container_width=True, height=520)

    # Download
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, file_name="self_service_export.csv", mime="text/csv")


def render_search_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    """Cortex Search: configurable via YAML.
    YAML fields:
      - cortex_search_service: fully qualified service name
      - default_limit: int
      - default_columns: [col,...]
      - examples: [str,...]
    """
    service = tab_cfg.get("cortex_search_service", "").strip()
    default_limit = int(tab_cfg.get("default_limit", 25))
    default_cols = tab_cfg.get("default_columns", [])
    examples = tab_cfg.get("examples", ["parking issues", "food quality", "game experience"]) 
    search_semantic_model = tab_cfg.get("semantic_model_file")  # optional: derive fields from ML/semantic file
    # Strict YAML-driven roles (optional). If absent, we render Grid only.
    roles_cfg = tab_cfg.get("roles", {}) or {}

    if not service:
        st.error("No Cortex Search service configured in YAML (cortex_search_service).")
        return

    # If default_columns not provided, attempt to derive from semantic model file in @DB.SCHEMA.SEMANTIC_MODELS
    if not default_cols and search_semantic_model:
        try:
            db = cfg["app"]["data_source"]["database"]
            sc = cfg["app"]["data_source"]["schema"]
            stage = f"@{db}.{sc}.SEMANTIC_MODELS"
            model_path = search_semantic_model
            if not model_path.startswith("@"):
                model_path = f"{stage}/{model_path.split('/')[-1]}"
            # Load YAML from stage
            df_sem = session.sql(f"SELECT $1 FROM {model_path}").to_pandas()
            sem_yaml = "\n".join([row for row in df_sem.iloc[:, 0].tolist() if row is not None])
            sem = yaml.safe_load(sem_yaml)
            # Extract candidate columns from semantic model (tables -> dimensions/facts/time_dimensions)
            cols = []
            for tbl in sem.get("tables", []):
                for sec in ("dimensions", "facts", "time_dimensions"):
                    for item in tbl.get(sec, []) or []:
                        nm = item.get("name") or item.get("expr")
                        if nm:
                            cols.append(nm)
            default_cols = sorted(list(dict.fromkeys(cols)))[:30]  # cap to reasonable size
        except Exception as e:
            st.warning(f"Could not infer columns from semantic model: {e}")

    with st.form("search_form"):
        search_query = st.text_input("Search", placeholder="e.g., 'parking issues'")
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=default_limit)
        submitted = st.form_submit_button("Run Search")

    if submitted and search_query:
        try:
            # Request ONLY the YAML-declared (or derived) columns.
            cols = list(dict.fromkeys(default_cols or []))
            params = {
                "query": search_query,
                "limit": int(limit)
            }
            if cols:
                params["columns"] = cols
            json_param = json.dumps(params)
            sql = f"""
            WITH search_results AS (
              SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                '{service}',
                '{json_param}'
              ) AS search_result
            )
            SELECT
              result.value as result_json,
              result.index + 1 as result_rank
            FROM search_results,
            LATERAL FLATTEN(input => PARSE_JSON(search_results.search_result):results) as result
            ORDER BY result.index
            """
            df = session.sql(sql).to_pandas()
            if df.empty:
                st.info("No results.")
            else:
                # Expand selected columns and render a clean table (hide raw JSON)
                expanded_rows = []
                for _, row in df.iterrows():
                    try:
                        rec = json.loads(row["RESULT_JSON"]) if isinstance(row["RESULT_JSON"], str) else row["RESULT_JSON"]
                    except Exception:
                        rec = {}
                    rec_ci = {str(k).lower(): v for k, v in (rec.items() if isinstance(rec, dict) else [])}
                    flat = {"RESULT_RANK": row.get("RESULT_RANK", None)}
                    for col in default_cols:
                        flat[str(col).upper()] = rec_ci.get(str(col).lower())
                    expanded_rows.append(flat)
                df_view = pd.DataFrame(expanded_rows)

                # Roles from YAML
                id_col = (roles_cfg.get("id") or "").upper()
                label_col = (roles_cfg.get("label") or "").upper()
                primary_col = (roles_cfg.get("primary_field") or "").upper()
                comment_col = (roles_cfg.get("comment") or "").upper()
                dim_cols = [str(c).upper() for c in roles_cfg.get("dimensions", []) or []]
                detail_items = roles_cfg.get("field_details", []) or []

                # Optional summary
                if primary_col and primary_col in df_view.columns:
                    try:
                        avg_val = pd.to_numeric(df_view.get(primary_col), errors="coerce").mean()
                    except Exception:
                        avg_val = None
                    s1, s2 = st.columns(2)
                    with s1:
                        st.metric("Average", f"{avg_val:.2f}" if avg_val is not None and pd.notna(avg_val) else "N/A")
                    with s2:
                        st.metric("Results", len(df_view))

                grid_tab, detail_tab = st.tabs(["Grid", "Details"])
                with grid_tab:
                    st.dataframe(df_view, use_container_width=True, height=520)
                    st.download_button("Download CSV", df_view.to_csv(index=False).encode("utf-8"), file_name="search_results.csv", mime="text/csv")

                with detail_tab:
                    # Only render detail UI if roles are specified
                    if roles_cfg:
                        for i, row in df_view.head(25).iterrows():
                            header_bits = []
                            if id_col and id_col in row:
                                header_bits.append(str(row.get(id_col)))
                            if label_col and label_col in row:
                                header_bits.append(str(row.get(label_col)))
                            if primary_col and primary_col in row:
                                header_bits.append(str(row.get(primary_col)))
                            if not header_bits:
                                header_bits = [f"Result #{row.get('RESULT_RANK', i+1)}"]
                            header = " - ".join(header_bits)
                            with st.expander(header, expanded=(i==0)):
                                if comment_col and comment_col in row and pd.notna(row.get(comment_col)):
                                    st.markdown("### 💬 Comment")
                                    st.markdown(f"*{row.get(comment_col)}*")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if dim_cols:
                                        st.markdown("**🏷️ Dimensions**")
                                        for d in dim_cols:
                                            label = d.replace("_", " ").title()
                                            st.markdown(f"• **{label}:** {row.get(d, 'N/A')}")
                                with col2:
                                    if detail_items:
                                        st.markdown("**📊 Field Details**")
                                        for item in detail_items:
                                            fld = str(item.get('field', '')).upper()
                                            if not fld:
                                                continue
                                            label = item.get('label') or fld.replace('_', ' ').title()
                                            val = row.get(fld)
                                            if pd.notna(val):
                                                st.markdown(f"• **{label}:** {val}")
        except Exception as e:
            st.error(f"Search error: {e}")

    st.markdown("### Quick Searches")
    cols = st.columns(min(4, len(examples)) or 1)
    for i, ex in enumerate(examples[:4]):
        with cols[i]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["search_form-search_query"] = ex  # populate field if same keying supported
                st.info(f"Use the form above with: {ex}")


def render_analyst_tab(cfg: Dict[str, Any], tab_cfg: Dict[str, Any]):
    """Cortex Analyst integrated with semantic models in @<DB>.<SCHEMA>.SEMANTIC_MODELS.
    YAML fields:
      - examples: [str,...]
    """
    db = cfg["app"]["data_source"]["database"]
    sc = cfg["app"]["data_source"]["schema"]
    stage = f"@{db}.{sc}.SEMANTIC_MODELS"
    examples = tab_cfg.get("examples", [
        "Top 5 themes by sentiment last quarter",
        "Average overall score by segment",
        "Total reviews by month"
    ])

    st.markdown(f"Using semantic models from stage: `{stage}`")

    # List YAML files in stage
    try:
        files_df = _cached_list(stage)
        files = [row[files_df.columns[0]] for _, row in files_df.iterrows() if str(row[files_df.columns[0]]).lower().endswith((".yaml", ".yml"))]
    except Exception as e:
        st.error(f"Could not list semantic models: {e}")
        files = []

    if not files:
        st.warning("No semantic model files found.")

    model = st.selectbox("Semantic Model", options=files, format_func=lambda p: p.split("/")[-1] if p else p)

    question_col1, question_col2 = st.columns([3,1])
    with question_col1:
        question = st.text_area("Ask a question", height=100)
    with question_col2:
        st.markdown("**Examples**")
        for ex in examples:
            if st.button(ex):
                question = ex
                st.session_state["analyst_prefill"] = ex
        if st.session_state.get("analyst_prefill") and not question:
            question = st.session_state["analyst_prefill"]

    if st.button("Ask Analyst") and question and model:
        try:
            import _snowflake
            request_body = {
                "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                "semantic_model_file": f"{stage}/{model.split('/')[-1]}"
            }
            resp = _snowflake.send_snow_api_request(
                "POST",
                "/api/v2/cortex/analyst/message",
                {}, {}, request_body, {}, 30000,
            )
            if resp["status"] >= 400:
                raise Exception(resp)
            content = json.loads(resp["content"]).get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "text":
                    st.markdown(item.get("text", ""))
                elif item.get("type") == "sql":
                    sql = item.get("statement", "")
                    if sql:
                        with st.expander("Generated SQL", expanded=False):
                            st.code(sql, language="sql")
                        try:
                            df = session.sql(sql.strip(";")) .to_pandas()
                            # persist for follow-up interactions (e.g., AI narrative)
                            st.session_state["analyst_last_sql"] = sql.strip(";")
                            st.session_state["analyst_last_df"] = df
                        except Exception as e:
                            st.error(f"SQL execution error: {e}")
        except Exception as e:
            st.error(f"Analyst error: {e}")

    # Persistent results section to avoid losing UI on reruns
    last_sql = st.session_state.get("analyst_last_sql")
    last_df = st.session_state.get("analyst_last_df")
    if last_sql:
        st.markdown("---")
        # Results area with persistent view selector
        if isinstance(last_df, pd.DataFrame) and not last_df.empty:
            st.subheader("Results")
            view = st.radio("View", ["Grid", "Bar", "Line"], horizontal=True, key="analyst_results_view")

            # Common helpers
            all_columns = list(last_df.columns)
            numeric_columns = [c for c in all_columns if pd.api.types.is_numeric_dtype(last_df[c])]
            categorical_columns = [c for c in all_columns if c not in numeric_columns]

            def _maybe_parse_datetime(series: pd.Series) -> pd.Series:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                    return parsed if parsed.notna().any() else series
                except Exception:
                    return series

            def _apply_grain(df: pd.DataFrame, dim_col: str, grain: str) -> pd.DataFrame:
                """Return df with three helper columns for plotting ordered period labels.
                - X_TS: period start timestamp
                - X_LABEL: friendly label string
                - X_ORDER: numeric for sorting
                """
                dt = _maybe_parse_datetime(df[dim_col])
                if not pd.api.types.is_datetime64_any_dtype(dt):
                    return df
                if grain == "day":
                    ts = dt.dt.to_period('D').dt.to_timestamp()
                    label = ts.dt.strftime('%b %d %Y')
                elif grain == "quarter":
                    ts = dt.dt.to_period('Q').dt.to_timestamp()
                    label = ts.dt.to_period('Q').astype(str)
                elif grain == "year":
                    ts = dt.dt.to_period('Y').dt.to_timestamp()
                    label = ts.dt.strftime('%Y')
                else:
                    ts = dt.dt.to_period('M').dt.to_timestamp()
                    label = ts.dt.strftime('%b %Y')
                out = df.copy()
                out["X_TS"] = ts
                out["X_LABEL"] = label
                out["X_ORDER"] = ts.view('int64')
                return out

            if view == "Grid":
                st.dataframe(last_df, use_container_width=True)

            elif view == "Bar":
                if not numeric_columns:
                    st.info("No numeric columns to chart.")
                else:
                    c1, c2, c3 = st.columns([2,1,1])
                    with c1:
                        bar_dim = st.selectbox("Dimension", options=categorical_columns or all_columns, key="analyst_bar_dim")
                    with c2:
                        stacked = st.checkbox("Stacked", value=True, key="analyst_bar_stacked")
                    # Grain selector if time-like
                    is_time_like = pd.api.types.is_datetime64_any_dtype(last_df[bar_dim]) or pd.to_datetime(last_df[bar_dim], errors="coerce").notna().sum() >= max(5, len(last_df) * 0.6)
                    with c3:
                        grain = st.selectbox("Grain", ["month","day","quarter","year"], index=0, key="analyst_bar_grain") if is_time_like else None

                    bar_series = st.multiselect(
                        "Series (metrics)", options=numeric_columns, default=numeric_columns[:2], key="analyst_bar_series"
                    )
                    if not bar_series:
                        st.info("Select at least one metric for the bar series.")
                    else:
                        try:
                            df_plot = last_df[[bar_dim] + bar_series].copy()
                            if grain:
                                df_plot = _apply_grain(df_plot, bar_dim, grain)
                            for m in bar_series:
                                df_plot[m] = pd.to_numeric(df_plot[m], errors="coerce")

                            # Use label for categorical x-axis to respect grain; sort by X_ORDER
                            id_vars = [bar_dim]
                            if "X_LABEL" in df_plot.columns:
                                id_vars.append("X_LABEL")
                            if "X_ORDER" in df_plot.columns:
                                id_vars.append("X_ORDER")
                            long_df = df_plot.melt(id_vars=id_vars, value_vars=bar_series, var_name="Series", value_name="Value")

                            if "X_LABEL" in long_df.columns:
                                long_df = long_df.sort_values("X_ORDER") if "X_ORDER" in long_df.columns else long_df
                                x_enc = alt.X("X_LABEL:N", sort=None, title=bar_dim)
                            else:
                                x_enc = alt.X(f"{bar_dim}:N")
                            if stacked:
                                chart = alt.Chart(long_df).mark_bar().encode(
                                    x=x_enc,
                                    y=alt.Y("Value:Q", stack="zero"),
                                    color="Series:N",
                                    tooltip=[alt.Tooltip("X_LABEL:N", title=bar_dim) if "X_LABEL" in long_df.columns else alt.Tooltip(f"{bar_dim}:N", title=bar_dim), "Series", "Value:Q"],
                                )
                            else:
                                chart = alt.Chart(long_df).mark_bar().encode(
                                    x=x_enc,
                                    y=alt.Y("Value:Q"),
                                    color="Series:N",
                                    xOffset="Series:N",
                                    tooltip=[alt.Tooltip("X_LABEL:N", title=bar_dim) if "X_LABEL" in long_df.columns else alt.Tooltip(f"{bar_dim}:N", title=bar_dim), "Series", "Value:Q"],
                                )
                            st.altair_chart(chart.properties(height=420), use_container_width=True)
                        except Exception as e:
                            st.error(f"Bar chart error: {e}")

            elif view == "Line":
                if not numeric_columns:
                    st.info("No numeric columns to chart.")
                else:
                    c1, c2, c3 = st.columns([2,2,1])
                    with c1:
                        line_dim = st.selectbox("Dimension", options=all_columns, key="analyst_line_dim")
                    # Grain selector if time-like
                    is_time_like = pd.api.types.is_datetime64_any_dtype(last_df[line_dim]) or pd.to_datetime(last_df[line_dim], errors="coerce").notna().sum() >= max(5, len(last_df) * 0.6)
                    with c2:
                        primary_series = st.multiselect(
                            "Primary series (metrics)", options=numeric_columns, default=numeric_columns[:2], key="analyst_line_primary"
                        )
                    with c3:
                        grain = st.selectbox("Grain", ["month","day","quarter","year"], index=0, key="analyst_line_grain") if is_time_like else None

                    secondary_metric = st.selectbox(
                        "Secondary (optional, RHS)", options=["(none)"] + [m for m in numeric_columns if m not in primary_series], key="analyst_line_secondary"
                    )

                    if not primary_series:
                        st.info("Select at least one primary series metric.")
                    else:
                        try:
                            df_plot = last_df.copy()
                            if grain:
                                df_plot = _apply_grain(df_plot, line_dim, grain)

                            # Use timestamp field for ordering and consistent ticks
                            x_field = line_dim
                            x_type = "T" if pd.api.types.is_datetime64_any_dtype(df_plot[line_dim]) else "N"

                            # Primary layer (melted)
                            melt_cols = [line_dim] + primary_series
                            df_primary = df_plot[melt_cols].copy()
                            for m in primary_series:
                                df_primary[m] = pd.to_numeric(df_primary[m], errors="coerce")
                            id_vars = [line_dim]
                            long_p = df_primary.melt(id_vars=id_vars, value_vars=primary_series, var_name="Series", value_name="Value")

                            layer_primary = alt.Chart(long_p).mark_line(point=True).encode(
                                x=alt.X(f"{x_field}:{x_type}"),
                                y=alt.Y("Value:Q", title="Value"),
                                color="Series:N",
                                tooltip=[alt.Tooltip(f"{x_field}:{'T' if x_type=='T' else 'N'}", title=line_dim), "Series", "Value:Q"],
                            )

                            if secondary_metric and secondary_metric != "(none)":
                                df_sec = df_plot[[line_dim, secondary_metric]].copy()
                                df_sec[secondary_metric] = pd.to_numeric(df_sec[secondary_metric], errors="coerce")
                                layer_secondary = alt.Chart(df_sec).mark_line(point=True, color="#ef4444").encode(
                                    x=alt.X(f"{x_field}:{x_type}"),
                                    y=alt.Y(f"{secondary_metric}:Q", axis=alt.Axis(title=secondary_metric, titleColor="#ef4444")),
                                    tooltip=[alt.Tooltip(f"{x_field}:{'T' if x_type=='T' else 'N'}", title=line_dim), alt.Tooltip(f"{secondary_metric}:Q")],
                                )
                                chart = alt.layer(layer_primary, layer_secondary).resolve_scale(y='independent')
                            else:
                                chart = layer_primary
                            st.altair_chart(chart.properties(height=420), use_container_width=True)
                        except Exception as e:
                            st.error(f"Line chart error: {e}")
        # Analysis section FIRST for business users (only runs on button click)
        st.subheader("AI Narrative (Cortex Complete)")
        with st.form("analyst_narrative_form"):
            colx1, colx2, colx3 = st.columns([2,1,1])
            with colx1:
                model = st.selectbox("Model", ["llama3.1-8b","llama3.1-70b","mixtral-8x7b","mistral-large"], index=0, key="analyst_cc_model_persist")
            with colx2:
                temperature = st.slider("Temp", 0.0, 1.0, 0.2, 0.1, key="analyst_cc_temp_persist")
            with colx3:
                max_tokens = st.number_input("Max Tokens", min_value=200, max_value=8000, value=1500, step=100, key="analyst_cc_tokens_persist")

            prompt_text = st.text_area(
                "Optional question for the AI about these results (leave blank for general analysis)",
                value=st.session_state.get("analyst_cc_prompt_persist", ""),
                height=80,
                key="analyst_cc_prompt_persist"
            )
            submitted_narrative = st.form_submit_button("Generate Analysis")

        def _run_cc_and_render(_df, _sql):
            try:
                sample = json.loads(_df.head(50).to_json(orient="records", date_format="iso"))
                context_obj = {
                    "columns": list(_df.columns),
                    "row_count": int(len(_df)),
                    "sample_rows": sample,
                    "original_sql": _sql
                }
                context_json = json.dumps(context_obj, ensure_ascii=False)
                context_escaped = context_json.replace("'","''")
                sys_prompt = st.session_state.get("analyst_cc_sys_prompt_persist", "You are a senior data analyst. Be concise and insightful.")
                user_text = prompt_text or "Provide an executive-style analysis: key trends, outliers, comparisons, and recommended next steps. Use bullet points where helpful."
                user_escaped = user_text.replace("'","''")
                cc_sql = f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    '{model}',
                    [
                        {{'role':'system','content':'{sys_prompt}'}},
                        {{'role':'user','content':'Context JSON: {context_escaped}\n\nQuestion: {user_escaped}'}}
                    ],
                    {{'temperature': {temperature}, 'max_tokens': {int(max_tokens)}}}
                ):choices[0]:messages::string AS AI_RESPONSE
                """
                cc_df = session.sql(cc_sql).to_pandas()
                if not cc_df.empty:
                    st.markdown(cc_df.iloc[0]["AI_RESPONSE"])
                else:
                    st.info("No AI response.")
            except Exception as ee:
                st.error(f"Cortex Complete error: {ee}")

        # Run only when the form button is clicked
        if submitted_narrative and isinstance(last_df, pd.DataFrame) and not last_df.empty:
            _run_cc_and_render(last_df, last_sql)

        # Technical details BELOW in an expander
        with st.expander("Technical details (Analyst + Cortex Complete)", expanded=False):
            st.markdown("**Generated SQL (last)**")
            st.code(last_sql, language="sql")
            st.markdown("**Model Parameters**")
            st.write({
                "model": model,
                "temperature": temperature,
                "max_tokens": int(max_tokens)
            })
            st.markdown("**System Prompt**")
            st.code(st.session_state.get("analyst_cc_sys_prompt_persist", "You are a senior data analyst. Be concise and insightful."))
            st.markdown("**User Prompt**")
            default_user_prompt = "Provide an executive-style analysis: key trends, outliers, comparisons, and recommended next steps. Use bullet points where helpful."
            st.code(prompt_text or default_user_prompt)

if __name__ == "__main__":
    main()


