import streamlit as st
import pandas as pd
import yaml
import json
import os
from snowflake.snowpark.functions import col
import altair as alt
import re

st.set_page_config(
    page_title="TastyBytes From Zero to Snowflake Vignette - 5 (Collaboration)",
    page_icon="https://storage.googleapis.com/lee-demo-streamlit-icon-bucket/tastybytes.png",
    layout="wide"
)

from snowflake.snowpark.context import get_active_session

# Snowpark session
session = get_active_session()

# Get the current database from the Snowflake session context
def get_current_database():
    try:
        result = session.sql("SELECT CURRENT_DATABASE()").collect()
        if result and result[0][0]:
            return result[0][0]
        else:
            raise Exception("Could not determine current database from session")
    except Exception as e:
        raise Exception(f"Failed to get current database: {str(e)}. Ensure DataOps deployment completed successfully.")

# DataOps database configuration - use current database context
db_name = get_current_database()


# -- YAML Utilities

# -- Define reusable yaml functions to select demo and load the metadata necessary to run it

# -- list_yaml_folders_in_stage - checks the stage to get the the subfolders ie areas

def list_yaml_folders_in_stage(session, stage: str) -> list:
    try:
        files_df = session.sql(f"LIST {stage}").to_pandas()
        file_col = files_df.columns[0]
        prefix = "framework_yaml_stage/"
        all_paths = [row[file_col] for _, row in files_df.iterrows()]
        # Only consider files that start with the prefix
        paths_no_prefix = [p[len(prefix):] for p in all_paths if p.startswith(prefix)]
        # Get first dir after the prefix, e.g. tastybytes/...
        areas = sorted({p.split('/')[0] for p in paths_no_prefix if '/' in p})

        return areas
    except Exception as e:
        # Don't show error for missing database/stage - that's expected if no data exists yet
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            return []  # Return empty list silently
        # For other errors, show the error
        st.error(f"Error listing folders in stage: {str(e)}")
        return []
    
# -- list_yaml_files_in_stage - checks the stage to get the yaml files which are all the demos available to run in this area

def list_yaml_files_in_stage(stage: str) -> list:
    try:
        files = session.sql(f"LIST {stage}").to_pandas()
        file_col = files.columns[0]
        yaml_files = [row[file_col] for idx, row in files.iterrows() if row[file_col].lower().endswith(('.yaml', '.yml'))]
        return yaml_files
    except Exception as e:
        # Don't show error for missing database/stage - that's expected if no data exists yet
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            return []  # Return empty list silently
        # For other errors, show the error
        st.error(f"Error listing YAML files in stage: {str(e)}")
        return []

# -- load_yaml_from_stage - this takes the yaml file which has the metadata for the demo and enables you to run it
def load_yaml_from_stage(stage: str, filename: str) -> dict:
    # Remove duplicate stage prefix if present (case-insensitive)
    stage_prefix = stage.replace("@", "")  # e.g., {db_name}.CONFIGS.FRAMEWORK_YAML_STAGE
    stage_leaf = stage_prefix.split(".")[-1]  # FRAMEWORK_YAML_STAGE

    # Remove the double prefix if present (case-insensitive)
    if filename.lower().startswith(f"{stage_leaf.lower()}/"):
        filename = filename[len(stage_leaf)+1:]  # +1 for the slash

    full_path = f"{stage}/{filename}"
 
    try:
        df = session.sql(
            f"SELECT $1 FROM {full_path} (file_format => '{db_name}.CONFIGS.YAML_CSV_FORMAT')"
        ).to_pandas()
    except Exception as e:
        # Handle database/stage access errors gracefully
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            st.warning(f"Database or stage not accessible. Please ensure DataOps deployment completed successfully.")
            return None
        else:
            st.error(f"Error loading YAML file: {str(e)}")
            return None

    if df.empty:
        st.error(f"The file {filename} is empty or not found in the stage.")
        return None

    # Filter out None rows (blank lines in file)
    yaml_content = "\n".join([row for row in df.iloc[:, 0].tolist() if row is not None])
    
    if not yaml_content.strip():
        st.error(f"The YAML file {filename} is empty.")
        return None

    try:
        data = yaml.safe_load(yaml_content)
        if data is None:
            st.error(f"YAML file {filename} parsed as empty. Please check formatting!")
        return data
    except Exception as e:
        st.error(f"Failed to parse YAML in {filename}: {e}")
        return None

# -- show_demo_info - enables you to see what the demo is you want

def show_demo_info(demo_yaml: dict):
    # Assume demo_yaml is already loaded (parsed YAML dict)
    demo = demo_yaml.get('demo', {})
    
    # List of metadata fields to show in table
    meta_fields = [
        ('Topic', 'topic'),
        ('Sub-topic', 'sub_topic'),
        ('Tertiary Topic', 'tertiary_topic'),
        #('Database', 'database'),
        #('Schema', 'schema'),
        #('Owner', 'owner'),
        ('Title', 'title'),
        #('Harness Type', 'harness_type'),
        #('Date Created', 'date_created'),
    ]
    
    # Set up columns: image (1/4 width), metadata (3/4 width)
    col1, col2 = st.columns([1, 3])

    with col1:
        if demo.get("logo_url"):
            st.image(demo["logo_url"], width=100)

    with col2:
        # Collect all metadata rows, skipping empty values
        meta_rows = [
            f"**{label}:** {demo.get(key, '')}" for label, key in meta_fields
            if demo.get(key, "")
        ]
        st.markdown("<br>".join(meta_rows), unsafe_allow_html=True)

    # Overview/description as a block of text
    overview = demo.get("overview", "")
    if overview:
        st.markdown("**Overview:**")
        st.markdown(f"> {overview.replace(chr(10), '<br>')}", unsafe_allow_html=True)
    
    # Optionally show cleanup_commands as a code block
    #if demo.get("cleanup_commands"):
        #st.markdown("**Cleanup Commands:**")
        #st.code("\n".join(demo["cleanup_commands"]), language="sql")

# --- substitute_vars -- responsible for substituting the variables in statements pre-execution

def substitute_vars(sql_list, variables):
    return [stmt.format(**variables) for stmt in sql_list]

# -- SQL Utilities
    
# --- run_sql_batch - Cleanup Scripts Function + Generalized Initialization ---
def run_sql_batch(session, sql_statements: list[str]):
    """
    Executes a list of SQL statements one by one using the provided Snowpark session.
    Silently skips failures (or logs them if needed).
    """
    #st.write(f"statements: {sql_statements}")
    for stmt in sql_statements:
        try:
            session.sql(stmt).collect()
        except Exception as e:
            st.write(f"⚠️ Skipping failed statement:\n{stmt}\nError: {e}")



# -- render_query_block - basically takes the section of code that is going to be run and executes it and renders with optional visualization options

def safe_format_sql(sql_code: str, variable_fields: dict) -> str:
    """
    Safely format SQL code with variables while preserving JSON syntax in Cortex calls.
    This handles cases where SQL contains both {VARIABLE} placeholders and JSON syntax like {'role': 'system'}.
    """
    if not variable_fields:
        return sql_code
    
    # Step 1: Find all legitimate variable placeholders (uppercase variables like {MY_VARIABLE})
    variable_pattern = r'\{([A-Z_][A-Z0-9_]*)\}'
    legitimate_vars = re.findall(variable_pattern, sql_code)
    
    # Step 2: Create temporary markers for legitimate variables
    temp_markers = {}
    temp_sql = sql_code
    
    for i, var in enumerate(legitimate_vars):
        if var in variable_fields:
            marker = f"__TEMP_VAR_{i}__"
            temp_markers[marker] = variable_fields[var]
            temp_sql = temp_sql.replace(f"{{{var}}}", marker)
    
    # Step 3: Escape all remaining curly braces (JSON syntax)
    temp_sql = temp_sql.replace('{', '{{').replace('}', '}}')
    
    # Step 4: Restore legitimate variables as format placeholders
    for marker, value in temp_markers.items():
        temp_sql = temp_sql.replace(marker, f"{{{marker}}}")
    
    # Step 5: Apply format substitution
    try:
        result = temp_sql.format(**temp_markers)
        return result
    except KeyError as e:
        st.error(f"Variable substitution error: {e}")
        return sql_code


def render_query_block(
    session,
    block_id: str,
    title: str,
    query_code: str,
    talk_track: str,
    instructions: str,
    instructions_title: str,
    save_as: str = None,   # <-- New parameter
    variable_fields: dict = None,
    default_chart = None,
    cortex_type: str = None,  # <-- New parameter for Cortex capabilities
    cortex_search_service: str = None,  # <-- For Cortex Search
    semantic_model_file: str = None  # <-- For Cortex Analyst
):

    # Substitute variables in the query_code safely
    if variable_fields:
        query_code = safe_format_sql(query_code, variable_fields)

    # --- Step 1: Show Instructions ---
    if instructions_title:
        st.header(instructions_title)
    if instructions:
        with st.expander("Instructions"):
            st.write(instructions)

    st.markdown("---")
    st.subheader(title)
    
    # --- CORTEX ROUTING: Check for Cortex-specific rendering ---
    if cortex_type == "complete":
        render_cortex_complete_block(session, block_id, query_code, talk_track)
        return
    elif cortex_type == "search":
        render_cortex_search_block(session, block_id, query_code, talk_track, cortex_search_service)
        return
    elif cortex_type == "analyst":
        render_cortex_analyst_block(session, block_id, query_code, talk_track, semantic_model_file)
        return
    
    # --- Standard SQL Block Rendering ---
    st.code(query_code, language="sql")

    # --- Step 2: Run Query ---
    run_button_key = f"run_{block_id}"
    if st.button("Run Query", key=run_button_key):
        try:
            df = session.sql(query_code).to_pandas()
            st.session_state[block_id] = df
            if save_as:
                if not df.empty:
                    if save_as.upper() in df.columns:
                        st.session_state[save_as] = df.iloc[0][save_as.upper()]
                    else:
                        st.session_state[save_as] = df.iloc[0, 0]
                else:
                    st.session_state[save_as] = None
            st.success(f"{len(df)} rows returned.")
        except Exception as e:
            error_msg = str(e)
            # Parse common errors for user-friendly messages
            if "does not exist or not authorized" in error_msg:
                if "Database" in error_msg:
                    st.error("❌ **Database not found.** Please update your SQL query to use `CORTEX_AI_FRAMEWORK_DB` instead of the old database name.")
                elif "Table" in error_msg or "Object" in error_msg:
                    st.error("❌ **Table not found.** Make sure you've run the Synthetic Data Generator and Structured Tables apps first to create your data tables.")
                else:
                    st.error(f"❌ **Permission or object not found:** {error_msg}")
            elif "syntax error" in error_msg.lower():
                st.error(f"❌ **SQL syntax error.** Please check your query:\n\n{error_msg}")
            else:
                st.error(f"❌ **Query failed:** {error_msg}")
            st.info("💡 **Tip:** Check that your database name is `CORTEX_AI_FRAMEWORK_DB` and your tables exist in the correct schema.")

    
    # --- Step 3: Chart Type Selection & Rendering ---
    if block_id in st.session_state:
        df = st.session_state[block_id]


        chart_options = [
            "Table", "Bar Chart", "Line Chart", "Plot", "Map", "Network", "Metric", "Polygon", "Furthest", "H3"
        ]
        # Determine default index
        if default_chart and default_chart in chart_options:
            default_index = chart_options.index(default_chart)
        else:
            default_index = 0

        chart_type = st.selectbox(
            "Display as:",
            chart_options,
            index=default_index,
            key=f"chart_type_{block_id}"
        )

        # --- Render Based on Type ---
        if chart_type == "Table":
            render_table_chart(df)

        elif chart_type in ["Bar Chart", "Line Chart"]:
            render_line_or_bar_chart(df, block_id, chart_type)

        elif chart_type == "Map":
            render_map_chart(df)

        elif chart_type == "Network":
            render_network_chart(df, block_id)

        elif chart_type == "Plot":
            render_plot_chart(df, block_id)

        elif chart_type == "Metric":
            render_metric_chart(df, block_id)

        elif chart_type == "Polygon":
            if all(k in st.session_state for k in ["polygon_points", "bounding_polygon", "center_point"]):
                render_top_selling_polygon_chart(
                    df_points=st.session_state["polygon_points"],
                    df_polygon=st.session_state["bounding_polygon"],
                    df_center=st.session_state["center_point"],
                    location="Paris"
                )
            else:
                st.warning("Missing data for Polygon visualization. Please run all required queries first.")
        


        elif chart_type == "Furthest":
            if "furthest_locations" in st.session_state and variable_fields and "GEOMETRIC_CENTER_POINT" in variable_fields:
                center_point_val = variable_fields.get("GEOMETRIC_CENTER_POINT")
                #st.write(f"cpv: {center_point_val}")
        
                center_point_dict = None
        
                if isinstance(center_point_val, dict):
                    center_point_dict = center_point_val
                    #st.write("It's a dict!")
                elif isinstance(center_point_val, str):
                    if center_point_val.startswith("POINT"):
                        center_point_dict = parse_point_wkt(center_point_val)
                        #st.write("Converted from WKT POINT string.")
                    else:
                        try:
                            # Try to load as JSON
                            center_point_dict = json.loads(center_point_val)
                            #st.write("Parsed JSON string to dict.")
                        except Exception as e:
                            #st.write(f"JSON parse failed: {e}")
                            center_point_dict = None
                else:
                    st.write("Unrecognized type for GEOMETRIC_CENTER_POINT.")
        
                if center_point_dict:
                    render_furthest_locations_chart(
                        df_furthest=st.session_state["furthest_locations"],
                        center_point=center_point_dict
                    )
                else:
                    st.warning("Invalid or missing GEOMETRIC_CENTER_POINT")
            
        
        elif chart_type == "H3":
            if "h3_data" in st.session_state:
                selected_resolution = st.selectbox("Select H3 resolution", [4, 8, 12], key=f"h3_res_{block_id}")
                render_h3_resolution_chart(
                    df_hex=st.session_state["h3_data"],
                    resolution=selected_resolution
                )
            else:
                st.warning("Missing H3 data in session.")

        
        # --- Optional: Talk Track ---
        if talk_track:
            with st.expander("Talk Track", expanded=False):
                st.markdown(talk_track)

    else:
        st.info("Click 'Run Query' to begin.")

    st.markdown("---")

# --- Visualization and Charting Utilities
#
# -- render_top_selling_polygon_chart - visual rendering of top selling polygon (not tested)

def render_top_selling_polygon_chart(df_points: pd.DataFrame, df_polygon: pd.DataFrame, df_center: pd.DataFrame, location: str):
    st.subheader(f"Top Selling Locations in {location} – Bounding Polygon")

    # Base scatter points
    point_chart = alt.Chart(df_points).mark_circle(size=100).encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        color=alt.value("orange"),
        tooltip=['location_id', 'total_sales_usd']
    )

    # Bounding polygon
    polygon_chart = alt.Chart(df_polygon).mark_geoshape(
        fill='lightblue', stroke='blue', strokeWidth=2, opacity=0.3
    ).encode(
        tooltip=['area_in_sq_kilometers']
    )

    # Center point
    center_chart = alt.Chart(df_center).mark_point(shape='diamond', size=200, color='red').encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        tooltip=['label']
    )

    final_chart = (polygon_chart + point_chart + center_chart).project('identity').properties(
        width=700, height=500, title="Bounding Polygon and Center of Top Selling Points"
    )

    st.altair_chart(final_chart, use_container_width=True)
    st.metric(label="Area Covered", value=f"{df_polygon.iloc[0]['area_in_sq_kilometers']} km²")


# -- render_furthest_locations_chart form of a map diagram to show the furthest locations relative to the central point

def render_furthest_locations_chart(df_furthest: pd.DataFrame, center_point: dict):
    st.subheader("Furthest Locations from Top-Selling Hub")

    # Convert center_point to a DataFrame
    center_df = pd.DataFrame([{
        "longitude": center_point['coordinates'][0],
        "latitude": center_point['coordinates'][1]
    }])

    # Lines from center to each point
    line_data = pd.DataFrame([
        {
            "lat": center_point['coordinates'][1],
            "lon": center_point['coordinates'][0],
            "lat2": row['LATITUDE'],
            "lon2": row['LONGITUDE'],
            "distance_km": row['KILOMETER_FROM_TOP_SELLING_CENTER'],
            "location_name": row['LOCATION_NAME']
        } for _, row in df_furthest.iterrows()
    ])

    base = alt.Chart(line_data).encode(
        longitude='lon:Q',
        latitude='lat:Q',
        longitude2='lon2:Q',
        latitude2='lat2:Q',
        tooltip=['location_name:N', 'distance_km:Q']
    )

    line_chart = base.mark_line(stroke='firebrick').encode(
        strokeWidth=alt.Size('distance_km:Q', scale=alt.Scale(domain=[0, line_data['distance_km'].max()], range=[1, 5]))
    )

    point_chart = alt.Chart(df_furthest).mark_circle(size=80, color="orange").encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        tooltip=['LOCATION_NAME:N', 'KILOMETER_FROM_TOP_SELLING_CENTER:Q']
    )

    center_chart = alt.Chart(center_df).mark_point(shape='diamond', size=200, color='red')

    st.altair_chart((line_chart + point_chart + center_chart).project("identity").properties(
        width=700, height=500, title="Top 50 Furthest Locations from Hub"
    ), use_container_width=True)


# -- render_h3_resolution_chart generates a table that shows the underlying hex values for a particular resolution

def render_h3_resolution_chart(df_hex: pd.DataFrame, resolution: int):
    st.subheader(f"H3 Indexing at Resolution {resolution}")

    hex_col = f"H3_HEX_RESOLUTION_{resolution}"
    # st.write(f"hex_col: {hex_col}, df: {df_hex.columns}")
    if hex_col not in df_hex.columns:
        st.warning("Selected resolution column not found in dataframe.")
        return

    st.dataframe(df_hex[["LOCATION_ID", "LOCATION_NAME", hex_col]])
    st.info("H3 lets us generalize locations into reusable tiles for spatial analysis.")


# -- render_table_chart - traditional table data frame with little enhancements

def render_table_chart(df, block_id=None):
    st.dataframe(df)


# -- render_metric_chart - traditional metric display - enables you to select any numeric field in single row results for viewing

def render_metric_chart(df, block_id: str):

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found for metric display.")
        return

    selected_metric = st.selectbox(
        "Select a metric to highlight:",
        options=numeric_cols,
        key=f"metric_select_{block_id}"
    )

    # Assume first row holds the desired value
    metric_value = df[selected_metric].iloc[0] if not df.empty else None

    if metric_value is not None:
        st.metric(label=selected_metric.replace("_", " ").title(), value=f"{metric_value:,}")
    else:
        st.warning("No data available to display metric.")


# -- render_map_chart -- uses st.map() and creates a geographical mat, have to have LATITUDE/LONGITUDE

def render_map_chart(df, block_id=None):
    lat_cols = [col for col in df.columns if col.upper() in ("LATITUDE", "LAT")]
    lon_cols = [col for col in df.columns if col.upper() in ("LONGITUDE", "LON")]

    if not df.empty:
        # Let user pick a location to see details
        options = df.index if df.shape[0] > 1 else [df.index[0]]
        selected_idx = st.selectbox("Select a location to view details", options)
        row = df.loc[selected_idx]
        # Only show columns with H3 or location info
        info_cols = [col for col in df.columns if "h3" in col.lower() or "location" in col.lower()]
        st.markdown("#### Selected Location Details & H3 Indices")
        st.table(row[info_cols].to_frame().T)

    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        st.map(df[[lat_col, lon_col]].rename(columns={lat_col: "latitude", lon_col: "longitude"}))
    else:
        st.warning("Your data must include 'latitude' and 'longitude' columns to show a map.")


# -- render_network_chart - uses network (point to point graph) but needs LATITUDE_A / LONGITUDE A vs LATITUDE_B and LONGITUDE B for effective use

def render_network_chart(df, block_id: str):

    required_cols = ["LATITUDE_A", "LONGITUDE_A", "LATITUDE_B", "LONGITUDE_B"]
    if all(col in df.columns for col in required_cols):
        string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        label_col = st.selectbox("Select label (point name)", string_cols, index=0, key=f"label_{block_id}")
        metric_col = st.selectbox("Select metric (distance)", numeric_cols, index=0, key=f"metric_{block_id}")
        max_lines = st.slider("Maximum lines to show", min_value=10, max_value=len(df), value=50, step=5, key=f"slider_{block_id}")
        show_labels = st.checkbox("Show destination labels", value=False, key=f"show_labels_{block_id}")

        df_sorted = df.sort_values(by=metric_col, ascending=False).head(max_lines)

        # Create line segment DataFrame
        line_data = pd.DataFrame({
            "lat": df_sorted["LATITUDE_A"],
            "lon": df_sorted["LONGITUDE_A"],
            "lat2": df_sorted["LATITUDE_B"],
            "lon2": df_sorted["LONGITUDE_B"],
            "label": df_sorted[label_col],
            "metric": df_sorted[metric_col]
        })

        # Truncate long labels for overlay
        line_data["label_short"] = line_data["label"].str.slice(0, 30) + "…"

        # Compute hub (most connected point)
        hub_point = line_data.groupby("label")["metric"].sum().idxmax()
        hub_coords = line_data[line_data["label"] == hub_point][["lat", "lon"]].iloc[0]

        # Arrowheads layer
        arrow_layer = alt.Chart(line_data).mark_point(
            shape="triangle", angle=0, size=50, color="firebrick", opacity=0.8
        ).encode(
            longitude="lon2:Q",
            latitude="lat2:Q"
        )

        # Optional text labels at destination points
        label_layer = alt.Chart(line_data).mark_text(
            align="left", dx=4, dy=-4, fontSize=8, color="gray", opacity=0.6
        ).encode(
            longitude="lon2:Q",
            latitude="lat2:Q",
            text="label_short:N"
        )

        # Highlight hub
        hub_layer = alt.Chart(pd.DataFrame([{
            "lon": hub_coords["lon"],
            "lat": hub_coords["lat"],
            "label": hub_point
        }])).mark_point(
            shape="diamond", size=100, color="blue", opacity=0.5
        ).encode(
            longitude='lon:Q',
            latitude='lat:Q',
            tooltip=['label']
        )

        # Line chart with tooltip
        line_chart = alt.Chart(line_data).mark_line().encode(
            longitude='lon:Q',
            latitude='lat:Q',
            longitude2='lon2:Q',
            latitude2='lat2:Q',
            strokeWidth=alt.Size('metric:Q', scale=alt.Scale(domain=[0, df[metric_col].max()], range=[1, 10])),
            color=alt.Color('metric:Q', scale=alt.Scale(scheme="reds"), title=metric_col),
            tooltip=[
                alt.Tooltip('label:N', title="Connection"),
                alt.Tooltip('metric:Q', title="Distance")
            ]
        ).project('identity').properties(
            width=800,
            height=500,
            title="Network Distance Between Points"
        )

        # Layering logic
        layers = [line_chart, arrow_layer, hub_layer]
        if show_labels:
            layers.append(label_layer)

        final_chart = alt.layer(*layers).resolve_scale(
            color='independent', strokeWidth='independent'
        )

        st.altair_chart(final_chart, use_container_width=True)

        # CSV Export
        st.download_button("⬇️ Export Distance Data", line_data.to_csv(index=False), "distances.csv", mime="text/csv")

    else:
        st.warning("Your data must include LATITUDE_A, LONGITUDE_A, LATITUDE_B, LONGITUDE_B columns.")


# -- render_plot_chart - similar to map_chart also needs LATITUDE and LONGITUDE shows without map overlay but bubble chart with tooltips

def render_plot_chart(df, block_id):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    index_cols = df.select_dtypes(include=["object", "string", "datetime"]).columns.tolist()

    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        size_col = st.selectbox("Select column for point size", numeric_cols, key=f"map_size_{block_id}")
        default_tooltip = ["location_id"] if "location_id" in df.columns else []

        tooltip_cols = st.multiselect(
            "Select columns for tooltip",
            options=df.columns.tolist(),
            default=default_tooltip,
            key=f"map_tooltip_{block_id}"
        )

        chart = alt.Chart(df).mark_circle(size=100).encode(
            longitude='LONGITUDE:Q',
            latitude='LATITUDE:Q',
            size=alt.Size(f"{size_col}:Q", scale=alt.Scale(range=[10, 500])),
            color=alt.value("orange"),
            tooltip=tooltip_cols
        ).project('identity').properties(
            width=700,
            height=500,
            title='Location Map'
        )

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("Your data must include 'LATITUDE' and 'LONGITUDE' columns to show a map.")


# -- render_line_or_bar_chart - enables you to have line chart with up to 2 metrics shown (separate axis)

def render_line_or_bar_chart(df: pd.DataFrame, block_id: str, chart_type: str):


    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    index_candidates = df.select_dtypes(include=["object", "datetime", "string"]).columns.tolist()

    if numeric_cols and index_candidates:
        index_col = st.selectbox("Select x-axis (index)", index_candidates, key=f"index_{block_id}")
        value_col_1 = st.selectbox("Primary y-axis", numeric_cols, key=f"value1_{block_id}")

        secondary_options = ["None"] + [col for col in numeric_cols if col != value_col_1]
        value_col_2 = st.selectbox("Secondary y-axis (optional)", secondary_options, key=f"value2_{block_id}")

        slice_type = st.radio("Which rows to show?", ["Top 10", "All"], key=f"slice_{block_id}")
        df_sorted = df.sort_values(by=value_col_1, ascending=False)
        df_to_plot = df_sorted.head(10) if slice_type == "Top 10" else df_sorted

        x_dtype = df_to_plot[index_col].dtype

        if pd.api.types.is_datetime64_any_dtype(x_dtype):
            x_type = 'temporal'
        elif pd.api.types.is_numeric_dtype(x_dtype):
            x_type = 'quantitative'
        else:
            x_type = 'nominal'  # most likely the case
        
        base = alt.Chart(df_to_plot).encode(
            x=alt.X(f"{index_col}:N", axis=alt.Axis(title=index_col), type=x_type)
        )

        mark_fn = alt.Chart.mark_bar if chart_type == "Bar Chart" else alt.Chart.mark_line

        chart1 = base.mark_line(color="steelblue").encode(
            y=alt.Y(f"{value_col_1}:Q", axis=alt.Axis(title=value_col_1))
        ) if chart_type == "Line Chart" else base.mark_bar(color="steelblue").encode(
            y=alt.Y(f"{value_col_1}:Q", axis=alt.Axis(title=value_col_1))
        )

        if value_col_2 != "None":
            chart2 = base.mark_line(color="firebrick").encode(
                y=alt.Y(f"{value_col_2}:Q", axis=alt.Axis(title=value_col_2))
            ) if chart_type == "Line Chart" else base.mark_bar(color="firebrick").encode(
                y=alt.Y(f"{value_col_2}:Q", axis=alt.Axis(title=value_col_2))
            )

            chart = alt.layer(chart1, chart2).resolve_scale(y='independent')
            st.altair_chart(chart, use_container_width=True)
        else:
            st.altair_chart(chart1, use_container_width=True)
    else:
        st.warning("Need at least one numeric and one string/date column to plot.")


# --- Helper Utilities

# -- parse_point_wkt - converts a string which is really a co-ordinates dictionary into proper floated values for use by the chart

def parse_point_wkt(wkt: str):
    # Expects format "POINT (longitude latitude)"
    #st.write("hi")
    nums = wkt.replace("POINT (", "").replace(")", "").split()
    return {"coordinates": [float(nums[0]), float(nums[1])]}

# -- extract_variables_from_sql - creates a dictionary of all variables that have been encoded in SQL blocks using {VAR}        
def extract_variables_from_sql(sql_block):
    # Finds all {VARNAME} patterns in the SQL string (non-greedy, avoids nested braces)
    return re.findall(r"\{([A-Za-z0-9_]+)\}", sql_block)

# -- overview_to_html --  Build HTML for overview as bullet points if it uses - or *

def overview_to_html(overview):
    # If overview is a markdown bullet list, convert to <ul><li>
    lines = [line.strip() for line in overview.strip().splitlines() if line.strip()]
    if all(line.startswith("- ") or line.startswith("* ") for line in lines):
        items = [line[2:].strip() for line in lines]
        bullets = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul>{bullets}</ul>"
    else:
        # Otherwise, just render as paragraphs
        return "<br>".join(lines)

# -- resolve_variables -- Resolve all variables so we can do more sophisticated substitution within SQL queries

def resolve_variables(variable_names, session_state, global_variables={}, defaults={}):
    resolved = {}
    #st.write(f"gv: {global_variables}, variables: {variable_names}")
    for var in variable_names:
        if var in session_state:
            resolved[var] = session_state[var]
        elif var.lower() in session_state:
            resolved[var] = session_state[var.lower()]
        elif var.upper() in session_state:
            resolved[var] = session_state[var.upper()]
        elif var in global_variables:
            resolved[var] = global_variables[var]
        elif var in defaults:
            resolved[var] = defaults[var]
        else:
            resolved[var] = ""  # Or raise, warn, etc.
    #st.write(f"resolved: {resolved}")
    return resolved


# ========================================
# CORTEX-SPECIFIC RENDERING FUNCTIONS
# ========================================

def parse_cortex_complete_sql(sql_code: str) -> dict:
    """
    Parse CORTEX.COMPLETE SQL to extract model, prompts, and parameters.
    Returns a dictionary with extracted parameters.
    """
    try:
        # Extract model name (first parameter)
        model_match = re.search(r"CORTEX\.COMPLETE\s*\(\s*'([^']+)'", sql_code, re.IGNORECASE)
        model = model_match.group(1) if model_match else 'llama3.1-8b'
        
        # Extract system prompt
        system_match = re.search(r"'role':\s*'system'[^}]*'content':\s*'([^']+)'", sql_code)
        system_prompt = system_match.group(1) if system_match else 'You are a helpful assistant.'
        
        # Extract user prompt
        user_match = re.search(r"'role':\s*'user'[^}]*'content':\s*'([^']+)'", sql_code)
        user_prompt = user_match.group(1) if user_match else 'Please analyze the data.'
        
        # Extract temperature
        temp_match = re.search(r"'temperature':\s*([0-9.]+)", sql_code)
        temperature = float(temp_match.group(1)) if temp_match else 0.7
        
        # Extract max_tokens
        tokens_match = re.search(r"'max_tokens':\s*([0-9]+)", sql_code)
        max_tokens = int(tokens_match.group(1)) if tokens_match else 4000
        
        return {
            'model': model,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
    except Exception as e:
        st.error(f"Error parsing Cortex Complete SQL: {e}")
        return {
            'model': 'llama3.1-8b',
            'system_prompt': 'You are a helpful assistant.',
            'user_prompt': 'Please analyze the data.',
            'temperature': 0.7,
            'max_tokens': 4000
        }


def render_cortex_complete_block(session, block_id: str, query_code: str, talk_track: str):
    """
    Render an interactive Cortex Complete block with editable parameters.
    """
    st.markdown("### 🤖 Interactive Cortex Complete")
    st.markdown("*Experiment with different prompts and parameters to see how they affect the AI's responses*")
    
    # Parse the original SQL to get default values
    params = parse_cortex_complete_sql(query_code)
    
    # Create two columns for the interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**🎛️ Parameters**")
        
        # Model selection
        model_options = ['llama3.1-8b', 'llama3.1-70b', 'mixtral-8x7b', 'mistral-large']
        selected_model = st.selectbox(
            "Model:", 
            model_options, 
            index=model_options.index(params['model']) if params['model'] in model_options else 0,
            key=f"model_{block_id}"
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=params['temperature'],
            step=0.1,
            key=f"temp_{block_id}",
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        
        # Max tokens
        max_tokens = st.number_input(
            "Max Tokens:", 
            min_value=10, 
            max_value=8000, 
            value=params['max_tokens'],
            step=10,
            key=f"tokens_{block_id}"
        )
    
    with col2:
        st.markdown("**💬 Prompts**")
        
        # System prompt
        system_prompt = st.text_area(
            "System Prompt:", 
            value=params['system_prompt'],
            height=100,
            key=f"system_{block_id}",
            help="Instructions that define the AI's role and behavior"
        )
        
        # User prompt
        user_prompt = st.text_area(
            "User Prompt:", 
            value=params['user_prompt'],
            height=100,
            key=f"user_{block_id}",
            help="The specific task or question for the AI"
        )
    
    # Build the dynamic SQL
    dynamic_sql = f"""
    SELECT 
        SNOWFLAKE.CORTEX.COMPLETE(
            '{selected_model}',
            [
                {{'role': 'system', 'content': '{system_prompt}'}},
                {{'role': 'user', 'content': '{user_prompt}'}}
            ],
            {{'temperature': {temperature}, 'max_tokens': {max_tokens}}}
        ):choices[0]:messages::string as ai_response,
        '{selected_model}' as model_used,
        {temperature} as temperature_used,
        {max_tokens} as max_tokens_used
    """
    
    # Show the generated SQL
    with st.expander("🔍 Generated SQL", expanded=False):
        st.code(dynamic_sql, language="sql")
    
    # Run button
    if st.button("🚀 Run Cortex Complete", key=f"run_{block_id}"):
        with st.spinner("Running Cortex Complete..."):
            try:
                df = session.sql(dynamic_sql).to_pandas()
                st.session_state[block_id] = df
                
                if not df.empty:
                    st.success("✅ Cortex Complete executed successfully!")
                    
                    # Display the AI response
                    st.markdown("### 🎯 AI Response")
                    response = df.iloc[0]['AI_RESPONSE']
                    st.markdown(response)
                    
                    # Show metadata
                    with st.expander("📊 Execution Metadata"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", df.iloc[0]['MODEL_USED'])
                        with col2:
                            st.metric("Temperature", df.iloc[0]['TEMPERATURE_USED'])
                        with col3:
                            st.metric("Max Tokens", df.iloc[0]['MAX_TOKENS_USED'])
                else:
                    st.warning("No results returned.")
                    
            except Exception as e:
                st.error(f"Error executing Cortex Complete: {e}")
    
    # Show results if available
    if block_id in st.session_state:
        df = st.session_state[block_id]
        if not df.empty:
            st.markdown("### 📋 Full Results")
            render_table_chart(df)
    
    # Talk track
    if talk_track:
        with st.expander("🎤 Talk Track", expanded=False):
            st.markdown(talk_track)
    
    st.markdown("---")


def parse_cortex_search_sql(sql_code: str) -> dict:
    """
    Parse Cortex Search SQL to extract parameters for interactive interface.
    """
    import re
    import json
    
    # Default values
    search_service = ""
    search_query = ""
    columns = []
    limit = 10
    
    try:
        # Find SEARCH_PREVIEW call
        search_pattern = r'SNOWFLAKE\.CORTEX\.SEARCH_PREVIEW\s*\(\s*[\'"]([^\'\"]+)[\'"],\s*[\'"]([^\'\"]+)[\'"]'
        match = re.search(search_pattern, sql_code, re.IGNORECASE)
        
        if match:
            search_service = match.group(1)
            # Try to parse the JSON parameter
            json_param = match.group(2)
            
            # Handle case where JSON is in the string
            if json_param.startswith('{'):
                try:
                    params = json.loads(json_param)
                    search_query = params.get('query', '')
                    columns = params.get('columns', [])
                    limit = params.get('limit', 10)
                except:
                    # If JSON parsing fails, treat as simple query
                    search_query = json_param
            else:
                search_query = json_param
        
        # Extract columns from SELECT part if not in JSON
        if not columns:
            # Look for result.value:columnname patterns
            column_pattern = r'result\.value:([^:]+)::string'
            column_matches = re.findall(column_pattern, sql_code)
            columns = [col for col in column_matches if col != 'full_result_json']
        
        # Extract limit from LIMIT clause if not in JSON
        if limit == 10:  # default
            limit_pattern = r'[\'"]limit[\'"]:\s*(\d+)'
            limit_match = re.search(limit_pattern, sql_code)
            if limit_match:
                limit = int(limit_match.group(1))
                
    except Exception as e:
        pass  # Use defaults
    
    return {
        'search_service': search_service,
        'search_query': search_query,
        'columns': columns,
        'limit': limit
    }


def render_cortex_search_block(session, block_id: str, query_code: str, talk_track: str, search_service: str):
    """
    Render an interactive Cortex Search block with search capabilities.
    """
    st.markdown("### 🔍 Interactive Cortex Search")
    st.markdown("*Search through your data using natural language queries*")
    
    # Parse the original SQL to get default values
    params = parse_cortex_search_sql(query_code)
    
    # Use search_service parameter if provided, otherwise use parsed value
    if search_service:
        params['search_service'] = search_service
    
    if not params['search_service']:
        st.error("No search service specified. Please add 'cortex_search_service' to your YAML configuration.")
        return
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query:", 
            value=params['search_query'],
            placeholder="e.g., 'benny the bull' or 'parking issues'",
            key=f"search_{block_id}"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        search_button = st.button("🔍 Search", key=f"search_btn_{block_id}")
    
    # Additional search parameters
    with st.expander("🎛️ Search Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            limit = st.number_input("Result Limit:", min_value=1, max_value=100, value=params['limit'], key=f"limit_{block_id}")
        with col2:
            columns_default = ", ".join(params['columns']) if params['columns'] else ""
            columns_input = st.text_input("Columns to Return:", value=columns_default, placeholder="e.g., aggregate_comment, game_experience_score", key=f"cols_{block_id}")
    
    # Build and execute search
    if search_button and search_query:
        # Parse columns input
        columns_list = [col.strip() for col in columns_input.split(',') if col.strip()] if columns_input else []
        
        # Build JSON parameter
        search_json = {
            "query": search_query,
            "limit": limit
        }
        if columns_list:
            search_json["columns"] = columns_list
        
        # Build the search SQL with proper structure
        search_sql = f"""WITH search_results AS (
    SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        '{params['search_service']}',
        '{json.dumps(search_json)}'
    ) as search_result
)
SELECT"""
        
        # Add column selections
        if columns_list:
            for col in columns_list:
                search_sql += f"""
    result.value:{col}::string as {col},"""
        
        search_sql += """
    result.value as full_result_json,
    result.index + 1 as result_rank
FROM search_results,
LATERAL FLATTEN(input => PARSE_JSON(search_results.search_result):results) as result
ORDER BY result.index;"""
        
        with st.expander("🔍 Generated Search SQL", expanded=False):
            st.code(search_sql, language="sql")
        
        with st.spinner("Searching..."):
            try:
                df = session.sql(search_sql).to_pandas()
                st.session_state[block_id] = df
                
                if not df.empty:
                    st.success(f"✅ Found {len(df)} results for: '{search_query}'")
                    
                    # Display search results
                    st.markdown("### 🎯 Search Results")
                    
                    # Show first few results expanded
                    for i, row in df.head(5).iterrows():
                        with st.expander(f"Result {int(row['RESULT_RANK'])}", expanded=i < 3):
                            # Show specific columns if available
                            if columns_list:
                                for col in columns_list:
                                    col_upper = col.upper()
                                    if col_upper in row and pd.notna(row[col_upper]):
                                        st.markdown(f"**{col}:** {row[col_upper]}")
                            
                            # Show full JSON with checkbox toggle
                            if 'FULL_RESULT_JSON' in row:
                                show_json = st.checkbox("Show Full JSON", key=f"json_{block_id}_{i}")
                                if show_json:
                                    st.json(row['FULL_RESULT_JSON'])
                        
                else:
                    st.warning("No results found.")
                    
            except Exception as e:
                st.error(f"Error executing search: {e}")
                st.code(search_sql, language="sql")
    
    # Show full results table
    if block_id in st.session_state:
        df = st.session_state[block_id]
        if not df.empty:
            with st.expander("📊 Full Results Table", expanded=False):
                render_table_chart(df)
    
    # Talk track
    if talk_track:
        with st.expander("🎤 Talk Track", expanded=False):
            st.markdown(talk_track)
    
    st.markdown("---")


def render_cortex_analyst_block(session, block_id: str, query_code: str, talk_track: str, semantic_model_file: str):
    """
    Render an interactive Cortex Analyst block using the proper API approach.
    """
    st.markdown("### 🧠 Interactive Cortex Analyst")
    st.markdown("*Ask natural language questions and see the AI-generated SQL and results*")
    
    if not semantic_model_file:
        st.error("No semantic model file specified. Please add 'semantic_model_file' to your YAML configuration.")
        return
    
    # Initialize session state for chat history
    if f"messages_{block_id}" not in st.session_state:
        st.session_state[f"messages_{block_id}"] = []
    
    # Load and display semantic model
    with st.expander("📋 Semantic Model", expanded=False):
        try:
            # Load semantic model from stage
            semantic_df = session.sql(f"SELECT $1 FROM {semantic_model_file}").to_pandas()
            if not semantic_df.empty:
                semantic_content = "\n".join([row for row in semantic_df.iloc[:, 0].tolist() if row is not None])
                st.code(semantic_content, language="yaml")
            else:
                st.warning("Semantic model file is empty or not found.")
        except Exception as e:
            st.error(f"Error loading semantic model: {e}")
    
    # Sample questions for inspiration
    st.markdown("**💡 Sample Questions:**")
    sample_questions = [
        "What are the top 5 records by revenue?",
        "Show me trends over the last quarter", 
        "Which segments perform best?",
        "What is the average score by category?",
        "How many records do we have in total?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(f"📝 {question}", key=f"sample_{block_id}_{i}"):
                st.session_state[f"active_question_{block_id}"] = question
    
    # Question interface
    default_question = st.session_state.get(f"active_question_{block_id}", "")
    question = st.text_area(
        "Ask a Question:", 
        value=default_question,
        placeholder="e.g., 'What are the top 5 products by revenue this quarter?'",
        height=100,
        key=f"question_{block_id}"
    )
    
    # Clear the active question after using it
    if f"active_question_{block_id}" in st.session_state:
        del st.session_state[f"active_question_{block_id}"]
    
    # Additional parameters
    with st.expander("🎛️ Analyst Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            include_sql = st.checkbox("Show Generated SQL", value=True, key=f"show_sql_{block_id}")
        with col2:
            auto_execute = st.checkbox("Auto-execute Generated SQL", value=True, key=f"auto_exec_{block_id}")
    
    # Helper function to send message to Cortex Analyst API
    def send_analyst_message(prompt: str) -> dict:
        """Send a message to Cortex Analyst API and return the response."""
        try:
            import _snowflake
            
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "semantic_model_file": semantic_model_file,
            }
            
            resp = _snowflake.send_snow_api_request(
                "POST",
                f"/api/v2/cortex/analyst/message",
                {},
                {},
                request_body,
                {},
                30000,
            )
            
            if resp["status"] < 400:
                return json.loads(resp["content"])
            else:
                raise Exception(f"API request failed with status {resp['status']}: {resp}")
                
        except Exception as e:
            st.error(f"Error calling Cortex Analyst API: {e}")
            return None
    
    # Analyze button
    if st.button("🧠 Analyze", key=f"analyze_{block_id}") and question:
        with st.spinner("Analyzing question..."):
            try:
                # Call Cortex Analyst API
                response = send_analyst_message(question)
                
                if response and "message" in response:
                    content = response["message"]["content"]
                    
                    # Add to chat history
                    st.session_state[f"messages_{block_id}"].append({
                        "role": "user", 
                        "content": [{"type": "text", "text": question}]
                    })
                    st.session_state[f"messages_{block_id}"].append({
                        "role": "assistant", 
                        "content": content
                    })
                    
                    # Process and display the response
                    for item in content:
                        if item["type"] == "text":
                            st.markdown(f"**AI Response:** {item['text']}")
                        
                        elif item["type"] == "sql":
                            generated_sql = item["statement"]
                            
                            if include_sql:
                                with st.expander("🔍 Generated SQL", expanded=False):
                                    st.code(generated_sql, language="sql")
                            
                            if auto_execute:
                                st.markdown("### 📊 Results")
                                with st.spinner("Executing generated SQL..."):
                                    try:
                                        results_df = session.sql(generated_sql.strip(";")).to_pandas()
                                        st.session_state[block_id] = results_df
                                        
                                        if not results_df.empty:
                                            st.success(f"✅ Query executed successfully! ({len(results_df)} rows)")
                                            
                                            # Show results with charts (similar to the example)
                                            if len(results_df) > 1:
                                                data_tab, line_tab, bar_tab = st.tabs(["Data", "Line Chart", "Bar Chart"])
                                                
                                                with data_tab:
                                                    st.dataframe(results_df)
                                                
                                                if len(results_df.columns) > 1:
                                                    chart_df = results_df.set_index(results_df.columns[0])
                                                    
                                                    with line_tab:
                                                        try:
                                                            st.line_chart(chart_df)
                                                        except Exception as e:
                                                            st.write(f"Line chart not available: {e}")
                                                    
                                                    with bar_tab:
                                                        try:
                                                            st.bar_chart(chart_df)
                                                        except Exception as e:
                                                            st.write(f"Bar chart not available: {e}")
                                            else:
                                                st.dataframe(results_df)
                                        else:
                                            st.warning("Query executed but returned no results.")
                                            
                                    except Exception as e:
                                        st.error(f"Error executing generated SQL: {e}")
                                        st.code(generated_sql, language="sql")
                        
                        elif item["type"] == "suggestions":
                            with st.expander("💡 Suggestions", expanded=True):
                                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                                    if st.button(suggestion, key=f"suggestion_{block_id}_{suggestion_index}"):
                                        st.session_state[f"active_question_{block_id}"] = suggestion
                                        st.experimental_rerun()
                else:
                    st.error("No valid response from Cortex Analyst API.")
                    
            except Exception as e:
                st.error(f"Error with Cortex Analyst: {e}")
    
    # Display chat history
    if st.session_state[f"messages_{block_id}"]:
        st.markdown("### 💬 Chat History")
        for message_index, message in enumerate(st.session_state[f"messages_{block_id}"]):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"][0]["text"])
                else:
                    # Display assistant content
                    for item in message["content"]:
                        if item["type"] == "text":
                            st.markdown(item["text"])
                        elif item["type"] == "sql":
                            with st.expander("SQL Query", expanded=False):
                                st.code(item["statement"], language="sql")
    
    # Show results if available
    if block_id in st.session_state:
        df = st.session_state[block_id]
        if not df.empty:
            with st.expander("📊 Full Results Table", expanded=False):
                render_table_chart(df)
    
    # Talk track
    if talk_track:
        with st.expander("🎤 Talk Track", expanded=False):
            st.markdown(talk_track)
    
    st.markdown("---")

# ---- MAIN STARTS HERE
# ----  THE ACTUAL CODE VS THE HELPER FUNCTIONS

stage_base = f"@{db_name}.CONFIGS.FRAMEWORK_YAML_STAGE"
stage = None

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 400px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---



with st.sidebar:

    snowflake_logo_url='https://storage.googleapis.com/lee-demo-streamlit-icon-bucket/Snowflake_Logo.png'
    html_block = f"""
    <div style="text-align: center;">
        <img src="{snowflake_logo_url}" width="120" style="margin-bottom: 10px;" />
        <h2 style="margin-bottom: 0;">Snowflake Demo Runner</h2>
    </div>
    """
    
    st.markdown(html_block, unsafe_allow_html=True)


    folders = list_yaml_folders_in_stage(session, stage_base)
    if not folders:
        st.warning("No folders/areas found in this stage.")
        st.stop()
    
    # Insert a placeholder at the top
    folders_display = ["[Select an Area]"] + folders
    selected_index = st.selectbox("Select Area", range(len(folders_display)), format_func=lambda i: folders_display[i])
    if selected_index == 0:
        st.info("Please select an area to view demos.")
        st.stop()
    
    selected_folder = folders[selected_index - 1]  # Adjust for the placeholder
    stage = f"{stage_base}/{selected_folder}"

    # Reset active demo when area changes
    if "current_area" not in st.session_state:
        st.session_state.current_area = None
    
    if st.session_state.current_area != selected_folder:
        st.session_state.current_area = selected_folder
        if "active_demo" in st.session_state:
            del st.session_state.active_demo  # Clear active demo when switching areas

    if stage:
        yaml_files = list_yaml_files_in_stage(stage)
    
        if not yaml_files:
            st.warning("No demo YAML files found in the stage.")
            st.stop()
    
        # Let user scroll/select demos
        
        selected_index = st.selectbox(
            "Select a Demo",
            range(len(yaml_files)),
            format_func=lambda i: yaml_files[i].split("/")[-1],
            key="yaml_index"
        )
        selected_file = yaml_files[selected_index]
        demo_yaml = load_yaml_from_stage(stage, selected_file)
        # Show demo info as you scroll/select (NO YAML RAW)
        if demo_yaml and "demo" in demo_yaml:
            show_demo_info(demo_yaml)
        else:
            st.warning("Demo info missing or invalid YAML.")
    
        # Button to LAUNCH the demo
        launch_demo = st.button("Run Demo", key="launch_demo")

# --- MAIN PANEL ---
if "active_demo" not in st.session_state:
    st.session_state.active_demo = None

# Only set active_demo when Run Demo is clicked
if launch_demo:
    st.session_state.active_demo = selected_file

active_demo_file = st.session_state.active_demo

if active_demo_file and stage:
    # Always reload YAML for the active demo
    demo_yaml = load_yaml_from_stage(stage, active_demo_file)
    if not demo_yaml or "demo" not in demo_yaml:
        st.error("Could not load the selected demo YAML.")
        st.stop()
    
    # Extract variables for use in steps (these are not for dynamic substitution within SQL step code)
    demo = demo_yaml["demo"]
    logo_url = demo.get("logo_url", "")
    DATABASE = demo.get("database", "")
    SCHEMA = demo.get("schema", "")
    QUERY_TAG = demo.get("query_tag")
    WAREHOUSE = demo.get("warehouse")
    ROLE = demo.get("role")
    cleanup_commands = demo.get("cleanup_commands", [])
    init_commands = demo.get("init_commands",[])
    overview = demo.get("overview","")
    overview_html = overview_to_html(overview)
    topic = demo.get("topic","")
    subtopic = demo.get("sub_topic")
    tertiarytopic = demo.get("tertiary_topic")
    title = " ".join(x for x in [topic, subtopic, tertiarytopic] if x)
    subtitle=demo.get("title","")

    # -- for when you want to substitute within SQL blocks global variables like GL_DATABASE as an example

    defaults = {}
    global_variables = {f"GV_{k.upper()}": v for k, v in demo.items()}


    # -- silent initialization 
    if init_commands:
        # Substitute variables in each init command
        substituted_init_cmds = substitute_vars(init_commands, global_variables)
        run_sql_batch(session, substituted_init_cmds)

        
    # Construct the HTML block
    html_block = f"""
    <div style="text-align: center;">
        <img src="{logo_url}" width="120" style="margin-bottom: 10px;" />
        <h2 style="margin-bottom: 0;">{title}</h2>
        <h2 style="margin-bottom: 0;">{subtitle}</h2>
        {overview_html}
    </div>
    """
    
    st.markdown(html_block, unsafe_allow_html=True)

    # --- Show Steps / Main Demo ---
    steps = demo_yaml.get("steps", [])
    if not steps:
        st.warning("No steps found in this demo!")
    else:
        for step in steps:
            sql = step.get("query", "")
            variable_names = extract_variables_from_sql(sql)
            resolved_vars = resolve_variables(variable_names, st.session_state, global_variables, defaults)
            # Now substitute:
            # sql_substituted = sql.format(**resolved_vars)
            # Now you can run sql_substituted
            # 1. Gather variable fields needed for this block
            #st.write("DEBUG variable_names:", resolved_vars)

            render_query_block(
                session,
                block_id=step.get("id", ""),
                title=step.get("title", ""),
                query_code=step.get("query", ""),
                talk_track=step.get("talk_track", ""),
                instructions=step.get("instructions", ""),
                instructions_title=step.get("instructions_title", ""),
                save_as=step.get("save_as", None),
                variable_fields=resolved_vars,
                default_chart=step.get("default_chart", None),
                cortex_type=step.get("cortex_type", None),
                cortex_search_service=step.get("cortex_search_service", None),
                semantic_model_file=step.get("semantic_model_file", None)
            )

    # Call this at the end of your app if cleanup is needed
        if st.button("Reset Environment") and cleanup_commands:
            substituted_cleanup_commands = substitute_vars(cleanup_commands, global_variables)
            run_sql_batch(session, substituted_cleanup_commands)
            # Clear Streamlit session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            #st.rerun()  # Force app to reload
            st.success("Environment reset completed.")

else:
    # Show helpful message when no demo is active
    st.info("👈 Select an area and demo from the sidebar, then click 'Run Demo' to begin.")
            
 