import streamlit as st
import pandas as pd
import yaml
import json
from typing import List, Dict, Any, Tuple, Optional

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except Exception:
    session = None


# =========================================================================================================
# ENHANCED TAB GENERATION FUNCTION - Creates full tab suite instead of just 3 basic tabs
# =========================================================================================================

def generate_enhanced_tabs(dimensions: List[str], metric_defs: List[Dict], time_col: str, default_grain: str, default_cards: List[str]) -> List[Dict]:
    """
    Generate enhanced tab suite including all missing functionality
    Creates 10+ tabs instead of just 3 basic ones
    """
    
    # Handle empty selections gracefully - provide basic tabs even with no selections
    safe_dimensions = dimensions if dimensions else []
    safe_metrics = metric_defs if metric_defs else []
    
    
    # Detect data characteristics for intelligent tab generation
    sentiment_dimensions = [col for col in safe_dimensions if "SENTIMENT" in col]
    comment_dimensions = [col for col in safe_dimensions if "COMMENT" in col]  
    recommendation_dimensions = [col for col in safe_dimensions if "RECOMMENDATION" in col]
    
    tabs = []
    
    # ALWAYS CREATE BASIC TABS (even with no selections) - user can configure later
    
    # 1. Overview (first - executive dashboard)
    tabs.append({
        "key": "overview", 
        "type": "overview", 
        "title": "Overview", 
        "metric_cards": {"metrics": default_cards, "comparisons": ["mom","yoy"]}, 
        "grid": {"type": "rank_grid", "dimension_selector": safe_dimensions, "metric_source": "selected_card", "top_n_default": 50, "sort": "desc"},
        "timeseries": {"type": "line", "metric_source": "selected_card", "default_grain": default_grain, "grains_allowed": ["day","week","month","quarter","year"]}
    })
    
    # 2. Product / Category (second - entity deep dive)
    tabs.append({
        "key": "product",
        "type": "product", 
        "title": "Product / Category",
        "entity_dimensions": safe_dimensions,
        "asset_url_field": "",
        "metrics_allowed": [m["key"] for m in safe_metrics]
    })
    
    # 3. VS Comparison (third - head-to-head analysis)
    # Always create VS tab, even if no dimensions selected yet
    # Choose default compare dimension if available (first selected dimension)
    default_dim = safe_dimensions[0] if safe_dimensions else None

    vs_tab = {
            "key": "vs",
            "type": "compare",
            "title": "VS",
        "selectable_dimensions": safe_dimensions if safe_dimensions else [],
        "metrics_allowed": [m["key"] for m in safe_metrics] if safe_metrics else [],
            "asset_url_field": "",
            "timeseries": {
                "type": "line",
                "default_grain": default_grain
            }
        }
    if default_dim:
        vs_tab["dimension"] = default_dim
    tabs.append(vs_tab)
    
    # 4. Top N (fourth - ranked analysis)
    tabs.append({
        "key": "topn", 
        "type": "topn", 
        "title": "Top N", 
        "split_dimensions": safe_dimensions if safe_dimensions else [],
        "metric_source": "selected_card", 
        "n_options": [5,10,25,50]
    })
    
    # 5. Self Service (fifth - flexible analysis)
    tabs.append({
        "key": "self_service", 
        "type": "self_service", 
        "title": "Self Service", 
        "selectable_dimensions": safe_dimensions if safe_dimensions else [],
        "selectable_metrics": [m["key"] for m in safe_metrics] if safe_metrics else []
    })
    
    # 6. Search (if available - placeholder, will be replaced with smart version)
    tabs.append({
        "key": "search",
        "type": "search", 
        "title": "Search",
        "cortex_search_service": "",
        "default_limit": 25,
        "examples": ["high value consumers", "mobile users", "premium segments"]
    })
    
    # 7. Analyst (always available)
    tabs.append({
        "key": "analyst",
        "type": "analyst",
        "title": "AI Assistant", 
        "examples": [
            "What are the top performing segments?",
            "Compare engagement across demographics",
            "Show trends over time"
        ]
    })
    
    # 8. Raw Data (ALWAYS LAST - for demo purposes)
    tabs.append({
        "key": "raw_data",
        "type": "raw_data",
        "title": "🗂️ Raw Data",
        "description": "Explore the underlying table structure and data for demo purposes"
    })
    
    return tabs


def generate_search_tab(dimensions: List[str], metric_defs: List[Dict], search_service: str = "", search_columns: List[str] = []) -> Dict:
    """Generate search tab configuration with proper column validation"""
    
    # 🚨 KEY INSIGHT: Search services index RAW TABLE COLUMNS, not calculated metrics!
    if search_columns:
        # Use ONLY the actual indexed columns from the search service
        search_cols_lower = [col.lower() for col in search_columns]
        
        # Find comment columns in the indexed columns
        comment_cols = [col for col in search_columns if "comment" in col.lower()]
        score_cols = [col for col in search_columns if "score" in col.lower()]
        id_cols = [col for col in search_columns if col.lower() in ["id", "pk", "key"]]
        
        # Use the actual indexed columns - DON'T mix with calculated metrics
        default_columns = search_columns[:12]  # Use first 12 indexed columns
        
        # Set up roles using actual indexed column names
        roles = {
            "id": id_cols[0] if id_cols else search_columns[0],
            "label": next((col for col in search_columns if col.lower() in ["segment", "name", "label", "title"]), search_columns[1] if len(search_columns) > 1 else "label"),
            "primary_field": score_cols[0] if score_cols else next((col for col in search_columns if "score" in col.lower() or "rating" in col.lower()), search_columns[0]),
            "comment": comment_cols[0] if comment_cols else "",
            "dimensions": [col for col in search_columns if col not in id_cols and col not in comment_cols and col not in score_cols][:5]
        }
        
        # Generate realistic examples based on actual columns
        examples = []
        if comment_cols:
            examples.append(f"Issues with {comment_cols[0].lower().replace('_', ' ')}")
        if score_cols:
            examples.append(f"High {score_cols[0].lower().replace('_', ' ')}")
        examples.append("Recent feedback")
        
    else:
        # Fallback when no search service info available
        comment_dimensions = [col for col in dimensions if "COMMENT" in col]
        default_columns = dimensions + [m["key"] for m in metric_defs]
        
        roles = {
            "id": dimensions[0] if dimensions else "id",
            "label": dimensions[1] if len(dimensions) > 1 else (dimensions[0] if dimensions else "label"), 
            "primary_field": metric_defs[0]["key"] if metric_defs else "score",
            "comment": comment_dimensions[0] if comment_dimensions else "",
            "dimensions": dimensions[:5] if dimensions else []
        }
        
        examples = [
            f"High performing {dimensions[0].lower().replace('_', ' ')}" if dimensions else "High performers",
            f"Issues with {comment_dimensions[0].lower().replace('_', ' ')}" if comment_dimensions else "Common issues", 
            "Recent trends"
        ]
    
    return {
        "key": "search",
        "type": "search", 
        "title": "Search",
        "cortex_search_service": search_service,
        "default_limit": 25,
        "default_columns": default_columns,
        "roles": roles,
        "examples": examples
    }

    # 6. Search (sixth - intelligent search) - NOW USES SMART COLUMN DETECTION
    tabs.append({
        "key": "search",
        "type": "search",
        "title": "Search", 
        "cortex_search_service": "",  # Will be populated by generate_search_tab function
        "default_limit": 25,
        "default_columns": (dimensions + [m["key"] for m in metric_defs]) if dimensions or metric_defs else ["PLACEHOLDER"],
        "roles": {
            "id": dimensions[0] if dimensions else "id",
            "label": dimensions[1] if len(dimensions) > 1 else (dimensions[0] if dimensions else "label"),
            "primary_field": metric_defs[0]["key"] if metric_defs else "score", 
            "comment": comment_dimensions[0] if comment_dimensions else "",
            "dimensions": dimensions if dimensions else []
        },
        "examples": [
            f"Top performing {dimensions[0].lower().replace('_', ' ')}" if dimensions else "High performers",
            f"Issues with {comment_dimensions[0].lower().replace('_', ' ')}" if comment_dimensions else "Common issues", 
            "Recent trends"
        ]
    })
    
    # 7. Analyst (seventh - AI assistant)
    tabs.append({
        "key": "analyst",
        "type": "analyst", 
        "title": "Analyst",  # Keep simple title like your manual
        "semantic_model_file": "",
        "examples": [
            f"What are the top 10 {dimensions[0].lower().replace('_', ' ')} by {metric_defs[0]['key'].lower().replace('_', ' ')}?" if dimensions and metric_defs else "What are the top performers?",
            f"Show {metric_defs[1]['key'].lower().replace('_', ' ')} trends over time" if len(metric_defs) > 1 else "Show performance trends over time", 
            f"Compare {dimensions[0].lower().replace('_', ' ')} across {dimensions[1].lower().replace('_', ' ')}" if len(dimensions) > 1 else "Compare categories",
            "What patterns do you see in the data?",
            "Which segments show the strongest performance?"
        ]
    })
    
    return tabs

def generate_enhanced_tabs_with_search(dimensions: List[str], metric_defs: List[Dict], time_col: str, default_grain: str, default_cards: List[str], search_service: str = "", search_columns: List[str] = []) -> List[Dict]:
    """
    Generate tabs with proper search service integration and column validation
    """
    # Get base tabs in correct order
    tabs = generate_enhanced_tabs(dimensions, metric_defs, time_col, default_grain, default_cards)
    
    # Safety check - ensure tabs is always a list
    if tabs is None:
        tabs = []
    
    # Replace the search tab with smart column-aware version
    search_tab = generate_search_tab(dimensions, metric_defs, search_service, search_columns)
    
    # Replace the default search tab with the smart one if search service is available
    if search_service and search_tab:
        for i, tab in enumerate(tabs):
            if tab.get("key") == "search":
                tabs[i] = search_tab
                break
    else:
        # Remove search tab if no search service
        tabs = [tab for tab in tabs if tab.get("key") != "search"]
    
    return tabs

@st.cache_data(ttl=300)  # Cache for 5 minutes
def detect_cortex_search_services(database: str, schema: str) -> List[str]:
    """
    🆕 AUTO-DETECT: Find existing Cortex Search services in the database/schema
    """
    if not session:
        return []
    
    try:
        # Query for Cortex Search services
        search_query = f"SHOW CORTEX SEARCH SERVICES IN SCHEMA {database}.{schema}"
        result = session.sql(search_query).collect()
        
        services = []
        for row in result:
            service_name = row.as_dict().get("name", "")
            if service_name:
                full_service_name = f"{database}.{schema}.{service_name}"
                services.append(full_service_name)
        
        return services
        
    except Exception as e:
        # No search services or permission issues
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_search_service_columns(search_service: str) -> List[str]:
    """
    Get the actual indexed columns from a Cortex Search service using DESC
    """
    if not session or not search_service:
        return []
    
    try:
        # Use DESC CORTEX SEARCH SERVICE to get column info
        desc_result = session.sql(f"DESC CORTEX SEARCH SERVICE {search_service}").collect()
        
        # Extract columns from DESC output
        if desc_result:
            row_dict = desc_result[0].as_dict()  # First row contains the service info
            
            # The 'columns' field contains comma-separated column names
            if 'columns' in row_dict and row_dict['columns']:
                columns_str = row_dict['columns']
                return [col.strip() for col in columns_str.split(',')]
            
            # Fallback to attribute_columns 
            elif 'attribute_columns' in row_dict and row_dict['attribute_columns']:
                columns_str = row_dict['attribute_columns']
                # Also include the search column
                search_col = row_dict.get('search_column', '')
                all_columns = []
                if search_col:
                    all_columns.append(search_col)
                all_columns.extend([col.strip() for col in columns_str.split(',')])
                return all_columns
        
        return []
        
    except Exception as e:
        st.error(f"Error analyzing search service: {e}")
        return []

def generate_agent_sql(selected_db: str, selected_schema: str, selected_table: str, 
                      yaml_filename: str, selected_search_service: str, yaml_content: str) -> str:
    """Generate SQL to create a Snowflake Intelligence Agent"""
    try:
        # Extract agent name from table name
        agent_name = selected_table.upper().replace("_", "")
        if len(agent_name) > 20:
            agent_name = agent_name[:20]
            
        # Extract semantic model filename
        semantic_model_filename = yaml_filename.replace('.yaml', '.yaml') if yaml_filename else f"{selected_table.lower()}.yaml"
        
        # Extract search service details
        search_parts = selected_search_service.split('.')
        search_db = search_parts[0] if len(search_parts) > 2 else selected_db
        search_schema = search_parts[1] if len(search_parts) > 2 else selected_schema
        search_service_name = search_parts[-1]
        
        agent_sql = f"""-- ==================================================================
-- SNOWFLAKE INTELLIGENCE AGENT SETUP
-- Auto-generated from SnowViz YAML Wizard
-- ==================================================================

-- Step 1: Create Intelligence Database and Schema (if needed) - NON-DESTRUCTIVE
-- Check if database exists first to avoid dropping existing agents
-- SHOW DATABASES LIKE 'SNOWFLAKE_INTELLIGENCE';
-- Only create if doesn't exist:
CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE;
GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE PUBLIC;

CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE PUBLIC;

GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ACCOUNTADMIN;

-- Step 2: Create Semantic Model Stage (if needed)
CREATE SCHEMA IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()};
CREATE STAGE IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS DIRECTORY = ( ENABLE = TRUE );

-- Step 3: Upload YAML file to stage
-- Save your YAML file locally first, then run:
-- PUT file:///path/to/{semantic_model_filename} @{selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS AUTO_COMPRESS=FALSE;
-- 
-- OR copy-paste the YAML content into a file in Snowflake:
-- CREATE FILE @{selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS/{semantic_model_filename}
-- FROM (SELECT $$[PASTE_YAML_CONTENT_HERE]$$);

-- Step 4: Create the Snowflake Intelligence Agent
CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS."{agent_name}"
COMMENT="Agent for {selected_table} - Auto-generated by SnowViz Wizard"
FROM SPECIFICATION $$
models:
  orchestration: "auto"

instructions:
  response: "You are a data science agent focused on analytical queries for {selected_table} data. Use the search tool to find specific records and the analyst tool to generate SQL queries and insights. Always provide actionable business intelligence and explain your analysis clearly."

tools:
  - tool_spec:
      type: "cortex_search"
      name: "search_data"
  - tool_spec:
      type: "cortex_analyst_text_to_sql"
      name: "analyst_queries"

tool_resources:
  search_data:
    id_column: "SEARCHABLE_PROFILE"
    name: "{search_db.upper()}.{search_schema.upper()}.{search_service_name.upper()}"
  analyst_queries:
    semantic_model_file: "@{selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS/{semantic_model_filename}"
$$;

-- Step 5: Test the Agent
-- Go to https://ai.snowflake.com/ and log in to interact with your agent
-- Your agent name: SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name}

-- ==================================================================
-- AGENT USAGE EXAMPLES
-- ==================================================================

/*
Example queries to test your agent:

1. Search-focused queries:
   "Find high-income consumers with excellent credit in major metros"
   "Show mobile-first millennials with heavy social media usage"
   "Search for brand switchers with high purchase intent and good credit"
   "Find consumers who engage with video content and have premium financial capacity"

2. Analytical queries:
   "What are the top 5 consumer segments by targeting priority score?"
   "Compare KARGO engagement rates vs TransUnion financial capacity"
   "Show DISC personality distribution across enhanced segments"
   "Analyze American Eagle targeting effectiveness before vs after TransUnion data"
   "What's the correlation between device type and financial tier?"

3. Business intelligence:
   "Which TransUnion-enhanced segments show highest conversion potential?"
   "How does credit score data improve Kargo's basic segmentation?"
   "What marketing opportunities exist for DISC Eagle personalities?"
   "Which consumers moved from 'removed' to 'added' in American Eagle targeting?"
   "Show ROI potential of TransUnion financial intelligence for advertisers"

4. TransUnion value proposition:
   "Compare targeting precision: Kargo-only vs TransUnion-enhanced"
   "Which financial capacity tiers have highest engagement rates?"
   "How many consumers gained verified purchase intent through TransUnion data?"
*/"""

        return agent_sql
        
    except Exception as e:
        st.error(f"Error generating agent SQL: {e}")
        return None

def generate_agent_sql_with_name(agent_name: str, selected_db: str, selected_schema: str, selected_table: str, 
                                yaml_filename: str, selected_search_service: str) -> str:
    """Generate SQL to create a Snowflake Intelligence Agent using provided agent name"""
    try:
        # Extract semantic model filename
        semantic_model_filename = yaml_filename.replace('.yaml', '.yaml') if yaml_filename else f"{selected_table.lower()}.yaml"
        
        # Extract search service details
        search_parts = selected_search_service.split('.')
        search_db = search_parts[0] if len(search_parts) > 2 else selected_db
        search_schema = search_parts[1] if len(search_parts) > 2 else selected_schema
        search_service_name = search_parts[-1]
        
        agent_sql = f"""-- ==================================================================
-- SNOWFLAKE INTELLIGENCE AGENT SETUP
-- Auto-generated from SnowViz YAML Wizard
-- ==================================================================

-- Step 1: Create Intelligence Database and Schema (if needed) - NON-DESTRUCTIVE
CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE;
GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE PUBLIC;

CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE PUBLIC;

GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ACCOUNTADMIN;

-- Step 2: Create Semantic Model Stage (if needed)
CREATE SCHEMA IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()};
CREATE STAGE IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS DIRECTORY = ( ENABLE = TRUE );

-- Step 3: Create the Snowflake Intelligence Agent
CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS."{agent_name}"
COMMENT="Agent for {selected_table} - Auto-generated by SnowViz Wizard - Name: {agent_name}"
FROM SPECIFICATION $$
models:
  orchestration: "auto"

instructions:
  response: "You are a data science agent focused on analytical queries for {selected_table} data. Use the search tool to find specific records and the analyst tool to generate SQL queries and insights. Always provide actionable business intelligence and explain your analysis clearly."

tools:
  - tool_spec:
      type: "cortex_search"
      name: "search_data"
  - tool_spec:
      type: "cortex_analyst_text_to_sql"
      name: "analyst_queries"

tool_resources:
  search_data:
    id_column: "SEARCHABLE_PROFILE"
    name: "{search_db.upper()}.{search_schema.upper()}.{search_service_name.upper()}"
  analyst_queries:
    semantic_model_file: "@{selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS/{semantic_model_filename}"
$$;

-- Step 4: Test the Agent
-- Go to https://ai.snowflake.com/ and log in to interact with your agent
-- Your agent name: SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name}
"""

        return agent_sql
        
    except Exception as e:
        st.error(f"Error generating agent SQL: {e}")
        return None

def list_semantic_models(session, selected_db: str, selected_schema: str) -> List[str]:
    """List available YAML files in the SEMANTIC_MODELS stage"""
    try:
        stage_path = f"{selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS"
        
        # List files in the semantic models stage
        list_sql = f"LIST @{stage_path}"
        result = session.sql(list_sql).collect()
        
        yaml_files = []
        for row in result:
            row_dict = row.as_dict()
            filename = row_dict.get('name', '')
            if filename:
                lower = filename.lower()
                if lower.endswith('.yaml') or lower.endswith('.yml') or lower.endswith('.yaml.gz') or lower.endswith('.yml.gz'):
                    # Extract just the filename from the full path
                    yaml_files.append(filename.split('/')[-1])
        
        return sorted(yaml_files)
        
    except Exception as e:
        st.warning(f"Could not list semantic models in @{stage_path}: {e}")
        return []

def create_snowflake_agent(session, agent_name: str, selected_db: str, selected_schema: str, selected_table: str,
                          stage_name: str, yaml_file_name: str, selected_search_service: str) -> bool:
    """Execute SQL to create Snowflake Intelligence Agent directly"""
    try:
        if not session:
            st.error("No Snowflake session available")
            return False
            
        # Extract search service details
        search_parts = selected_search_service.split('.')
        search_db = search_parts[0] if len(search_parts) > 2 else selected_db
        search_schema = search_parts[1] if len(search_parts) > 2 else selected_schema
        search_service_name = search_parts[-1]
        
        # Generate the agent SQL using current form values
        agent_sql = f"""-- ==================================================================
-- SNOWFLAKE INTELLIGENCE AGENT: {agent_name}
-- Auto-generated from SnowViz YAML Wizard
-- ==================================================================

-- Step 1: Setup (non-destructive)
CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE;
GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE PUBLIC;
CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE PUBLIC;
GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ACCOUNTADMIN;

-- Step 2: Create the Agent
CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS."{agent_name}"
COMMENT="Agent for {selected_table} - User configured name: {agent_name}"
FROM SPECIFICATION $$
models:
  orchestration: "auto"

instructions:
  response: "You are a data science agent focused on analytical queries for {selected_table} data. Use the search tool to find specific records and the analyst tool to generate SQL queries and insights. Always provide actionable business intelligence and explain your analysis clearly."

tools:
  - tool_spec:
      type: "cortex_search"
      name: "search_data"
  - tool_spec:
      type: "cortex_analyst_text_to_sql"
      name: "analyst_queries"

tool_resources:
  search_data:
    id_column: "SEARCHABLE_PROFILE"
    name: "{search_db.upper()}.{search_schema.upper()}.{search_service_name.upper()}"
  analyst_queries:
    semantic_model_file: "{stage_name}/{yaml_file_name}"
$$;

-- Step 3: Test at https://ai.snowflake.com/
-- Agent name: SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name}
"""
        
        # Store the SQL for the expandable display
        st.session_state["agent_sql"] = agent_sql
        
        # Non-destructive setup - check if database exists first
        try:
            # Check if SNOWFLAKE_INTELLIGENCE database exists
            db_check = session.sql("SHOW DATABASES LIKE 'SNOWFLAKE_INTELLIGENCE'").collect()
            if not db_check:
                st.info("Creating SNOWFLAKE_INTELLIGENCE database...")
                session.sql("CREATE DATABASE SNOWFLAKE_INTELLIGENCE").collect()
                session.sql("GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE PUBLIC").collect()
            else:
                st.info("✅ SNOWFLAKE_INTELLIGENCE database exists")
                
            # Ensure schema and permissions exist (these are safe to re-run)
            session.sql("CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS").collect()
            session.sql("GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE PUBLIC").collect()
            
            # Grant permissions (safe to re-run)
            try:
                session.sql("GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ACCOUNTADMIN").collect()
            except Exception as grant_error:
                st.warning(f"Grant may have failed (often expected): {grant_error}")
            
            # Create semantic models stage (safe)
            session.sql(f"CREATE SCHEMA IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()}").collect()
            session.sql(f"CREATE STAGE IF NOT EXISTS {selected_db.upper()}.{selected_schema.upper()}.SEMANTIC_MODELS DIRECTORY = ( ENABLE = TRUE )").collect()
            
        except Exception as setup_error:
            st.warning(f"Setup step failed: {setup_error}")
                
        # Create the agent using current form values
        agent_create_sql = f'''CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS."{agent_name}"
COMMENT="Agent for {selected_table} - User configured: {agent_name}"
FROM SPECIFICATION $$
models:
  orchestration: "auto"

instructions:
  response: "You are a data science agent focused on analytical queries for {selected_table} data. Use the search tool to find specific records and the analyst tool to generate SQL queries and insights. Always provide actionable business intelligence and explain your analysis clearly."

tools:
  - tool_spec:
      type: "cortex_search"
      name: "search_data"
  - tool_spec:
      type: "cortex_analyst_text_to_sql"
      name: "analyst_queries"

tool_resources:
  search_data:
    id_column: "SEARCHABLE_PROFILE"
    name: "{search_db.upper()}.{search_schema.upper()}.{search_service_name.upper()}"
  analyst_queries:
    semantic_model_file: "{stage_name}/{yaml_file_name}"
$$'''
        
        # Execute agent creation
        session.sql(agent_create_sql).collect()
        
        return True
        
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return False

def render_yaml_generation_tab(session, db: str, sc: str, selected_name: str, dimensions: List[str], 
                              metric_defs: List[Dict], time_col: str, cols_df_cache) -> None:
    """Render the YAML generation tab with all controls and functionality"""
    st.subheader("🚀 Generate Final YAML")
    
    # Simple, predictable default - user can customize as needed
    default_app_name = f"Snow Visualizer for {selected_name.replace('_', ' ').title()}"
    
    app_name = st.text_input("App Name", value=default_app_name, key="custom_app_name")
    app_desc = st.text_input("Description", value=f"Interactive dashboard for {db}.{sc}.{selected_name}", key="custom_app_desc")
    default_window = st.selectbox("Default Time Window", options=["last_12_months","last_4_weeks","ytd","fytd"], index=0, key="custom_window")
    default_grain = st.selectbox("Default Grain", options=["month","day","week","quarter","year"], index=0, key="custom_grain")
    
    # Summary before generation
    col1, col2 = st.columns(2)
    with col1:
        dim_count = len([d for d in dimensions if st.session_state.get("dim_customs", {}).get(d, {}).get("included", True)])
        st.metric("Final Dimensions", dim_count)
    with col2:
        selected_metric_keys = [m["key"] for m in metric_defs]
        met_count = len([m for m in selected_metric_keys if st.session_state.get("met_customs", {}).get(m, {}).get("included", True)])
        st.metric("Final Metrics", met_count)

    # Filename configuration
    st.markdown("**📁 File Configuration**")
    default_filename = f"{selected_name.lower()}_snowviz.yaml"
    filename = st.text_input("YAML Filename", value=default_filename, help="Filename for the generated YAML configuration")
        
    # Search service integration moved to Intelligence tab
    # Get search service info for YAML generation
    search_services = detect_cortex_search_services(db, sc)
    selected_search_service = ""
    search_columns = []
    
    if search_services and len(search_services) > 0:
        # Use the first available search service for YAML generation
        selected_search_service = search_services[0]
        search_columns = get_search_service_columns(selected_search_service)
        st.info(f"🔗 Using search service: {selected_search_service} (configure in Intelligence tab)")

    # 🆕 GENERATE BUTTON: Only generate when user explicitly requests
    # Show current dimension and metric customizations snapshot
    with st.expander("🧪 Customizations snapshot", expanded=False):
        st.write("**Keys Only:**", {
            "dim_customs_keys": list(st.session_state.get("dim_customs", {}).keys()),
            "met_customs_keys": list(st.session_state.get("met_customs", {}).keys()),
            "selected_dimensions": dimensions,
            "selected_metric_keys": [m["key"] for m in metric_defs],
        })
        
        st.write("**Full Dimension Customizations:**")
        st.json(st.session_state.get("dim_customs", {}))
        
        st.write("**Full Metric Customizations:**")
        st.json(st.session_state.get("met_customs", {}))
    if st.button("🎯 Generate Customized YAML", help="Generate YAML configuration with your customizations", key="generate_final_yaml"):
        # Clear any existing YAML from session state
        st.session_state.pop("svw_yaml_text", None)
        
        # Get customized configurations
        customized_dimensions = []
        for dim in dimensions:
            dim_config = st.session_state.get("dim_customs", {}).get(dim, {})
            if dim_config.get("included", True):  # Default include if not specified
                customized_dimensions.append({
                    "key": dim,
                    "label": dim_config.get("label", dim.replace("_", " ").title()),
                    "column": dim,
                    "type": "categorical",  # Default type for now
                    "description": dim_config.get("description", f"Analysis dimension for {dim.lower().replace('_', ' ')}"),
                    "order": dim_config.get("order", 0)  # Include order for sorting
                })
        
        # 🚨 CRITICAL FIX: Sort dimensions by custom order before generating YAML
        customized_dimensions = sorted(customized_dimensions, key=lambda x: x.get("order", 0))
                
        customized_metrics = []
        for metric in metric_defs:
            key = metric["key"]
            met_config = st.session_state.get("met_customs", {}).get(key, {})
            if met_config.get("included", True):  # Default include if not specified
                customized_metrics.append({
                    "key": key,
                    "label": met_config.get("label", metric["label"]),
                    "sql": met_config.get("sql", metric["sql"]),
                    "format": met_config.get("format", metric["format"]),
                    "round": met_config.get("round", metric["round"]),
                    "description": met_config.get("description", f"Calculated metric for {key.lower().replace('_', ' ')}"),
                    "order": met_config.get("order", 0)  # Include order for sorting
                })
        
        # 🚨 CRITICAL FIX: Sort metrics by custom order before generating YAML
        customized_metrics = sorted(customized_metrics, key=lambda x: x.get("order", 0))
        
        # Generate final YAML
        yaml_text = generate_yaml_with_customizations(
            db, sc, selected_name, customized_dimensions, customized_metrics,
            time_col, default_window, default_grain, app_name, app_desc, True, 
            selected_search_service, search_columns
        )
        
        if yaml_text:
            st.session_state["svw_yaml_text"] = yaml_text
            st.session_state["svw_saved_filename"] = filename
            st.success("✅ Customized YAML Generated!")
            
            # Show YAML content in expandable section to avoid scrolling
            with st.expander("📋 View Generated YAML Configuration", expanded=False):
                st.code(yaml_text, language="yaml")
            
            # Show tab summary for debugging
            dimension_keys = [d["key"] for d in customized_dimensions]
            tabs = generate_enhanced_tabs_with_search(dimension_keys, customized_metrics, time_col, default_grain, [], selected_search_service, search_columns)
            if tabs and len(tabs) > 0:
                st.success(f"📊 Generated {len(tabs)} tabs: {', '.join([t['title'] for t in tabs])}")
            else:
                st.error(f"❌ No tabs generated! Dimensions: {dimension_keys}, Metrics: {len(customized_metrics)}")
        else:
            st.error("❌ Failed to generate YAML")
    else:
        if st.session_state.get("svw_yaml_text"):
            st.info("✅ YAML ready for download and agent creation")
            
            # Show tab summary for debugging
            # Get customized dimensions for tab generation
            customized_dimensions = []
            for dim in dimensions:
                dim_config = st.session_state.get("dim_customs", {}).get(dim, {})
                if dim_config.get("included", True):
                    customized_dimensions.append({"key": dim})
            
            customized_metrics = []
            for metric in metric_defs:
                key = metric["key"]
                met_config = st.session_state.get("met_customs", {}).get(key, {})
                if met_config.get("included", True):
                    customized_metrics.append(metric)
            
            dimension_keys = [d["key"] for d in customized_dimensions]  
            tabs = generate_enhanced_tabs_with_search(dimension_keys, customized_metrics, time_col, default_grain, [], selected_search_service, search_columns)
            if tabs and len(tabs) > 0:
                # Show tab summary for debugging
                st.info(f"📊 Generated {len(tabs)} tabs: {', '.join([t['title'] for t in tabs])}")
            else:
                st.info("👆 Configure your dimensions and metrics above, then click 'Generate Customized YAML'")

    # Download section
    col1, col2 = st.columns(2)
    with col1:
        # Use customized YAML if available
        yaml_to_download = st.session_state.get("svw_yaml_text", "")
        if yaml_to_download:
            st.subheader("📥 Download YAML")
            st.download_button(
                label="📥 Download YAML Configuration",
                data=yaml_to_download.encode("utf-8"),
                file_name=filename,
                mime="text/yaml",
                help="Download the generated YAML file to upload to VISUALIZATION_YAML_STAGE"
            )
        else:
            st.info("👆 Generate your customized YAML first")
            
    with col2:
        # Config save functionality - use loaded config name if available
        default_cfg_name = st.session_state.get("svw_loaded_config_name", f"snowviz_{selected_name.lower()}")
        cfg_name = st.text_input("Config Name (save)", value=default_cfg_name)
        
        # 🚨 IMPORTANT WORKFLOW WARNING
        dim_customs_count = len(st.session_state.get("dim_customs", {}))
        met_customs_count = len(st.session_state.get("met_customs", {}))
        
        if dim_customs_count == 0 and met_customs_count == 0:
            st.warning("⚠️ No customizations detected! Make sure to:")
            st.warning("   1. Go to Dimensions tab and click 'Apply All Dimension Changes'")
            st.warning("   2. Go to Metrics tab and click 'Apply All Metric Changes'")
            st.warning("   3. Then come back here to save")
        else:
            st.success(f"✅ Ready to save: {dim_customs_count} dimension + {met_customs_count} metric customizations")
        
        if st.button("💾 Save to CORTEX_AI_FRAMEWORK_DB.CONFIGS"):
            selected_metric_keys = [m["key"] for m in metric_defs]
            
            # Get current customizations
            dim_customs = st.session_state.get("dim_customs", {})
            met_customs = st.session_state.get("met_customs", {})
            
            # Save comprehensive metadata including all new configuration points
            meta = {
                "app_name": app_name,
                "description": app_desc,
                "time_col": time_col,
                "default_grain": default_grain,
                "default_window": default_window,
                "filename": filename,
                "dimensions": dimensions,
                "selected_metrics": selected_metric_keys,
                # New configuration points
                "search_service": selected_search_service if 'selected_search_service' in locals() else "",
                "search_columns": search_columns if 'search_columns' in locals() else [],
                "dimension_customizations": dim_customs,
                "metric_customizations": met_customs,
                "tab_count": len(tabs) if 'tabs' in locals() and tabs else 0,
                "config_version": "2.0",  # Track new format
                "has_intelligence_features": bool('selected_search_service' in locals() and selected_search_service),
                "saved_timestamp": str(pd.Timestamp.now())
            }
            ok = save_config_to_database(
                session,
                cfg_name,
                db,
                sc,
                selected_name,
                cols_df_cache if cols_df_cache is not None else pd.DataFrame(),
                st.session_state.get("svw_yaml_text", "") or "",
                meta,
            )
            if ok:
                st.success(f"✅ Saved config: {cfg_name}")
                st.session_state["svw_saved_filename"] = filename
                # 🆕 Update loaded config name so subsequent saves use the same name
                st.session_state["svw_loaded_config_name"] = cfg_name
                
                # Show what was saved for verification
                dim_customs_count = len(st.session_state.get("dim_customs", {}))
                met_customs_count = len(st.session_state.get("met_customs", {}))
                st.info(f"💾 Saved {dim_customs_count} dimension customizations and {met_customs_count} metric customizations")
                st.info("💡 **Config Saved**: You can now use the Intelligence tab to create an agent, or download your YAML file above.")
                
    st.info("💡 Configure search service and create agent in the Intelligence tab")
    
    # Upload instructions (full width, outside columns)
    yaml_to_download = st.session_state.get("svw_yaml_text", "")
    if yaml_to_download:
        st.subheader("📤 Upload Instructions")
        
        with st.expander("📋 Complete Upload Instructions", expanded=True):
            upload_instructions = f"""
**Snowsight UI Upload:**

1. Navigate to **CORTEX_AI_FRAMEWORK_DB.CONFIGS.VISUALIZATION_YAML_STAGE**
2. Click "Upload" → Select your YAML file  
3. In path field, enter a project name (e.g., `/customer_orders/`, `/dashboards/`, `/analytics/`)
4. Click "Upload" - directory created automatically!

💡 **Done!** Your dashboard is ready for Snow Viz (App 6)
"""
            st.markdown(upload_instructions)

    st.markdown("---")

def render_intelligence_tab(session, db: str, sc: str, selected_name: str, dimensions: List[str], 
                           metric_defs: List[Dict], filename: str) -> None:
    """Render the Intelligence/Agent creation tab"""
    st.subheader("🤖 Snowflake Intelligence & Search")
    
    # 🆕 AUTO-DETECT: Check for existing Cortex Search services
    st.markdown("### 🔍 Cortex Search Integration")
    search_services = detect_cortex_search_services(db, sc)
    
    selected_search_service = ""
    search_columns = []
    
    if search_services:
        st.success(f"✅ Found {len(search_services)} Cortex Search services")
        search_options = ["None (disable search tab)"] + search_services
        selected_search = st.selectbox("Use existing Cortex Search service?", search_options)
        
        if selected_search != "None (disable search tab)":
            selected_search_service = selected_search
            
            # 🆕 ENHANCED: Get indexed columns from search service
            search_columns = get_search_service_columns(selected_search_service)
            
            if search_columns:
                st.success(f"🔗 Will use search service: {selected_search_service}")
                st.info(f"📋 Indexed columns: {', '.join(search_columns[:10])}")
                
                # Show which of your dimensions/metrics are available
                search_columns = search_columns or []  # Safety check for None
                selected_metric_keys = [m["key"] for m in metric_defs]
                available_dims = [d for d in dimensions if d.lower() in [col.lower() for col in search_columns]]
                available_mets = [m for m in selected_metric_keys if m.lower() in [col.lower() for col in search_columns]]
                
                if available_dims:
                    st.info(f"🎯 Available dimensions: {', '.join(available_dims)}")
                if available_mets:
                    st.info(f"📊 Available metrics: {', '.join(available_mets)}")
                if not available_dims and not available_mets:
                    st.warning("⚠️ None of your selected dimensions/metrics are indexed in this search service")
                    st.info("💡 You may need to recreate the search service with your current table schema")
    else:
        st.info("🔍 No Cortex Search services found in this database/schema")
        st.info("💡 Create a Cortex Search service first to enable semantic search in your dashboard")
        
    
    # Agent Creation Section
    st.markdown("### 🤖 Snowflake Intelligence Agent")
    
    yaml_to_download = st.session_state.get("svw_yaml_text", "")
    if yaml_to_download and selected_search_service:
        # Agent configuration inputs
        agent_name = st.text_input(
            "Agent Name", 
            value=selected_name.upper().replace("_", "")[:20],
            help="Name for your Snowflake Intelligence Agent"
        )
        
        stage_name = st.text_input(
            "Semantic Models Stage", 
            value=f"@{db.upper()}.{sc.upper()}.SEMANTIC_MODELS",
            help="Stage where your YAML semantic model is stored"
        )
        
        # Auto-detect available semantic models
        available_yamls = list_semantic_models(session, db, sc)
        
        if available_yamls:
            st.success(f"✅ Found {len(available_yamls)} semantic model(s)")
            
            # Let user select from available YAML files
            default_yaml_index = 0
            if filename in available_yamls:
                default_yaml_index = available_yamls.index(filename)
            elif any(filename.lower() in yaml.lower() for yaml in available_yamls):
                # Find partial match
                for i, yaml in enumerate(available_yamls):
                    if filename.lower() in yaml.lower():
                        default_yaml_index = i
                        break
            
            yaml_file_name = st.selectbox(
                "Select Semantic Model",
                options=available_yamls,
                index=default_yaml_index,
                help="Choose the YAML semantic model to use for the agent"
            )
            
            # Show preview of selected semantic model
            if yaml_file_name:
                with st.expander(f"📋 Preview: {yaml_file_name}", expanded=False):
                    try:
                        # Get the YAML content from the stage (read all lines)
                        get_file_sql = f"SELECT $1 FROM @{db.upper()}.{sc.upper()}.SEMANTIC_MODELS/{yaml_file_name} (FILE_FORMAT => 'CORTEX_AI_FRAMEWORK_DB.CONFIGS.YAML_CSV_FORMAT')"
                        yaml_content = session.sql(get_file_sql).collect()
                        if yaml_content:
                            # Concatenate all rows to get complete YAML content
                            content_lines = [row[0] for row in yaml_content if row[0] is not None]
                            content_str = '\n'.join(content_lines)
                            if content_str.strip():
                                st.code(content_str, language="yaml")
                                st.info(f"📄 {len(content_lines)} lines in semantic model")
                            else:
                                st.warning("YAML file appears to be empty")
                        else:
                            st.warning("Could not read YAML content")
                    except Exception as e:
                        st.warning(f"Could not preview YAML: {e}")
                        st.info("💡 Try uploading the YAML file to the stage first")
        else:
            st.warning("⚠️ No YAML files found in semantic models stage")
            st.info("💡 Upload your YAML file first, or check the stage path above")
            st.caption("Tip: supported extensions: .yaml, .yml, optionally .gz compressed")
            yaml_file_name = st.text_input(
                "YAML File Name (manual)", 
                value=filename,
                help="Enter YAML filename manually if auto-detection failed"
            )
        
        if st.button("🚀 Create & Execute Agent", help="Create Snowflake Agent with Cortex Search and Analyst"):
            with st.spinner("Creating Snowflake Intelligence Agent..."):
                success = create_snowflake_agent(
                    session, agent_name, db, sc, selected_name,
                    stage_name, yaml_file_name, selected_search_service
                )
                if success:
                    st.success(f"✅ Agent '{agent_name}' created successfully!")
                    st.info(f"🎯 Test your agent: Go to https://ai.snowflake.com/ and log in to interact with agent '{agent_name}'")
                else:
                    st.error("❌ Failed to create agent - check expandable SQL section below")
    else:
        if not yaml_to_download:
            st.info("💡 Generate your YAML configuration first (in Generate tab)")
        if not selected_search_service:
            st.info("💡 Select a Cortex Search service above")
        if not yaml_to_download and not selected_search_service:
            st.info("🤖 Complete YAML generation + search service setup to create agent")
            
    # Show generated agent SQL in expandable section
    if "agent_sql" in st.session_state and st.session_state["agent_sql"]:
        with st.expander("🔍 View Generated Agent SQL", expanded=False):
            st.code(st.session_state["agent_sql"], language="sql")
            
            # Download button for agent SQL in expander
            agent_filename = f"{filename.replace('.yaml', '')}_agent.sql"
            st.download_button(
                "📥 Download Agent SQL", 
                st.session_state["agent_sql"].encode("utf-8"), 
                file_name=agent_filename, 
                mime="text/sql",
                key="download_agent_sql_expander"
            )

def render_metrics_tab(metric_defs: List[Dict]) -> None:
    """Render the metrics customization tab with all controls"""
    st.subheader("📈 Customize Metrics")
    st.info("💡 Make your changes below, then click 'Apply All Metric Changes' to save them.")
    
    # Initialize metric customizations (handle both new and existing session state)
    if "met_customs" not in st.session_state:
        st.session_state.met_customs = {}
    
    # 🆕 FIXED: Ensure all metrics have all required fields (handles upgrades)
    for i, m in enumerate(metric_defs):
        key = m["key"]
        if key not in st.session_state.met_customs:
            st.session_state.met_customs[key] = {}
        
        defaults = {
            "included": True,
            "label": m["label"],
            "description": f"Calculated metric for {key.lower().replace('_', ' ')}",
            "order": i,  # Default ordering
            "sql": m["sql"],
            "format": m["format"],
            "round": m["round"]
        }
        
        for field, default_value in defaults.items():
            if field not in st.session_state.met_customs[key]:
                st.session_state.met_customs[key][field] = default_value
            # 🚨 CRITICAL FIX: Always update order to match current position
            elif field == "order":
                st.session_state.met_customs[key][field] = default_value
    
    
    # Use form to batch all metric changes and prevent page refresh
    with st.form("metrics_form"):
        # Edit each metric
        metric_changes = {}  # Store changes temporarily
        for i, metric in enumerate(metric_defs):
            key = metric["key"]
            # Force expansion by using container instead of expander for better UX
            st.markdown(f"### 📊 {key}")
            with st.container():
                st.markdown("---")  # Visual separator
                
                # Row 1: Basic settings
                col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
                
                with col1:
                    default_incl = st.session_state.met_customs.get(key, {}).get("included", True)
                    incl = st.checkbox("Include", value=default_incl, key=f"minc_{key}")
                
                with col2:
                    default_lbl = st.session_state.met_customs.get(key, {}).get("label", metric["label"])
                    lbl = st.text_input("Label", value=default_lbl, key=f"mlbl_{key}")
                
                with col3:
                    default_desc = st.session_state.met_customs.get(key, {}).get("description", f"Calculated metric for {key.lower().replace('_', ' ')}")
                    desc = st.text_area("Description", value=default_desc, key=f"mdsc_{key}", height=50)
                
                with col4:
                    default_ord = st.session_state.met_customs.get(key, {}).get("order", i)
                    ord = st.number_input("Priority", min_value=0, value=default_ord, key=f"mord_{key}")
                
                # Store changes for batch update
                metric_changes[key] = {
                    "included": incl,
                    "label": lbl, 
                    "description": desc,
                    "order": ord
                }
                
                # Row 2: SQL Calculation & Formatting
                st.markdown("**📝 SQL Calculation**")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Editable SQL calculation
                    default_sql = st.session_state.met_customs.get(key, {}).get("sql", metric["sql"])
                    sql_calc = st.text_area(
                        "SQL Calculation",
                        value=default_sql, 
                        key=f"msql_{key}",
                        height=80,
                        help="Edit the SQL calculation for this metric (e.g., AVG(COLUMN), SUM(COLUMN), COUNT(*))"
                    )
                    
                    # Show SQL preview
                    st.code(sql_calc, language="sql")
                
                with col2:
                    format_options = ["number", "integer", "percent", "currency"]
                    current_format = st.session_state.met_customs.get(key, {}).get("format", metric["format"])
                    format_index = format_options.index(current_format) if current_format in format_options else 0
                    
                    fmt = st.selectbox("Format", format_options, 
                                     index=format_index,
                                     key=f"mfmt_{key}")
                    
                    if fmt in ["number", "percent", "currency"]:
                        default_round = st.session_state.met_customs.get(key, {}).get("round", metric["round"])
                        rnd = st.number_input("Decimals", min_value=0, max_value=6, 
                                            value=default_round,
                                            key=f"mrnd_{key}")
                    else:
                        rnd = 0  # No decimals for integer format
                
                # Add to batch changes
                metric_changes[key].update({
                    "sql": sql_calc,
                    "format": fmt,
                    "round": rnd if fmt in ["number", "percent", "currency"] else 0
                })
        
        # Single form submit button to apply all changes
        metrics_form_submitted = st.form_submit_button("💾 Apply All Metric Changes")
        
        # Only update session state when form is submitted
        if metrics_form_submitted:
            for key, changes in metric_changes.items():
                st.session_state.met_customs[key].update(changes)
            st.success(f"✅ All metric changes applied! Saved {len(metric_changes)} customizations to session state.")
            st.info("💡 **Next Step**: Click on the 'Generate' tab to create your YAML with these customizations.")
    
    # SQL validation helper removed to prevent page refresh issues
    
    # Apply customizations to metrics list
    customized_metrics = []
    for metric in metric_defs:
        key = metric["key"]
        met_config = st.session_state.met_customs.get(key, {})
        if met_config.get("included", True):  # Default include if not specified
            customized_metrics.append({
                "key": key,
                "label": met_config.get("label", metric["label"]),
                "sql": met_config.get("sql", metric["sql"]),  # Use edited SQL, or fall back to original
                "format": met_config.get("format", metric["format"]),
                "round": met_config.get("round", metric["round"]),
                "description": met_config.get("description", f"Calculated metric for {key.lower().replace('_', ' ')}")
            })
    
    # Sort by priority order
    customized_metrics = sorted(customized_metrics, key=lambda x: st.session_state.met_customs.get(x["key"], {}).get("order", 0))
    st.info(f"✅ {len(customized_metrics)} metrics configured")
    
    # Show top metrics preview
    if customized_metrics:
        st.markdown("**📈 Top Priority Metrics:**")
        for i, m in enumerate(customized_metrics[:6]):
            priority = st.session_state.met_customs.get(m["key"], {}).get("order", 0)
            st.write(f"{i+1}. **{m['label']}** (priority: {priority}, format: {m['format']}) - {m['description'][:50]}...")
        if customized_metrics:
            st.code(f"SQL: {customized_metrics[0]['sql']}", language="sql")  # Show first metric's customized SQL
    
    # Section to verify saving
    with st.expander("🔧 Current Metric Settings", expanded=False):
        st.json(st.session_state.get("met_customs", {}))

def generate_yaml_with_customizations(selected_db: str, selected_schema: str, selected_table: str, 
                                     customized_dimensions: List[Dict], customized_metrics: List[Dict], 
                                     time_col: str, default_window: str, default_grain: str, 
                                     app_name: str, app_desc: str, exclude_object_type: bool, 
                                     search_service: str = "", search_columns: List[str] = []) -> str:
    """
    Generate YAML using customized dimensions and metrics instead of raw database columns
    """
    
    # Extract just the dimension keys for tab configuration
    dimension_keys = [d["key"] for d in customized_dimensions]
    
    # Build top priority metrics for default cards (first 6 by order)
    default_cards = [m["key"] for m in customized_metrics[:6]]
    
    yaml_obj = {
        "version": 0.1,
        "app": {
            "name": app_name,
            "description": app_desc,
            "data_source": {
                "database": selected_db,
                "schema": selected_schema,
                "table": selected_table
            },
            "time": {
                "column": time_col,
                "default_grain": default_grain,
                "supported_grains": ["day", "week", "month", "quarter", "year"],
                "default_window": default_window,
                "windows": ["last_4_weeks", "last_12_months", "ytd", "fytd"]
            }
        },
        "sql_macros": {
            "base_from": "FROM {DB}.{SCHEMA}.{TABLE}",
            "time_bucket": "DATE_TRUNC({GRAIN}, {TIME_COL})",
            "filter_time_window": {
                "last_4_weeks": "{TIME_COL} >= DATEADD(week, -4, CURRENT_DATE())",
                "last_12_months": "{TIME_COL} >= DATEADD(month, -12, CURRENT_DATE())",
                "ytd": "YEAR({TIME_COL}) = YEAR(CURRENT_DATE())",
                "fytd": "YEAR({TIME_COL}) = YEAR(CURRENT_DATE())"
            }
        },
        "dimensions": customized_dimensions,  # 🆕 CONFIRMED: Use customized dimensions with edited labels/descriptions
        "metrics": customized_metrics,        # 🆕 CONFIRMED: Use customized metrics with edited labels/SQL/ordering
        "comparisons": {"mom": {"label": "MoM", "grain": "month"}, "yoy": {"label": "YoY", "grain": "year"}},
        "cards": {"default_metrics": default_cards, "comparisons": ["mom","yoy"]},
        "tabs": generate_enhanced_tabs_with_search([d["key"] for d in customized_dimensions], customized_metrics if customized_metrics else [], time_col or "PRIMARY_DATE", default_grain, default_cards, search_service, search_columns),  # Use actual dimension keys
        "filters": {"time_window": default_window, "where": ""}
    }
    
    try:
        yaml_text = yaml.safe_dump(yaml_obj, sort_keys=False, allow_unicode=True)
        return yaml_text
    except Exception as e:
        return f"Error generating YAML: {str(e)}"

# =========================================================================================================

st.set_page_config(page_title="SnowViz YAML Wizard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input { border: 1px solid #d1d5db; border-radius: 6px; }
.stButton > button { border: 1px solid #3b82f6; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def _run(sql: str) -> pd.DataFrame:
    if session is None:
        st.error("No active Snowflake session.")
        return pd.DataFrame()
    return session.sql(sql).to_pandas()

def _find_col(df: pd.DataFrame, target_lower: str) -> Optional[str]:
    for c in list(df.columns):
        if str(c).lower() == target_lower:
            return c
    return None

def _pick_name_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    # Try explicit candidates first
    for cand in preferred:
        col = _find_col(df, cand)
        if col:
            return col
    # Fallback: first object/string-like column (skip datetimes)
    for c in list(df.columns):
        dt = df[c].dtype
        if str(dt).startswith("datetime"):
            continue
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt):
            return c
    # As a last resort, first column
    return list(df.columns)[0] if len(df.columns) else None


# =============================
# Config persistence (CORTEX_AI_FRAMEWORK_DB)
# =============================

def setup_config_database(_sess) -> bool:
    if not _sess:
        return False
    try:
        _sess.sql("CREATE DATABASE IF NOT EXISTS CORTEX_AI_FRAMEWORK_DB").collect()
        _sess.sql("CREATE SCHEMA IF NOT EXISTS CORTEX_AI_FRAMEWORK_DB.CONFIGS").collect()
        # Create table with TEXT columns
        _sess.sql(
            """
            CREATE TABLE IF NOT EXISTS CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS (
                CONFIG_ID STRING PRIMARY KEY,
                CONFIG_NAME STRING,
                SOURCE_DB STRING,
                SOURCE_SCHEMA STRING,
                SOURCE_OBJECT STRING,
                COLUMN_INFO TEXT,
                YAML_TEXT TEXT,
                METADATA TEXT,
                CREATED_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
            """
        ).collect()
        return True
    except Exception as e:
        st.error(f"Error setting up CORTEX_AI_FRAMEWORK_DB.CONFIGS: {e}")
        return False


def save_config_to_database(_sess, config_name: str, source_db: str, source_schema: str,
                            source_object: str, columns_df: pd.DataFrame,
                            yaml_text: str, meta: Dict[str, Any]) -> bool:
    if not _sess:
        return False
    try:
        import uuid
        cfg_id = str(uuid.uuid4())
        # Remove existing by name
        _sess.sql(
            "DELETE FROM CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS WHERE CONFIG_NAME = ?",
            [config_name]
        ).collect()

        col_info = columns_df.to_json(orient="records") if isinstance(columns_df, pd.DataFrame) else "[]"
        meta_json = json.dumps(meta)
        
        
        _sess.sql(
            """
            INSERT INTO CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS
            (CONFIG_ID, CONFIG_NAME, SOURCE_DB, SOURCE_SCHEMA, SOURCE_OBJECT, COLUMN_INFO, YAML_TEXT, METADATA)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [cfg_id, config_name, source_db, source_schema, source_object, col_info, yaml_text, meta_json]
        ).collect()
        
        st.success(f"✅ Successfully wrote to database")
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


@st.cache_data(ttl=60)  # Cache for 1 minute - configs change frequently
def load_saved_configurations(_sess) -> List[Dict[str, Any]]:
    if not _sess:
        return []
    try:
        # Show what we're trying to query
        query = "SELECT CONFIG_ID, CONFIG_NAME, SOURCE_DB, SOURCE_SCHEMA, SOURCE_OBJECT, LAST_UPDATED FROM CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS ORDER BY LAST_UPDATED DESC"
        
        rows = _sess.sql(query).collect()
        configs = [r.as_dict() for r in rows]  # 🆕 FIXED: Use .as_dict() for Snowpark Row objects
        
        # Show what we found
        if len(configs) == 0:
            st.info("🔍 Table exists but no configurations found")
        else:
            st.success(f"✅ Found {len(configs)} configurations")
            
        return configs
        
    except Exception as e:
        # Show the actual error instead of hiding it
        st.error(f"❌ Error loading configurations: {str(e)}")
        st.info("💡 Check if table CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS exists")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_config_from_database(_sess, config_id: str) -> Optional[Dict[str, Any]]:
    if not _sess:
        return None
    try:
        rows = _sess.sql(
            """
            SELECT CONFIG_ID, CONFIG_NAME, SOURCE_DB, SOURCE_SCHEMA, SOURCE_OBJECT,
                   COLUMN_INFO, YAML_TEXT, METADATA, CREATED_TIMESTAMP, LAST_UPDATED
            FROM CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS
            WHERE CONFIG_ID = ?
            """,
            [config_id]
        ).collect()
        if not rows:
            return None
        r = rows[0].as_dict()  # 🆕 FIXED: Use .as_dict() for Snowpark Row objects
        out = {
            "config_id": r["CONFIG_ID"],
            "config_name": r["CONFIG_NAME"],
            "source_db": r["SOURCE_DB"],
            "source_schema": r["SOURCE_SCHEMA"],
            "source_object": r["SOURCE_OBJECT"],
            "columns": pd.read_json(r["COLUMN_INFO"]) if r.get("COLUMN_INFO") else pd.DataFrame(),
            "yaml": r.get("YAML_TEXT", ""),
            "metadata": json.loads(r.get("METADATA", "{}")),
            "created": r.get("CREATED_TIMESTAMP"),
            "updated": r.get("LAST_UPDATED"),
        }
        return out
    except Exception as e:
        st.error(f"Load failed: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_databases() -> List[str]:
    try:
        df = _run("SHOW DATABASES")
        if df.empty:
            return []
        name_col = _pick_name_column(df, ["name", "database_name"]) or list(df.columns)[0]
        vals = df[name_col]
        # Defensive: cast non-strings to string
        return sorted(vals.astype(str).tolist())
    except Exception:
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_schemas(db: str) -> List[str]:
    try:
        df = _run(f"SHOW SCHEMAS IN DATABASE {db}")
        if df.empty:
            return []
        name_col = _pick_name_column(df, ["name", "schema_name"]) or list(df.columns)[0]
        vals = df[name_col]
        return sorted(vals.astype(str).tolist())
    except Exception:
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_tables_and_views(db: str, sc: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        tdf = _run(f"SELECT TABLE_NAME AS NAME, TABLE_TYPE AS TYPE FROM {db}.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '{sc}'")
        if not tdf.empty:
            name_col = _pick_name_column(tdf, ["name","table_name"]) or "NAME"
            type_col = _pick_name_column(tdf, ["type","table_type"]) or "TYPE"
            for _, r in tdf.iterrows():
                out.append((str(r[name_col]), str(r[type_col])))
    except Exception:
        pass
    try:
        vdf = _run(
            f"SELECT TABLE_NAME AS NAME, CASE WHEN IS_SECURE = 'YES' THEN 'SECURE VIEW' ELSE 'VIEW' END AS TYPE FROM {db}.INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA = '{sc}'"
        )
        if not vdf.empty:
            name_col = _pick_name_column(vdf, ["name","table_name"]) or "NAME"
            type_col = _pick_name_column(vdf, ["type"]) or "TYPE"
            for _, r in vdf.iterrows():
                out.append((str(r[name_col]), str(r[type_col])))
    except Exception:
        pass
    try:
        mvdf = _run(
            f"SELECT TABLE_NAME AS NAME, 'MATERIALIZED VIEW' AS TYPE FROM {db}.INFORMATION_SCHEMA.MATERIALIZED_VIEWS WHERE TABLE_SCHEMA = '{sc}'"
        )
        if not mvdf.empty:
            name_col = _pick_name_column(mvdf, ["name","table_name"]) or "NAME"
            type_col = _pick_name_column(mvdf, ["type"]) or "TYPE"
            for _, r in mvdf.iterrows():
                out.append((str(r[name_col]), str(r[type_col])))
    except Exception:
        pass
    return sorted(out, key=lambda x: (x[1], x[0]))


@st.cache_data(ttl=600)  # Cache for 10 minutes - column info changes less frequently
def describe_columns(db: str, sc: str, obj: str) -> pd.DataFrame:
    sql = f"""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM {db}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{sc}' AND TABLE_NAME = '{obj}'
    ORDER BY ORDINAL_POSITION
    """
    return _run(sql)


def find_pk_columns(db: str, sc: str, obj: str, cols_df: pd.DataFrame) -> List[str]:
    """Return primary key or GUID-like columns.
    1) Try information_schema constraints for PRIMARY KEY
    2) Fallback to name heuristics: id, *_id, guid, uuid, pk, primary_key, unique_id
    """
    pk_cols: List[str] = []
    try:
        q = f"""
        SELECT k.COLUMN_NAME
        FROM {db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
        JOIN {db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
          ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
         AND t.CONSTRAINT_SCHEMA = k.CONSTRAINT_SCHEMA
         AND t.CONSTRAINT_CATALOG = k.CONSTRAINT_CATALOG
        WHERE t.TABLE_SCHEMA = '{sc}'
          AND t.TABLE_NAME = '{obj}'
          AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY k.ORDINAL_POSITION
        """
        df = _run(q)
        if not df.empty:
            name_col = _pick_name_column(df, ["column_name"]) or list(df.columns)[0]
            pk_cols = [str(v) for v in df[name_col].tolist()]
    except Exception:
        pass
    if not pk_cols and not cols_df.empty:
        # Heuristics on column names
        name_col = _pick_name_column(cols_df, ["column_name","name"]) or list(cols_df.columns)[0]
        candidates = []
        for _, r in cols_df.iterrows():
            c = str(r[name_col])
            cl = c.lower()
            if cl == 'id' or cl.endswith('_id') or cl in ('guid','uuid','pk','primary_key','unique_id') or cl.endswith('_guid') or cl.endswith('_uuid'):
                candidates.append(c)
        # Keep unique order
        seen = set()
        pk_cols = [c for c in candidates if not (c in seen or seen.add(c))]
    return pk_cols


@st.cache_data(ttl=600)  # Cache for 10 minutes
def sample_rows(db: str, sc: str, obj: str, n: int = 20) -> pd.DataFrame:
    return _run(f"SELECT * FROM {db}.{sc}.{obj} LIMIT {int(n)}")


def classify_columns(cols: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_types = {"NUMBER","DECIMAL","NUMERIC","INT","INTEGER","BIGINT","SMALLINT","TINYINT","BYTEINT","FLOAT","DOUBLE","DOUBLE PRECISION","REAL"}
    date_types = {"DATE","TIME","TIMESTAMP","TIMESTAMP_LTZ","TIMESTAMP_NTZ","TIMESTAMP_TZ"}
    bool_types = {"BOOLEAN"}
    text_types = {"VARCHAR","STRING","TEXT","CHAR","CHARACTER"}

    out = {"numeric": [], "boolean": [], "date": [], "text": [], "other": []}
    for _, r in cols.iterrows():
        name = str(r["COLUMN_NAME"])  # type: ignore[index]
        dt = str(r["DATA_TYPE"]).upper()  # type: ignore[index]
        base_dt = dt.split("(")[0].strip()
        if base_dt in numeric_types:
            out["numeric"].append(name)
        elif base_dt in bool_types:
            out["boolean"].append(name)
        elif base_dt in date_types:
            out["date"].append(name)
        elif base_dt in text_types:
            out["text"].append(name)
        else:
            out["other"].append(name)
    return out


st.markdown("""
<div style='background: linear-gradient(to right, #1e40af, #0ea5e9); padding: 8px 12px; border-radius: 6px; margin-bottom: 10px;'>
  <h2 style='color: white; margin: 0;'>🧭 SnowViz YAML Wizard</h2>
  <div style='color: #e0f2fe; font-size: 12px;'>Configs saved in CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS</div>
</div>
""", unsafe_allow_html=True)

# Setup config store
if session is not None and setup_config_database(session):
    st.success("Config store ready: CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS")

# Mode selection: create new or modify existing
mode = st.radio("Mode", ["Create new", "Modify existing"], horizontal=True)

# If modifying, allow user to pick and load a saved configuration
if mode == "Modify existing" and session is not None:
    saved_for_edit = load_saved_configurations(session)
    if saved_for_edit:
        st.markdown("#### Modify existing configuration")
        opt_labels = [f"{r['CONFIG_NAME']} ({r['SOURCE_DB']}.{r['SOURCE_SCHEMA']}.{r['SOURCE_OBJECT']})" for r in saved_for_edit]
        pick_label = st.selectbox("Select configuration", ["Select..."] + opt_labels)
        if pick_label != "Select...":
            if st.button("Load configuration"):
                idx = opt_labels.index(pick_label)
                loaded_cfg = load_config_from_database(session, saved_for_edit[idx]["CONFIG_ID"])
                if loaded_cfg:
                    meta = loaded_cfg.get("metadata", {}) or {}
                    st.session_state["svw_saved_db"] = loaded_cfg.get("source_db")
                    st.session_state["svw_saved_schema"] = loaded_cfg.get("source_schema")
                    st.session_state["svw_saved_object"] = loaded_cfg.get("source_object")
                    st.session_state["svw_saved_dimensions"] = meta.get("dimensions", [])
                    st.session_state["svw_saved_metrics"] = meta.get("selected_metrics", [])
                    st.session_state["svw_saved_time_col"] = meta.get("time_col")
                    st.session_state["svw_saved_filename"] = meta.get("filename") or f"{loaded_cfg.get('source_object','').lower()}_snowviz.yaml"
                    st.session_state["svw_yaml_text"] = loaded_cfg.get("yaml", "")
                    # 🆕 CRITICAL FIX: Store loaded config name for save dialog
                    st.session_state["svw_loaded_config_name"] = loaded_cfg.get("config_name", f"snowviz_{loaded_cfg.get('source_object','').lower()}")
                    
                    # 🆕 CRITICAL FIX: Restore dimension and metric customizations
                    if "dimension_customizations" in meta and meta["dimension_customizations"]:
                        st.session_state["dim_customs"] = meta["dimension_customizations"]
                        st.success(f"✅ Restored {len(meta['dimension_customizations'])} dimension customizations")
                    else:
                        st.warning("⚠️ No dimension_customizations found in metadata")
                    
                    if "metric_customizations" in meta and meta["metric_customizations"]:
                        st.session_state["met_customs"] = meta["metric_customizations"]
                        st.success(f"✅ Restored {len(meta['metric_customizations'])} metric customizations")
                    else:
                        st.warning("⚠️ No metric_customizations found in metadata")
                    
                    st.success("✅ Configuration loaded with all customizations restored!")
    else:
        st.info("No saved configurations found.")

st.markdown("### Source Selection")
col1, col2, col3 = st.columns(3)
with col1:
    dbs = list_databases()
    saved_db = st.session_state.get("svw_saved_db")
    db_default_index = (dbs.index(saved_db) if (saved_db in dbs) else 0) if dbs else 0
    db = st.selectbox("Database", options=dbs, index=db_default_index)
with col2:
    if db:
        schemas = list_schemas(db)
        saved_sc = st.session_state.get("svw_saved_schema")
        sc_default_index = (schemas.index(saved_sc) if (saved_sc in schemas) else 0) if schemas else 0
        sc = st.selectbox("Schema", options=schemas, index=sc_default_index)
    else:
        sc = None
with col3:
    if db and sc:
        objs = list_tables_and_views(db, sc)
        obj_display = [f"{name} ({otype})" for name, otype in objs]
        saved_obj = st.session_state.get("svw_saved_object")
        default_disp = None
        if saved_obj:
            for name, otype in objs:
                if name == saved_obj:
                    default_disp = f"{name} ({otype})"
                    break
        disp_index = (obj_display.index(default_disp) if (default_disp in obj_display) else 0) if obj_display else 0
        selected_disp = st.selectbox("Table / View", options=obj_display if obj_display else [""], index=disp_index)
        if obj_display:
            try:
                sel_idx = obj_display.index(selected_disp)
            except ValueError:
                sel_idx = None
        else:
            sel_idx = None
    else:
        objs = []
        obj_display = []
        sel_idx = None

cfg: Dict[str, Any] = {}
yaml_text = ""
cols_df_cache: Optional[pd.DataFrame] = None

if db and sc and sel_idx is not None:
    selected_name, selected_type = objs[sel_idx]
    if selected_type == "SECURE VIEW":
        st.error("Secure views are not supported for this wizard. Pick a table or non-secure view.")
    else:
        st.success(f"Selected: {db}.{sc}.{selected_name} ({selected_type})")

        cols_df = describe_columns(db, sc, selected_name)
        cols_df_cache = cols_df.copy()
        st.markdown("### Columns")
        st.dataframe(cols_df, use_container_width=True, height=260)

        st.markdown("### Sample (first 20 rows)")
        st.dataframe(sample_rows(db, sc, selected_name), use_container_width=True, height=260)

        classes = classify_columns(cols_df)
        st.markdown("### Configure Dimensions, Metrics, Time Column")
        c1, c2 = st.columns(2)
        with c1:
            dim_candidates = sorted(classes["text"], key=lambda s: s.lower())
            saved_dims = st.session_state.get("svw_saved_dimensions", [])
            default_dims = [d for d in dim_candidates if d in saved_dims] if saved_dims else []
            dimensions = st.multiselect(
                "Dimensions",
                options=dim_candidates,
                default=default_dims,
                help="Choose text/categorical fields for grouping and filtering"
            )
        with c2:
            date_candidates = sorted(classes["date"], key=lambda s: s.lower())
            saved_time = st.session_state.get("svw_saved_time_col")
            t_index = (date_candidates.index(saved_time) if (saved_time in date_candidates) else 0) if date_candidates else None
            time_col = st.selectbox(
                "Time Column (date spine)",
                options=date_candidates,
                index=t_index
            )

        # Metrics
        metric_defs: List[Dict[str, Any]] = []
        st.markdown("#### Metrics")
        st.caption("Defaults: count(*), plus SUM/AVG/MIN/MAX for numeric, AVG for boolean")
        metric_defs.append({"key": "total_rows", "label": "Total Rows", "sql": "COUNT(*)", "format": "integer", "round": 0})

        def _add_metric(key: str, label: str, sql_expr: str, fmt: str = "number", rnd: int = 2):
            metric_defs.append({"key": key, "label": label, "sql": sql_expr, "format": fmt, "round": rnd})

        # Primary key / GUID candidates -> offer COUNT and COUNT DISTINCT
        pk_candidates = find_pk_columns(db, sc, selected_name, cols_df)
        for col in pk_candidates:
            base = col.lower()
            # Ensure unique keys
            count_key = f"count_{base}"
            cdist_key = f"count_distinct_{base}"
            if any(m["key"] == count_key for m in metric_defs):
                count_key += "_1"
            if any(m["key"] == cdist_key for m in metric_defs):
                cdist_key += "_1"
            _add_metric(count_key, f"Count {col}", f"COUNT({col})", fmt="integer", rnd=0)
            _add_metric(cdist_key, f"Count Distinct {col}", f"COUNT(DISTINCT {col})", fmt="integer", rnd=0)

        # Numeric aggregations
        for col in classes["numeric"]:
            base = col.lower()
            _add_metric(f"avg_{base}", f"Avg {col}", f"AVG({col})")
            _add_metric(f"sum_{base}", f"Sum {col}", f"SUM({col})")
            _add_metric(f"min_{base}", f"Min {col}", f"MIN({col})")
            _add_metric(f"max_{base}", f"Max {col}", f"MAX({col})")

        # Boolean aggregations (treat as 0/1)
        for col in classes["boolean"]:
            base = col.lower()
            _add_metric(f"pct_{base}", f"% {col}", f"AVG(COALESCE({col}, 0))", fmt="percent", rnd=3)

        # Allow users to curate metrics
        metric_keys = sorted([m["key"] for m in metric_defs], key=lambda s: s.lower())
        saved_metrics = st.session_state.get("svw_saved_metrics", [])
        default_metric_keys = [m for m in metric_keys if m in saved_metrics] if saved_metrics else []
        selected_metric_keys = st.multiselect(
            "Select metrics to include",
            options=metric_keys,
            default=default_metric_keys,
            help="Choose which metrics to include in your dashboard"
        )
        metric_defs = [m for m in metric_defs if m["key"] in selected_metric_keys]

        # Cards defaults (empty - user selects what they want)
        default_cards = []

        # =====================================
        # CUSTOMIZATION INTERFACE
        # =====================================
        
        st.markdown("### 🎨 Customize Dashboard Configuration")
        st.markdown("Set up labels, descriptions, and generate your YAML")
        
        # Always use advanced customization mode
        # 🆕 WORKAROUND: Add navigation help since Streamlit tabs reset after form submissions
        st.info("💡 **Tab Navigation Tip**: After clicking 'Apply Changes' or 'Save' buttons, you'll be taken back to the first tab. Just click on the tab you want to return to.")
        
        # Customization tabs
        dim_tab, metric_tab, intelligence_tab, yaml_tab = st.tabs(["📊 Dimensions", "📈 Metrics", "🤖 Intelligence", "🚀 Generate"])
            
            # ===================
            # DIMENSIONS TAB
            # ===================
        with dim_tab:
                st.subheader("📊 Customize Dimensions")
                st.info("💡 Make your changes below, then click 'Apply All Dimension Changes' to save them.")
                
                # Initialize dimension customizations
                if "dim_customs" not in st.session_state:
                    st.session_state.dim_customs = {
                        dim: {
                            "included": True,
                            "label": dim.replace("_", " ").title(),
                            "description": f"Analysis dimension for {dim.lower().replace('_', ' ')}",
                            "order": i
                        }
                        for i, dim in enumerate(dimensions)
                    }
                
                # Use form to batch all dimension changes and prevent page refresh
                with st.form("dimensions_form"):
                    # Edit each dimension
                    dimension_changes = {}  # Store changes temporarily
                    for i, dim in enumerate(dimensions):
                        # Force expansion by using container instead of expander for better UX
                        st.markdown(f"### 🔧 {dim}")
                        with st.container():
                            st.markdown("---")  # Visual separator
                            col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
                            
                            with col1:
                                default_included = st.session_state.dim_customs.get(dim, {}).get("included", True)
                                incl = st.checkbox("Include", value=default_included, key=f"dinc_{dim}")
                            
                            with col2:
                                default_label = st.session_state.dim_customs.get(dim, {}).get("label", dim.replace("_", " ").title())
                                lbl = st.text_input("Label", value=default_label, key=f"dlbl_{dim}")
                            
                            with col3:
                                default_desc = st.session_state.dim_customs.get(dim, {}).get("description", f"Analysis dimension for {dim.lower().replace('_', ' ')}")
                                desc = st.text_area("Description", value=default_desc, key=f"ddsc_{dim}", height=50)
                            
                            with col4:
                                default_order = st.session_state.dim_customs.get(dim, {}).get("order", i)
                                ord = st.number_input("Priority", min_value=0, value=default_order, key=f"dord_{dim}")
                            
                            # Store changes for batch update
                            dimension_changes[dim] = {
                                "included": incl,
                                "label": lbl,
                                "description": desc,
                                "order": ord
                            }
                    
                    # Single form submit button to apply all changes
                    dimensions_form_submitted = st.form_submit_button("💾 Apply All Dimension Changes")
                    
                    # Only update session state when form is submitted
                    if dimensions_form_submitted:
                        for dim, changes in dimension_changes.items():
                            # Initialize the dimension in session state if it doesn't exist
                            if dim not in st.session_state.dim_customs:
                                st.session_state.dim_customs[dim] = {}
                            st.session_state.dim_customs[dim].update(changes)
                        st.success(f"✅ All dimension changes applied! Saved {len(dimension_changes)} customizations to session state.")
                        st.info("💡 **Next Step**: Click on the 'Metrics' tab to customize your metrics, or 'Generate' tab to create your YAML.")
                
                # Apply customizations to dimensions list
                customized_dimensions = []
                for dim in dimensions:
                    dim_config = st.session_state.dim_customs.get(dim, {})
                    if dim_config.get("included", True):  # Default include if not specified
                        customized_dimensions.append({
                            "key": dim,
                            "label": dim_config.get("label", dim.replace("_", " ").title()),
                            "column": dim,
                            "type": "categorical",
                            "description": dim_config.get("description", f"Analysis dimension for {dim.lower().replace('_', ' ')}")
                        })
                
                # Sort by order
                customized_dimensions = sorted(customized_dimensions, key=lambda x: st.session_state.dim_customs.get(x["key"], {}).get("order", 0))
                st.info(f"✅ {len(customized_dimensions)} dimensions configured")
                
                # Show top dimensions preview
                if customized_dimensions:
                    st.markdown("**📊 Top Priority Dimensions:**")
                    for i, d in enumerate(customized_dimensions[:6]):
                        priority = st.session_state.dim_customs.get(d["key"], {}).get("order", 0)
                        st.write(f"{i+1}. **{d['label']}** (priority: {priority}) - {d['description'][:50]}...")
                
                # Section to verify saving
                with st.expander("🔧 Current Dimension Settings", expanded=False):
                    st.json(st.session_state.get("dim_customs", {}))
            
            # ===================
            # METRICS TAB
            # ===================
        with metric_tab:
                render_metrics_tab(metric_defs)
            
            # ===================
            # INTELLIGENCE TAB  
            # ===================
        with intelligence_tab:
                render_intelligence_tab(session, db, sc, selected_name, dimensions, metric_defs, st.session_state.get("svw_saved_filename", f"{selected_name.lower()}_snowviz.yaml"))
            
            # ===================
            # GENERATE YAML TAB
            # ===================
        with yaml_tab:
                render_yaml_generation_tab(session, db, sc, selected_name, dimensions, metric_defs, time_col, cols_df_cache)


# Saved configurations (reload)
if session is not None:
    st.markdown("### Saved Configurations (CORTEX_AI_FRAMEWORK_DB.CONFIGS)")
    saved = load_saved_configurations(session)
    if saved:
        col_l, col_r = st.columns([3,1])
        with col_l:
            show = {f"{r['CONFIG_NAME']} ({r['SOURCE_DB']}.{r['SOURCE_SCHEMA']}.{r['SOURCE_OBJECT']})": r['CONFIG_ID'] for r in saved}
            pick = st.selectbox("Select config", ["Select..."] + list(show.keys()))
        with col_r:
            if pick and pick != "Select...":
                if st.button("🔄 Load"):
                    loaded = load_config_from_database(session, show[pick])
                    if loaded:
                        # Load basic configuration
                        meta = loaded.get("metadata", {}) or {}
                        st.session_state["svw_saved_db"] = loaded.get("source_db")
                        st.session_state["svw_saved_schema"] = loaded.get("source_schema")
                        st.session_state["svw_saved_object"] = loaded.get("source_object")
                        st.session_state["svw_saved_dimensions"] = meta.get("dimensions", [])
                        st.session_state["svw_saved_metrics"] = meta.get("selected_metrics", [])
                        st.session_state["svw_saved_time_col"] = meta.get("time_col")
                        st.session_state["svw_saved_filename"] = meta.get("filename") or f"{loaded.get('source_object','').lower()}_snowviz.yaml"
                        st.session_state["svw_yaml_text"] = loaded.get("yaml", "")
                        # 🆕 CRITICAL FIX: Store loaded config name for save dialog
                        st.session_state["svw_loaded_config_name"] = loaded.get("config_name", f"snowviz_{loaded.get('source_object','').lower()}")
                        
                        # 🆕 CRITICAL FIX: Restore dimension and metric customizations
                        customizations_restored = 0
                        
                        if "dimension_customizations" in meta and meta["dimension_customizations"]:
                            st.session_state["dim_customs"] = meta["dimension_customizations"]
                            customizations_restored += len(meta["dimension_customizations"])
                        else:
                            st.warning("⚠️ No dimension_customizations found in metadata")
                        
                        if "metric_customizations" in meta and meta["metric_customizations"]:
                            st.session_state["met_customs"] = meta["metric_customizations"]
                            customizations_restored += len(meta["metric_customizations"])
                        else:
                            st.warning("⚠️ No metric_customizations found in metadata")
                        
                        st.success(f"✅ Loaded: {loaded['config_name']} with {customizations_restored} customizations restored!")
                        st.code(loaded.get("yaml", ""), language="yaml")
    else:
        st.info("No configs saved yet.")
        # Check if config table exists
        if session:
            try:
                table_check = session.sql("SELECT COUNT(*) as total_configs FROM CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS").collect()
                total = table_check[0]['TOTAL_CONFIGS'] if table_check else 0
                st.info(f"💾 Configuration table has {total} saved configs")
            except Exception as e:
                st.warning(f"⚠️ Cannot access config table")
                st.info("💡 Make sure CORTEX_AI_FRAMEWORK_DB.CONFIGS.SNOWVIZ_CONFIGURATIONS table exists")
