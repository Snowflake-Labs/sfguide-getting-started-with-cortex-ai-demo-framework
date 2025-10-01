# Data Transformer - Structured Tables Pipeline
# Automatically cleans LLM garbage and transforms JSON to structured tables

import streamlit as st
import json
import re
import os
from typing import List, Optional
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# Page configuration
st.set_page_config(
    page_title="Data Transformer - Structured Tables",
    page_icon="🔄",
    layout="wide"
)

# Get Snowflake session
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

# Title and description
st.title("🔄 Data Transformer - Structured Tables")
st.markdown("Transform synthetic JSON data into structured tables ready for analysis")

def setup_database_infrastructure(session: Session) -> bool:
    """DataOps version - infrastructure is managed by DataOps, just verify it exists"""
    # DataOps creates all necessary schemas (BRONZE_LAYER, SILVER_LAYER, CONFIGS)
    # We just need to verify we can access them
    try:
        # Test if we can access the schemas by running a simple query
        session.sql(f"SHOW SCHEMAS IN DATABASE {db_name}").collect()
        return True
    except Exception as e:
        st.error(f"Cannot access database infrastructure: {str(e)}. Ensure DataOps deployment completed successfully.")
        return False

def get_available_tables(session: Session) -> List[str]:
    """Get list of available tables with messages column from BRONZE_LAYER"""
    try:
        if not setup_database_infrastructure(session):
            return []
        
        result = session.sql(f"""
            SELECT TABLE_NAME
            FROM {db_name}.INFORMATION_SCHEMA.TABLES t
            WHERE t.TABLE_SCHEMA = 'BRONZE_LAYER'
            AND EXISTS (
                SELECT 1 FROM {db_name}.INFORMATION_SCHEMA.COLUMNS c 
                WHERE c.TABLE_NAME = t.TABLE_NAME 
                AND c.TABLE_SCHEMA = 'BRONZE_LAYER'
                AND c.COLUMN_NAME = 'MESSAGES'
            )
            ORDER BY TABLE_NAME
        """).collect()
        
        return [row['TABLE_NAME'] for row in result]
    except Exception as e:
        st.error(f"Error getting tables: {str(e)}")
        return []

def get_available_companies(session: Session, table_name: str) -> List[str]:
    """Get list of available companies from the selected table"""
    try:
        result = session.sql(f"""
            SELECT DISTINCT _meta_company_name
            FROM {db_name}.BRONZE_LAYER.{table_name}
            WHERE _meta_company_name IS NOT NULL
            ORDER BY _meta_company_name
        """).collect()
        
        return [row['_META_COMPANY_NAME'] for row in result]
    except Exception as e:
        st.error(f"Error getting companies: {str(e)}")
        return []

def get_available_topics(session: Session, table_name: str) -> List[str]:
    """Get list of available topics from the selected table"""
    try:
        result = session.sql(f"""
            SELECT DISTINCT _meta_topic
            FROM {db_name}.BRONZE_LAYER.{table_name}
            WHERE _meta_topic IS NOT NULL
            ORDER BY _meta_topic
        """).collect()
        
        return [row['_META_TOPIC'] for row in result]
    except Exception as e:
        st.error(f"Error getting topics: {str(e)}")
        return []

def auto_clean_llm_garbage(session: Session, source_table: str, company_name: Optional[str] = None, topic: Optional[str] = None) -> str:
    """AUTOMATICALLY clean common LLM garbage - no user intervention needed"""
    try:
        # Build WHERE clause for filtering
        where_clause = "WHERE messages IS NOT NULL AND LENGTH(messages) > 0"
        if company_name:
            where_clause += f" AND _meta_company_name = '{company_name}'"
        if topic:
            where_clause += f" AND _meta_topic = '{topic}'"
        
        # Check if we have any records first
        count_check = session.sql(f"""
            SELECT COUNT(*) as record_count
            FROM {db_name}.BRONZE_LAYER.{source_table}
            {where_clause}
        """).collect()
        
        if not count_check or count_check[0]['RECORD_COUNT'] == 0:
            st.warning("No records found matching the filter criteria")
            return source_table
        
        # Get the actual field names from _meta_fields_requested
        fields_result = session.sql(f"""
            SELECT DISTINCT _meta_fields_requested
            FROM {db_name}.BRONZE_LAYER.{source_table}
            {where_clause}
            AND _meta_fields_requested IS NOT NULL
            LIMIT 1
        """).collect()
        
        # Parse the field names (comma-separated)
        first_field = None
        if fields_result and fields_result[0]['_META_FIELDS_REQUESTED']:
            field_data = fields_result[0]['_META_FIELDS_REQUESTED']
            # Split by comma and take the first field
            fields_list = [f.strip() for f in field_data.split(',')]
            if fields_list and fields_list[0]:
                first_field = fields_list[0]
        
        # Debug: Show what we found
        if fields_result:
            st.write(f"DEBUG: _meta_fields_requested = {fields_result[0]['_META_FIELDS_REQUESTED']}")
        else:
            st.write("DEBUG: No _meta_fields_requested found")
        st.write(f"DEBUG: first_field = {first_field}")
        
        # TEMPORARY: Hardcode field name for testing
        if first_field is None:
            first_field = "review_comment"  # Hardcode based on your data
            st.write(f"DEBUG: Using hardcoded field: {first_field}")
        
        cleaned_table = f"{source_table}_AUTO_CLEANED"
        
        # Build the cleaning SQL using simple string functions
        happy_starting_string = '{"choices":[{"messages":"[{"'
        
        if first_field:
            field_to_find = first_field
        else:
            field_to_find = ''
        st.write(f"field to find: {field_to_find}")
        
        # Build dynamic truncation repair pattern
        if first_field:
            truncation_pattern = f"""
                CASE 
                    WHEN step1_cleaned IS NULL OR LENGTH(step1_cleaned) = 0 THEN step1_cleaned
                    WHEN step1_cleaned LIKE '%"{first_field}": "%' 
                         AND step1_cleaned NOT LIKE '%"}}%'
                         AND step1_cleaned NOT LIKE '%"]%' 
                    THEN step1_cleaned || '"}}]'
                    
                    WHEN step1_cleaned LIKE '%[%' 
                         AND step1_cleaned NOT LIKE '%]%'
                    THEN step1_cleaned || ']'
                    
                    WHEN step1_cleaned LIKE '%{{%' 
                         AND step1_cleaned NOT LIKE '%}}%'
                    THEN step1_cleaned || '}}'
                    
                    ELSE step1_cleaned
                END
            """
        else:
            # Fallback to generic patterns if we can't get field names
            truncation_pattern = f"""
                CASE 
                    WHEN step1_cleaned IS NULL OR LENGTH(step1_cleaned) = 0 THEN step1_cleaned
                    WHEN step1_cleaned LIKE '%": "%' 
                         AND step1_cleaned NOT LIKE '%"}}%'
                         AND step1_cleaned NOT LIKE '%"]%' 
                    THEN step1_cleaned || '"}}]'
                    
                    WHEN step1_cleaned LIKE '%[%' 
                         AND step1_cleaned NOT LIKE '%]%'
                    THEN step1_cleaned || ']'
                    
                    WHEN step1_cleaned LIKE '%{{%' 
                         AND step1_cleaned NOT LIKE '%}}%'
                    THEN step1_cleaned || '}}'
                    
                    ELSE step1_cleaned
                END
            """
        
        # Automatically clean all common LLM garbage
        session.sql(f"""
            CREATE OR REPLACE TABLE {db_name}.BRONZE_LAYER.{cleaned_table} AS
            SELECT 
                *,
                -- Step 1: Clean approach - happy string + everything after first field
                CASE 
                    WHEN messages IS NULL OR LENGTH(messages) = 0 THEN messages
                    WHEN POSITION('{field_to_find}' IN messages) > 0 THEN CONCAT(
                        '{happy_starting_string}',
                        REPLACE(
                            RIGHT(messages, LENGTH(messages) - POSITION('{field_to_find}' IN messages) + 1),
                            '""', '"'
                        )
                    )
                    ELSE messages
                END as step1_cleaned,
                
                -- Debug columns to track down the issue
                '{field_to_find}' as field_detected,
                POSITION('{field_to_find}' IN messages) as start_of_field,
                LENGTH(messages) as length_message,
                '{{' || RIGHT(messages, LENGTH(messages) - POSITION('{field_to_find}' IN messages) + 2) as rest_of_string,
                       
                -- Step 2: Fix common truncation patterns (using dynamic field names)
                {truncation_pattern} as cleaned_messages                
            FROM {db_name}.BRONZE_LAYER.{source_table}
            {where_clause}
        """).collect()
        
        # Verify the cleaned table was created successfully
        verify_result = session.sql(f"""
            SELECT COUNT(*) as cleaned_count
            FROM {db_name}.BRONZE_LAYER.{cleaned_table}
        """).collect()
        
        if verify_result and verify_result[0]['CLEANED_COUNT'] > 0:
            return cleaned_table
        else:
            st.error("Failed to create cleaned table - using original table")
            return source_table
        
    except Exception as e:
        st.error(f"Error auto-cleaning LLM garbage: {str(e)}")
        return source_table

def analyze_data_quality(session: Session, source_table: str, company_name: Optional[str] = None, topic: Optional[str] = None) -> dict:
    """Analyze data quality - SIMPLIFIED for clean JSON data"""
    try:
        # For clean data, we don't need the complex auto-cleaning logic
        # Just use the original messages column
        column_name = "messages"
        
        # Build WHERE clause for filtering
        where_clause = f"WHERE {column_name} IS NOT NULL AND LENGTH({column_name}) > 0"
        if company_name:
            where_clause += f" AND _meta_company_name = '{company_name}'"
        if topic:
            where_clause += f" AND _meta_topic = '{topic}'"
        
        # Simple analysis for clean data
        result = session.sql(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN TRY_PARSE_JSON({column_name}) IS NOT NULL THEN 1 END) as valid_json_records,
                COALESCE(AVG(LENGTH({column_name})), 0) as avg_message_length
            FROM {db_name}.BRONZE_LAYER.{source_table}
            {where_clause}
        """).collect()
        
        if result:
            stats = result[0]
            return {
                'cleaned_table': source_table,  # No cleaning needed for clean data
                'column_name': column_name,
                'total_records': stats['TOTAL_RECORDS'] or 0,
                'valid_json_records': stats['VALID_JSON_RECORDS'] or 0, 
                'invalid_json_records': (stats['TOTAL_RECORDS'] or 0) - (stats['VALID_JSON_RECORDS'] or 0),
                'very_short_records': 0,  # Not relevant for clean data
                'avg_message_length': stats['AVG_MESSAGE_LENGTH'] or 0
            }
        else:
            return {
                'cleaned_table': source_table,
                'column_name': column_name,
                'total_records': 0,
                'valid_json_records': 0,
                'invalid_json_records': 0,
                'very_short_records': 0,
                'avg_message_length': 0
            }
        
    except Exception as e:
        st.error(f"Error analyzing data quality: {str(e)}")
        return {
            'cleaned_table': source_table,
            'column_name': 'messages',
            'total_records': 0,
            'valid_json_records': 0,
            'invalid_json_records': 0,
            'very_short_records': 0,
            'avg_message_length': 0
        }

def get_sample_data_for_display(session: Session, cleaned_table: str, column_name: str, company_name: Optional[str] = None, topic: Optional[str] = None) -> List[dict]:
    """Get sample data for display with proper formatting"""
    try:
        # Build WHERE clause for filtering
        where_clause = f"WHERE {column_name} IS NOT NULL AND LENGTH({column_name}) > 0"
        if company_name:
            where_clause += f" AND _meta_company_name = '{company_name}'"
        if topic:
            where_clause += f" AND _meta_topic = '{topic}'"
        
        # Get sample data with safe preview handling
        sample_result = session.sql(f"""
            SELECT 
                _meta_company_name,
                _meta_topic,
                LENGTH({column_name}) as message_length,
                CASE 
                    WHEN TRY_PARSE_JSON({column_name}) IS NOT NULL THEN 'VALID'
                    ELSE 'INVALID'
                END as json_status,
                CASE 
                    WHEN LENGTH({column_name}) < 50 THEN 'VERY_SHORT'
                    WHEN LENGTH({column_name}) < 200 THEN 'SHORT'
                    WHEN LENGTH({column_name}) < 1000 THEN 'MEDIUM'
                    ELSE 'LONG'
                END as length_category,
                CASE 
                    WHEN {column_name} IS NULL THEN 'NULL'
                    WHEN LENGTH({column_name}) = 0 THEN 'EMPTY'
                    WHEN LENGTH({column_name}) <= 100 THEN {column_name}
                    WHEN LENGTH({column_name}) > 100 THEN CONCAT(LEFT({column_name}, 100), '...')
                    ELSE {column_name}
                END as preview
            FROM {db_name}.BRONZE_LAYER.{cleaned_table}
            {where_clause}
            ORDER BY RANDOM()
            LIMIT 10
        """).collect()
        
        sample_data = []
        for row in sample_result:
            sample_data.append({
                'Company': row['_META_COMPANY_NAME'],
                'Topic': row['_META_TOPIC'],
                'Length': row['MESSAGE_LENGTH'],
                'Size': row['LENGTH_CATEGORY'],
                'JSON Status': row['JSON_STATUS'],
                'Preview': row['PREVIEW'] if row['PREVIEW'] else 'Empty'
            })
        
        return sample_data
        
    except Exception as e:
        st.error(f"Error getting sample data: {str(e)}")
        return []

def clean_field_name(field_name: str) -> str:
    """Clean field name for SQL compatibility"""
    cleaned = re.sub(r'[^\w]', '_', field_name)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned.upper()

def extract_fields_from_cleaned_data(session: Session, cleaned_table: str, column_name: str, company_name: Optional[str] = None, topic: Optional[str] = None) -> List[str]:
    """Extract available fields from cleaned JSON data - SIMPLIFIED for clean JSON arrays"""
    try:
        # Build WHERE clause for filtering
        where_clause = f"WHERE {column_name} IS NOT NULL"
        if company_name:
            where_clause += f" AND _meta_company_name = '{company_name}'"
        if topic:
            where_clause += f" AND _meta_topic = '{topic}'"
        
        # Get sample data to understand structure
        sample_result = session.sql(f"""
            SELECT {column_name}
            FROM {db_name}.BRONZE_LAYER.{cleaned_table}
            {where_clause}
            LIMIT 1
        """).collect()
        
        if not sample_result:
            return []
        
        # Extract field names from clean JSON array
        all_fields = set()
        for row in sample_result:
            try:
                # Get the column value - convert row to dict for easier access
                row_dict = row.as_dict()
                column_value = row_dict.get(column_name.upper()) or row_dict.get(column_name)
                
                if not column_value:
                    continue
                    
                # Parse the JSON - expecting clean array format like [{"field1": "value1", ...}]
                json_data = json.loads(column_value)
                
                # Handle clean JSON array directly
                if isinstance(json_data, list) and len(json_data) > 0:
                    # Get fields from first object in array
                    first_item = json_data[0]
                    if isinstance(first_item, dict):
                        all_fields.update(first_item.keys())
                elif isinstance(json_data, dict):
                    # Single object case
                    all_fields.update(json_data.keys())
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                st.warning(f"Error parsing JSON: {str(e)}")
                continue
        
        return sorted(list(all_fields))
        
    except Exception as e:
        st.error(f"Error extracting fields: {str(e)}")
        return []

def transform_to_structured_table(session: Session, cleaned_table: str, column_name: str, target_table: str, fields: List[str], company_name: Optional[str] = None, topic: Optional[str] = None) -> bool:
    """Transform cleaned JSON data to structured table - SIMPLIFIED for clean JSON arrays"""
    try:
        # Clean field names for SQL
        sql_fields = [clean_field_name(field) for field in fields]
        
        # Create target table with dynamic columns
        columns_sql = ', '.join([f"{sql_field} STRING" for sql_field in sql_fields])
        
        session.sql(f"""
            CREATE OR REPLACE TABLE {db_name}.SILVER_LAYER.{target_table} (
                id STRING,
                {columns_sql},
                company_name STRING,
                topic STRING,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """).collect()
        
        # Build WHERE clause for filtering
        where_clause = f"WHERE TRY_PARSE_JSON({column_name}) IS NOT NULL"
        if company_name:
            where_clause += f" AND _meta_company_name = '{company_name}'"
        if topic:
            where_clause += f" AND _meta_topic = '{topic}'"
        
        # Create field extractions
        field_extractions = []
        for original_field, sql_field in zip(fields, sql_fields):
            field_extractions.append(f"flattened_data.value:{original_field}::STRING AS {sql_field}")
        
        fields_sql = ', '.join(field_extractions)
        
        # Insert transformed data with deduplication - SIMPLIFIED for clean JSON arrays 
        session.sql(f"""
            INSERT INTO {db_name}.SILVER_LAYER.{target_table} (id, {', '.join(sql_fields)}, company_name, topic)
            SELECT DISTINCT
                UUID_STRING() as id,
                {fields_sql},
                _meta_company_name as company_name,
                _meta_topic as topic
            FROM {db_name}.BRONZE_LAYER.{cleaned_table},
            LATERAL FLATTEN(
                input => TRY_PARSE_JSON({column_name})
            ) AS flattened_data
            {where_clause}
        """).collect()
        
        return True
        
    except Exception as e:
        st.error(f"Error transforming data: {str(e)}")
        return False

def save_transformation_config(session: Session, config_name: str, source_table: str, target_table: str, company_name: str, topic: str, fields: List[str]) -> bool:
    """Save transformation configuration"""
    try:
        # Create table with simpler structure - using CONFIGS schema to match DataOps and other apps
        session.sql(f"""
            CREATE TABLE IF NOT EXISTS {db_name}.CONFIGS.TRANSFORMATION_CONFIGS (
                CONFIG_NAME STRING,
                SOURCE_TABLE STRING,
                TARGET_TABLE STRING,
                COMPANY_NAME STRING,
                TOPIC STRING,
                FIELDS_REQUESTED STRING,
                CREATED_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """).collect()
        
        # Delete existing config if it exists
        session.sql(f"""
            DELETE FROM {db_name}.CONFIGS.TRANSFORMATION_CONFIGS 
            WHERE CONFIG_NAME = ?
        """, [config_name]).collect()
        
        # Insert new config
        session.sql(f"""
            INSERT INTO {db_name}.CONFIGS.TRANSFORMATION_CONFIGS 
            (CONFIG_NAME, SOURCE_TABLE, TARGET_TABLE, COMPANY_NAME, TOPIC, FIELDS_REQUESTED)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [config_name, source_table, target_table, company_name, topic, json.dumps(fields)]).collect()
        
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

# Main UI
st.header("📊 Transform Synthetic Data")
st.markdown("**BRONZE → SILVER Layer Transformation**")

# Get available tables
available_tables = get_available_tables(session)

if not available_tables:
    st.warning("No tables with synthetic data found in BRONZE_LAYER. Please run the Synthetic Data Generator first.")
    st.stop()

# Table selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Table (BRONZE_LAYER)")
    source_table = st.selectbox(
        "Select source table with synthetic data:",
        available_tables,
        help="Tables containing generated synthetic data with 'messages' column"
    )

with col2:
    st.subheader("Target Table (SILVER_LAYER)")
    target_table = st.text_input(
        "Enter name for structured table:",
        value=f"{source_table}_STRUCTURED" if source_table else "",
        help="Name for the new structured table (e.g., QUALTRICS_REVIEWS_BULLS)"
    )

# Company and Topic selection
if source_table:
    st.subheader("🏢 Filter by Company and Topic")
    
    # Get available companies and topics
    companies = get_available_companies(session, source_table)
    topics = get_available_topics(session, source_table)
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.selectbox(
            "Select Company:",
            companies,
            help="Filter data by company name"
        ) if companies else None
    
    with col2:
        topic = st.selectbox(
            "Select Topic:",
            topics,
            help="Filter data by topic"
        ) if topics else None

    # Get fields from selected table with filters
    if company_name and topic:
        st.markdown("---")
        st.subheader("🔄 Data Processing")
        
        # Analyze data quality first
        with st.spinner("Analyzing data quality..."):
            data_quality_stats = analyze_data_quality(session, source_table, company_name, topic)
        
        # Show data quality results
        st.subheader("📊 Data Quality Analysis")
        if data_quality_stats and data_quality_stats.get('total_records', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", data_quality_stats.get('total_records', 0))
                st.metric("Valid JSON", data_quality_stats.get('valid_json_records', 0))
            with col2:
                st.metric("Invalid JSON", data_quality_stats.get('invalid_json_records', 0))
                st.metric("Very Short", data_quality_stats.get('very_short_records', 0))
            with col3:
                st.metric("Avg Length", f"{data_quality_stats.get('avg_message_length', 0):.0f} chars")

            # Get sample data for display
            sample_data = get_sample_data_for_display(
                session, 
                data_quality_stats['cleaned_table'], 
                data_quality_stats['column_name'], 
                company_name, 
                topic
            )
            
            if sample_data:
                st.subheader("📋 Sample of Cleaned Data")
                st.dataframe(sample_data, use_container_width=True)
            else:
                st.warning("No sample data available for display")

            # Extract fields from cleaned data
            fields = extract_fields_from_cleaned_data(
                session, 
                data_quality_stats['cleaned_table'], 
                data_quality_stats['column_name'], 
                company_name, 
                topic
            )
            
            # Show fields found
            st.subheader("🔍 Fields Analysis")
            if fields:
                st.success(f"✅ Found {len(fields)} fields: {', '.join(fields)}")
                
                # Show preview of SQL field names
                sql_fields = [clean_field_name(field) for field in fields]
                with st.expander("📝 View SQL Column Names"):
                    st.write(f"SQL column names: {', '.join(sql_fields)}")
                
            else:
                st.warning("⚠️ Could not extract field information from source table. The data may not contain recognizable JSON structures.")
                
        else:
            st.error("❌ No valid data found for the selected company and topic. Please check your data or try different filters.")
            fields = []
        
        # Configuration and Transform sections - Always show if we have basic info
        st.markdown("---")
        st.subheader("⚙️ Configuration & Transform")
        
        # Configuration name for saving
        config_name = st.text_input(
            "💾 Configuration name:",
            value=f"{company_name}_{topic}_{source_table}",
            help="Name to save this transformation configuration"
        )
        
        # Create columns for buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Save config button - always available
            if st.button("💾 Save Configuration", type="secondary", key="save_config"):
                if config_name:
                    # Use empty fields if none found
                    fields_to_save = fields if fields else []
                    config_saved = save_transformation_config(session, config_name, source_table, target_table, company_name, topic, fields_to_save)
                    if config_saved:
                        st.success(f"✅ Configuration saved as: {config_name}")
                    else:
                        st.error("Failed to save configuration")
                else:
                    st.error("Please enter a configuration name")
        
        with col2:
            # Transform button - only if we have fields
            if fields and data_quality_stats and data_quality_stats.get('total_records', 0) > 0:
                if st.button("🔄 Transform Data", type="primary", key="transform_data"):
                    if target_table and target_table.strip():
                        with st.spinner("Transforming data..."):
                            success = transform_to_structured_table(
                                session, 
                                data_quality_stats['cleaned_table'], 
                                data_quality_stats['column_name'], 
                                target_table, 
                                fields, 
                                company_name, 
                                topic
                            )
                        
                        if success:
                            st.success(f"✅ Successfully transformed data to table: {target_table}")
                            
                            # Show sample of transformed data
                            try:
                                sample_result = session.sql(f"""
                                    SELECT * FROM {db_name}.SILVER_LAYER.{target_table}
                                    LIMIT 5
                                """).collect()
                                
                                if sample_result:
                                    st.subheader("📋 Sample of Transformed Data")
                                    st.dataframe([row.as_dict() for row in sample_result], use_container_width=True)
                                
                                # Show record count
                                count_result = session.sql(f"""
                                    SELECT COUNT(*) as record_count
                                    FROM {db_name}.SILVER_LAYER.{target_table}
                                """).collect()
                                
                                if count_result:
                                    st.info(f"Total records transformed: {count_result[0]['RECORD_COUNT']}")
                                    
                                # Show table location
                                st.info(f"📍 Table created: `{db_name}.SILVER_LAYER.{target_table}`")
                            
                            except Exception as e:
                                st.warning(f"Could not show sample data: {str(e)}")
                        
                        else:
                            st.error("Failed to transform data. Check the error message above.")
                    else:
                        st.error("Please enter a target table name")
            else:
                st.button("🔄 Transform Data", type="secondary", disabled=True, key="transform_disabled", 
                         help="Transform requires valid data and extracted fields")
        
        with col3:
            # Status indicator
            if fields and data_quality_stats and data_quality_stats.get('total_records', 0) > 0:
                st.success("✅ Ready to transform")
            else:
                st.warning("⚠️ Check data quality above")
    else:
        st.warning("Please select both Company and Topic to proceed.")

# Footer
st.markdown("---")
st.markdown("**Structured Tables Complete** → Your structured data is ready for analysis using the Demo Framework!") 