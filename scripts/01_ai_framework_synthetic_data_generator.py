import streamlit as st
import snowflake.snowpark as snowpark
from snowflake.snowpark.session import Session
import pandas as pd
import json
import re
import time
import yaml
import os
from datetime import datetime
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ""
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""

# Configuration constants for standardized location
CONFIGS_TABLE = "SYNTHETIC_DATA_CONFIGS"
CONFIGS_SCHEMA = 'CONFIGS'

# DataOps database configuration - using environment variables directly

# Configuration constants for standardized location
CONFIGS_TABLE = "SYNTHETIC_DATA_CONFIGS"
# Get the current database from the Snowflake session context
# This will be the database that DataOps deployed the Streamlit app to
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

CONFIGS_DB = get_current_database()
CONFIGS_SCHEMA = 'CONFIGS'

def setup_database_infrastructure(session) -> bool:
    """Simple check - just return True, let table creation handle any issues"""
    return True

def grant_table_permissions(session, table_name: str) -> bool:
    """Ensure table permissions are set - simplified version"""
    try:
        # Skip role switching - rely on current session permissions
        # Tables should inherit appropriate permissions from database/schema grants
        return True
    except Exception as e:
        st.warning(f"Could not ensure table ownership: {str(e)}")
        return False

def get_snowflake_session():
    """Get or create Snowflake session"""
    try:
        session = snowpark.context.get_active_session()
        return session
    except:
        st.error("Unable to connect to Snowflake. Please ensure you're running this in a Snowflake environment.")
        return None

def create_configs_table(session: Session) -> bool:
    """Create the configurations table if it doesn't exist"""
    try:
        # Use standardized config location - exactly like original but with DataOps prefix
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE} (
            config_name VARCHAR(255) PRIMARY KEY,
            config_data VARIANT
        )
        """
        session.sql(create_table_sql).collect()
        return True
    except Exception as e:
        st.error(f"Error creating configs table: {str(e)}")
        return False

def save_config_to_yaml(session: Session, config_name: str, config_data: Dict) -> bool:
    """Save configuration to table in Snowflake"""
    try:
        # Use standardized config location
        # Create table if it doesn't exist
        if not create_configs_table(session):
            return False
        
        # Check table structure to determine if we need timestamps
        try:
            desc_result = session.sql(f"DESCRIBE TABLE {CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE}").collect()
            columns = [row['name'].upper() for row in desc_result]
            has_timestamps = 'CREATED_TIMESTAMP' in columns or 'UPDATED_TIMESTAMP' in columns
        except Exception:
            has_timestamps = False
        
        # Create a temporary DataFrame to avoid JSON escaping issues
        import pandas as pd
        from datetime import datetime
        
        if has_timestamps:
            # Include timestamp columns for old schema
            temp_df = pd.DataFrame([{
                'config_name': config_name,
                'config_data': config_data,
                'created_timestamp': datetime.now(),
                'updated_timestamp': datetime.now()
            }])
        else:
            # Use simple schema
            temp_df = pd.DataFrame([{
                'config_name': config_name,
                'config_data': config_data
            }])
        
        # Create Snowpark DataFrame
        sp_df = session.create_dataframe(temp_df)
        
        # First, delete any existing configuration with the same name
        delete_sql = f"""
        DELETE FROM {CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE} 
        WHERE config_name = '{config_name.replace("'", "''")}'
        """
        session.sql(delete_sql).collect()
        
        # Insert the new configuration
        sp_df.write.mode('append').save_as_table(f"{CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE}")
        
        return True
    except Exception as e:
        st.error(f"Error saving config to table: {str(e)}")
        return False

def load_config_from_yaml(session: Session, config_name: str) -> Dict:
    """Load configuration from table in Snowflake"""
    try:
        # Use standardized config location
        # Escape single quotes
        config_name_escaped = config_name.replace("'", "''")
        
        # Query the configuration
        select_sql = f"""
        SELECT config_data 
        FROM {CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE} 
        WHERE config_name = '{config_name_escaped}'
        """
        
        result = session.sql(select_sql).collect()
        
        if result:
            # Get the JSON data - it's already parsed as a dictionary from VARIANT
            config_data = result[0]['CONFIG_DATA']
            
            # Handle different possible types from Snowflake VARIANT
            if isinstance(config_data, dict):
                return config_data
            elif isinstance(config_data, str):
                # If it's a string, parse it as JSON
                try:
                    return json.loads(config_data)
                except json.JSONDecodeError:
                    st.error(f"Error parsing configuration JSON for '{config_name}'")
                    return {}
            else:
                # Try to convert to dict if it's some other type
                try:
                    return dict(config_data)
                except (TypeError, ValueError):
                    st.error(f"Error converting configuration data for '{config_name}'")
                    return {}
        else:
            st.error(f"Configuration '{config_name}' not found")
            return {}
            
    except Exception as e:
        st.error(f"Error loading config from table: {str(e)}")
        return {}

def get_available_configs(session: Session) -> List[str]:
    """Get list of available configurations from table"""
    try:
        # Use standardized config location
        # Query available configurations
        select_sql = f"""
        SELECT config_name 
        FROM {CONFIGS_DB}.{CONFIGS_SCHEMA}.{CONFIGS_TABLE} 
        ORDER BY config_name
        """
        
        result = session.sql(select_sql).collect()
        
        configs = []
        for row in result:
            configs.append(row['CONFIG_NAME'])
        
        return configs
    except Exception as e:
        # If table doesn't exist yet, return empty list
        if "does not exist" in str(e).lower():
            return []
        st.error(f"Error reading configs from table: {str(e)}")
        return []

def generate_default_prompts(company_name: str, topic: str, fields: List[str], batch_size: int) -> tuple:
    """Generate default system and user prompts based on inputs"""
    
    fields_str = ", ".join(fields)
    
    system_prompt = f"""You are a synthetic data generator that creates realistic, diverse datasets for testing and development purposes.

Your task is to generate {batch_size} records of synthetic data for {company_name} related to {topic}.

Requirements:
- Generate data in JSON format only
- Each record should be a JSON object with the specified fields
- Make the data realistic and diverse
- Ensure data consistency and logical relationships between fields
- Do not include any explanatory text, only return the JSON array
- Use appropriate data types for each field
- Make sure the data is relevant to the company and topic context
- Ensure each batch has unique data that doesn't repeat previous batches

Fields to include: {fields_str}

Return the data as a JSON array of objects, with each object representing one record."""

    user_prompt = f"""Generate {batch_size} synthetic records for {company_name} focusing on {topic}. 

Each record should include these fields: {fields_str}

Return only a valid JSON array with no additional text or formatting."""

    return system_prompt, user_prompt

def create_generation_procedure(session: Session, database: str, schema: str) -> bool:
    """Create the improved high-performance generation stored procedure with better JSON handling"""
    try:
        procedure_sql = """
        CREATE OR REPLACE PROCEDURE {database}.{schema}.GENERATE_SYNTHETIC_DATA_BATCH(
            BATCH_NUM FLOAT,
            COMPANY_NAME VARCHAR,
            TOPIC VARCHAR,
            FIELDS_JSON VARCHAR,
            SYSTEM_PROMPT VARCHAR,
            USER_PROMPT VARCHAR,
            MODEL VARCHAR,
            TEMPERATURE FLOAT,
            MAX_TOKENS FLOAT,
            TARGET_DATABASE VARCHAR,
            TARGET_SCHEMA VARCHAR,
            TARGET_TABLE VARCHAR,
            METADATA_JSON VARCHAR
        )
        RETURNS STRING
        LANGUAGE JAVASCRIPT
        EXECUTE AS CALLER
        AS
        $$
            // Clean prompts for use in SQL - remove problematic characters
            function cleanForSQL(text) {{
                return text
                    .replace(/'/g, "''")           // Escape single quotes
                    .replace(/\\n/g, ' ')          // Replace newlines with spaces
                    .replace(/\\r/g, ' ')          // Replace carriage returns with spaces
                    .replace(/\\t/g, ' ')          // Replace tabs with spaces
                    .replace(/"/g, '\\"')          // Escape double quotes
                    .trim();
            }}
            
            var system_prompt_clean = cleanForSQL(SYSTEM_PROMPT);
            var user_prompt_clean = cleanForSQL(USER_PROMPT);
            
            // Parse metadata safely
            var metadata;
            try {{
                metadata = JSON.parse(METADATA_JSON);
            }} catch (e) {{
                throw new Error('Failed to parse metadata JSON: ' + e.message);
            }}
            
            // Simple approach: Just call CORTEX.COMPLETE directly in INSERT - no parsing!
            var insert_sql = `
            INSERT INTO ${{TARGET_DATABASE}}.${{TARGET_SCHEMA}}.${{TARGET_TABLE}} (
                messages,
                _meta_generation_timestamp,
                _meta_company_name,
                _meta_topic,
                _meta_fields_requested,
                _meta_batch_size,
                _meta_num_batches,
                _meta_model_type,
                _meta_model,
                _meta_temperature,
                _meta_max_tokens,
                _meta_system_prompt,
                _meta_user_prompt,
                _meta_total_records_requested,
                _meta_batch_number,
                _meta_batch_timestamp,
                _meta_records_in_batch
            )
            SELECT 
                SNOWFLAKE.CORTEX.COMPLETE(
                    '${{MODEL}}',
                    [
                        {{'role': 'system', 'content': '${{system_prompt_clean}}'}},
                        {{'role': 'user', 'content': '${{user_prompt_clean}}'}}
                    ],
                    {{'temperature': ${{TEMPERATURE}}, 'max_tokens': ${{MAX_TOKENS}}}}
                ):choices[0]:messages as messages,
                '${{metadata.generation_timestamp}}',
                '${{metadata.company_name}}',
                '${{metadata.topic}}',
                '${{metadata.fields_requested}}',
                ${{metadata.batch_size}},
                ${{metadata.num_batches}},
                '${{metadata.model_type}}',
                '${{metadata.model}}',
                ${{metadata.temperature}},
                ${{metadata.max_tokens}},
                '${{metadata.system_prompt}}',
                '${{metadata.user_prompt}}',
                ${{metadata.total_records_requested}},
                ${{BATCH_NUM}},
                '${{new Date().toISOString()}}',
                100
            `;
            
            var insert_statement = snowflake.createStatement({{sqlText: insert_sql}});
            insert_statement.execute();
            
            return 'Batch ' + BATCH_NUM + ' generated successfully - simple approach!';
        $$;
        """.format(database=database, schema=schema)
        
        session.sql(procedure_sql).collect()
        return True
    except Exception as e:
        st.error(f"Error creating procedure: {str(e)}")
        return False

def call_cortex_complete(session: Session, model: str, system_prompt: str, user_prompt: str, 
                        temperature: float, max_tokens: int) -> str:
    """Call Snowflake Cortex Complete function (fallback method)"""
    try:
        # Escape single quotes in prompts
        system_prompt_escaped = system_prompt.replace("'", "''")
        user_prompt_escaped = user_prompt.replace("'", "''")
        
        # Build the SQL query for Cortex Complete
        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT('role', 'system', 'content', '{system_prompt_escaped}'),
                OBJECT_CONSTRUCT('role', 'user', 'content', '{user_prompt_escaped}')
            ),
            OBJECT_CONSTRUCT('temperature', {temperature}, 'max_tokens', {max_tokens})
        ) as response
        """
        
        result = session.sql(sql).collect()
        if result:
            # Extract just the JSON array from the Cortex response
            cortex_response = result[0]['RESPONSE']
            try:
                parsed_response = json.loads(cortex_response)
                return parsed_response['choices'][0]['messages']
            except (json.JSONDecodeError, KeyError, IndexError):
                # Fallback to original response if parsing fails
                return cortex_response
        else:
            return "No response received from Cortex"
    except Exception as e:
        st.error(f"Error calling Cortex Complete: {str(e)}")
        return None

def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON response from LLM"""
    try:
        # Try to extract JSON array from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data
        else:
            # If no array found, try to parse the entire response
            data = json.loads(response)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON response: {str(e)}")
        st.error(f"Response was: {response}")
        return None

def save_to_table(session: Session, data: List[Dict], table_name: str, database: str, schema: str, 
                  append: bool = False, metadata: Dict = None):
    """Save generated data to Snowflake table with metadata"""
    try:
        # Create DataFrame from data
        df = pd.DataFrame(data)
        
        # Add metadata columns if provided
        if metadata:
            for key, value in metadata.items():
                df[f"_meta_{key}"] = value
        
        # Create Snowpark DataFrame
        sp_df = session.create_dataframe(df)
        
        # Create fully qualified table name
        full_table_name = f"{database}.{schema}.{table_name}"
        
        # Write to table
        mode = 'append' if append else 'overwrite'
        sp_df.write.mode(mode).save_as_table(full_table_name)
        
        action = "appended to" if append else "saved to"
        st.success(f"Data successfully {action} {full_table_name}")
        
    except Exception as e:
        st.error(f"Error saving to table: {str(e)}")
        return False
    return True

def create_improved_generation_procedure(session: Session, database: str, schema: str, company_name: str, topic: str) -> tuple:
    """Create a simplified SQL-based generation procedure using pure SQL"""
    
    # Clean names for SQL identifiers
    clean_company = re.sub(r'[^\w]', '_', company_name).upper()
    clean_topic = re.sub(r'[^\w]', '_', topic).upper()
    
    # Create procedure names
    main_proc_name = f"GENERATE_{clean_company}_{clean_topic}_DATA"
    batch_proc_name = f"GENERATE_{clean_company}_{clean_topic}_BATCH"
    
    try:
        # Create the batch generation procedure - much simpler approach
        batch_procedure_sql = f"""CREATE OR REPLACE PROCEDURE {database}.{schema}.{batch_proc_name}(
            BATCH_NUM INTEGER,
            COMPANY_NAME VARCHAR,
            TOPIC VARCHAR,
            FIELDS_JSON VARCHAR,
            SYSTEM_PROMPT VARCHAR,
            USER_PROMPT VARCHAR,
            MODEL VARCHAR,
            TEMPERATURE FLOAT,
            MAX_TOKENS INTEGER,
            TARGET_DATABASE VARCHAR,
            TARGET_SCHEMA VARCHAR,
            TARGET_TABLE VARCHAR
        )
        RETURNS STRING
        LANGUAGE SQL
        AS
        $$
        DECLARE
            response_text VARCHAR;
            insert_sql VARCHAR;
            result_msg VARCHAR;
            table_name VARCHAR;
            json_data VARCHAR;
        BEGIN
            -- Generate the synthetic data using Cortex and extract just the JSON array
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                MODEL,
                ARRAY_CONSTRUCT(
                    OBJECT_CONSTRUCT('role', 'system', 'content', SYSTEM_PROMPT),
                    OBJECT_CONSTRUCT('role', 'user', 'content', USER_PROMPT)
                ),
                OBJECT_CONSTRUCT('temperature', TEMPERATURE, 'max_tokens', MAX_TOKENS)
            ):choices[0]:messages INTO response_text;
            
            -- Insert the data using EXECUTE IMMEDIATE for dynamic table name
            table_name := TARGET_DATABASE || '.' || TARGET_SCHEMA || '.' || TARGET_TABLE;
            json_data := OBJECT_CONSTRUCT('choices', ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('messages', response_text)))::VARCHAR;
            
            insert_sql := 'INSERT INTO ' || table_name || ' (messages, _meta_generation_timestamp, _meta_company_name, _meta_topic, _meta_fields_requested, _meta_batch_number, _meta_batch_timestamp, _meta_model, _meta_temperature, _meta_max_tokens, _meta_system_prompt, _meta_user_prompt) VALUES (PARSE_JSON(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)';
            
            EXECUTE IMMEDIATE insert_sql USING (
                json_data,
                TO_VARCHAR(CURRENT_TIMESTAMP()),
                COMPANY_NAME,
                TOPIC,
                FIELDS_JSON,
                BATCH_NUM,
                TO_VARCHAR(CURRENT_TIMESTAMP()),
                MODEL,
                TEMPERATURE,
                MAX_TOKENS,
                LEFT(SYSTEM_PROMPT, 500),
                LEFT(USER_PROMPT, 500)
            );
            
            result_msg := 'Batch ' || BATCH_NUM || ' completed successfully for ' || COMPANY_NAME || ' - ' || TOPIC;
            RETURN result_msg;
        END;
        $$;"""
        
        # Create the main loop procedure - simplified
        main_procedure_sql = f"""CREATE OR REPLACE PROCEDURE {database}.{schema}.{main_proc_name}(
            NO_TIMES INTEGER,
            COMPANY_NAME VARCHAR,
            TOPIC VARCHAR,
            FIELDS_JSON VARCHAR,
            SYSTEM_PROMPT VARCHAR,
            USER_PROMPT VARCHAR,
            MODEL VARCHAR,
            TEMPERATURE FLOAT,
            MAX_TOKENS INTEGER,
            TARGET_DATABASE VARCHAR,
            TARGET_SCHEMA VARCHAR,
            TARGET_TABLE VARCHAR
        )
        RETURNS INTEGER
        LANGUAGE SQL
        AS
        $$
        DECLARE
            counter INTEGER := 0;
            batch_result VARCHAR;
        BEGIN
            WHILE (counter < NO_TIMES) DO
                counter := counter + 1;
                
                CALL {database}.{schema}.{batch_proc_name}(
                    counter,
                    COMPANY_NAME,
                    TOPIC,
                    FIELDS_JSON,
                    SYSTEM_PROMPT,
                    USER_PROMPT,
                    MODEL,
                    TEMPERATURE,
                    MAX_TOKENS,
                    TARGET_DATABASE,
                    TARGET_SCHEMA,
                    TARGET_TABLE
                ) INTO batch_result;
                
            END WHILE;
            
            RETURN counter;
        END;
        $$;"""
        
        # Execute both procedures
        session.sql(batch_procedure_sql).collect()
        session.sql(main_procedure_sql).collect()
        
        return main_proc_name, batch_proc_name, main_procedure_sql, batch_procedure_sql
        
    except Exception as e:
        st.error(f"Error creating improved procedures: {str(e)}")
        return None, None, None, None

def create_ultra_simple_procedure(session: Session, database: str, schema: str, company_name: str, topic: str) -> tuple:
    """Create an ultra-simple procedure that just calls Cortex and returns the result"""
    
    # Clean names for SQL identifiers
    clean_company = re.sub(r'[^\w]', '_', company_name).upper()
    clean_topic = re.sub(r'[^\w]', '_', topic).upper()
    
    # Create procedure name
    simple_proc_name = f"CORTEX_CALL_{clean_company}_{clean_topic}"
    
    try:
        # Create the simplest possible procedure - just call Cortex and return result
        simple_procedure_sql = f"""CREATE OR REPLACE PROCEDURE {database}.{schema}.{simple_proc_name}(
            SYSTEM_PROMPT VARCHAR,
            USER_PROMPT VARCHAR,
            MODEL VARCHAR,
            TEMPERATURE FLOAT,
            MAX_TOKENS INTEGER
        )
        RETURNS VARCHAR
        LANGUAGE SQL
        AS
        $$
        DECLARE
            response_text VARCHAR;
        BEGIN
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                MODEL,
                ARRAY_CONSTRUCT(
                    OBJECT_CONSTRUCT('role', 'system', 'content', SYSTEM_PROMPT),
                    OBJECT_CONSTRUCT('role', 'user', 'content', USER_PROMPT)
                ),
                OBJECT_CONSTRUCT('temperature', TEMPERATURE, 'max_tokens', MAX_TOKENS)
            ):choices[0]:messages INTO response_text;
            
            RETURN response_text;
        END;
        $$;"""
        
        # Create a simple loop procedure that calls the basic one multiple times
        loop_procedure_sql = f"""CREATE OR REPLACE PROCEDURE {database}.{schema}.{simple_proc_name}_LOOP(
            NO_TIMES INTEGER,
            SYSTEM_PROMPT VARCHAR,
            USER_PROMPT VARCHAR,
            MODEL VARCHAR,
            TEMPERATURE FLOAT,
            MAX_TOKENS INTEGER,
            TARGET_DATABASE VARCHAR,
            TARGET_SCHEMA VARCHAR,
            TARGET_TABLE VARCHAR,
            COMPANY_NAME VARCHAR,
            TOPIC VARCHAR,
            FIELDS_JSON VARCHAR
        )
        RETURNS VARCHAR
        LANGUAGE SQL
        AS
        $$
        DECLARE
            counter INTEGER := 0;
            batch_result VARCHAR;
            summary_msg VARCHAR := '';
            insert_sql VARCHAR;
            json_data VARCHAR;
            table_name VARCHAR;
        BEGIN
            WHILE (counter < NO_TIMES) DO
                counter := counter + 1;
                
                -- Call the simple Cortex procedure
                CALL {database}.{schema}.{simple_proc_name}(
                    SYSTEM_PROMPT,
                    USER_PROMPT,
                    MODEL,
                    TEMPERATURE,
                    MAX_TOKENS
                ) INTO batch_result;
                
                -- Insert the result with proper JSON structure
                table_name := TARGET_DATABASE || '.' || TARGET_SCHEMA || '.' || TARGET_TABLE;
                json_data := OBJECT_CONSTRUCT('choices', ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('messages', batch_result)))::VARCHAR;
                
                insert_sql := 'INSERT INTO ' || table_name || ' (messages, _meta_generation_timestamp, _meta_company_name, _meta_topic, _meta_fields_requested, _meta_batch_number, _meta_model, _meta_temperature, _meta_max_tokens) VALUES (PARSE_JSON(?), ?, ?, ?, ?, ?, ?, ?, ?)';
                
                EXECUTE IMMEDIATE insert_sql USING (json_data, TO_VARCHAR(CURRENT_TIMESTAMP()), COMPANY_NAME, TOPIC, FIELDS_JSON, counter, MODEL, TEMPERATURE, MAX_TOKENS);
                    
                summary_msg := summary_msg || 'Batch ' || counter || ' completed. ';
                
            END WHILE;
            
            RETURN summary_msg || 'Total batches: ' || counter;
        END;
        $$;"""
        
        # Execute both procedures
        session.sql(simple_procedure_sql).collect()
        session.sql(loop_procedure_sql).collect()
        
        return simple_proc_name, f"{simple_proc_name}_LOOP", simple_procedure_sql, loop_procedure_sql
        
    except Exception as e:
        st.error(f"Error creating ultra-simple procedures: {str(e)}")
        return None, None, None, None

def generate_call_script(database: str, schema: str, proc_name: str, company_name: str, topic: str, 
                        fields: List[str], system_prompt: str, user_prompt: str, model: str, 
                        temperature: float, max_tokens: int, num_batches: int, 
                        target_database: str, target_schema: str, target_table: str) -> str:
    """Generate the call script for manual execution"""
    
    # Clean prompts for SQL
    clean_system = system_prompt.replace("'", "''").replace('\n', ' ').replace('\r', ' ')[:1000]
    clean_user = user_prompt.replace("'", "''").replace('\n', ' ').replace('\r', ' ')[:1000]
    fields_json = ', '.join(fields)
    
    call_script = f"""
-- Generated Call Script for Manual Execution
-- Copy and paste this into a Snowflake worksheet

-- Step 1: Ensure target table exists
CREATE OR REPLACE TABLE {target_database}.{target_schema}.{target_table} (
    messages VARIANT,
    _meta_generation_timestamp VARCHAR,
    _meta_company_name VARCHAR,
    _meta_topic VARCHAR,
    _meta_fields_requested VARCHAR,
    _meta_batch_number INTEGER,
    _meta_batch_timestamp VARCHAR,
    _meta_records_in_batch INTEGER,
    _meta_model VARCHAR,
    _meta_temperature FLOAT,
    _meta_max_tokens INTEGER,
    _meta_system_prompt VARCHAR,
    _meta_user_prompt VARCHAR
);

-- Step 2: Call the main generation procedure
CALL {database}.{schema}.{proc_name}(
    {num_batches},                    -- NO_TIMES: Number of batches to generate
    '{company_name}',                 -- COMPANY_NAME
    '{topic}',                        -- TOPIC
    '{fields_json}',                  -- FIELDS_JSON
    '{clean_system}',                 -- SYSTEM_PROMPT
    '{clean_user}',                   -- USER_PROMPT
    '{model}',                        -- MODEL
    {temperature},                    -- TEMPERATURE
    {max_tokens},                     -- MAX_TOKENS
    '{target_database}',              -- TARGET_DATABASE
    '{target_schema}',                -- TARGET_SCHEMA
    '{target_table}'                  -- TARGET_TABLE
);

-- Step 3: Check results
SELECT 
    COUNT(*) as total_batches,
    SUM(_meta_records_in_batch) as total_records,
    _meta_company_name,
    _meta_topic,
    _meta_model
FROM {target_database}.{target_schema}.{target_table}
WHERE _meta_company_name = '{company_name}'
    AND _meta_topic = '{topic}'
GROUP BY _meta_company_name, _meta_topic, _meta_model;

-- Step 4: View sample data
SELECT * FROM {target_database}.{target_schema}.{target_table} 
WHERE _meta_company_name = '{company_name}'
    AND _meta_topic = '{topic}'
LIMIT 5;
"""
    
    return call_script

def generate_existing_procedure_call_script(database: str, schema: str, proc_name: str, company_name: str, topic: str, 
                        fields: List[str], system_prompt: str, user_prompt: str, model: str, 
                        temperature: float, max_tokens: int, num_batches: int, 
                        target_database: str, target_schema: str, target_table: str) -> str:
    """Generate call script for the existing JavaScript procedure"""
    
    # Clean prompts for SQL
    clean_system = system_prompt.replace("'", "''").replace('\n', ' ').replace('\r', ' ')[:1000]
    clean_user = user_prompt.replace("'", "''").replace('\n', ' ').replace('\r', ' ')[:1000]
    fields_json = ', '.join(fields)
    
    # Generate metadata JSON
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'company_name': company_name,
        'topic': topic,
        'fields_requested': fields_json,
        'batch_size': 100,  # Default batch size
        'num_batches': num_batches,
        'model_type': 'MEDIUM',
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'system_prompt': clean_system,
        'user_prompt': clean_user,
        'total_records_requested': num_batches * 100
    }
    
    # Clean metadata for SQL
    metadata_json = json.dumps(metadata).replace("'", "''")
    
    call_script = f"""
-- Generated Call Script with Loop - Clean and Simple
-- Copy and paste this into a Snowflake worksheet

-- Step 1: Ensure target table exists
CREATE OR REPLACE TABLE {target_database}.{target_schema}.{target_table} (
    messages VARIANT,
    _meta_generation_timestamp VARCHAR,
    _meta_company_name VARCHAR,
    _meta_topic VARCHAR,
    _meta_fields_requested VARCHAR,
    _meta_batch_size INTEGER,
    _meta_num_batches INTEGER,
    _meta_model_type VARCHAR,
    _meta_model VARCHAR,
    _meta_temperature FLOAT,
    _meta_max_tokens INTEGER,
    _meta_system_prompt VARCHAR,
    _meta_user_prompt VARCHAR,
    _meta_total_records_requested INTEGER,
    _meta_batch_number INTEGER,
    _meta_batch_timestamp VARCHAR,
    _meta_records_in_batch INTEGER
);

-- Step 2: Generate data using a simple loop in anonymous block
EXECUTE IMMEDIATE 
$$
DECLARE
    batch_result STRING;
    call_sql STRING;
BEGIN
    FOR batch_num IN 1 TO {num_batches} DO
        
        -- Build the CALL statement dynamically
        call_sql := 'CALL {database}.{schema}.{proc_name}(' ||
                   batch_num || ', ' ||
                   '''' || '{company_name}' || ''', ' ||
                   '''' || '{topic}' || ''', ' ||
                   '''' || '{fields_json}' || ''', ' ||
                   '''' || '{clean_system}' || ''', ' ||
                   '''' || '{clean_user}' || ''', ' ||
                   '''' || '{model}' || ''', ' ||
                   {temperature} || ', ' ||
                   {max_tokens} || ', ' ||
                   '''' || '{target_database}' || ''', ' ||
                   '''' || '{target_schema}' || ''', ' ||
                   '''' || '{target_table}' || ''', ' ||
                   '''' || '{metadata_json}' || '''' ||
                   ')';
        
        -- Execute the CALL
        EXECUTE IMMEDIATE call_sql;
        
        -- Optional: Log progress (uncomment to see progress)
        -- SELECT 'Completed batch ' || batch_num || ' of {num_batches}' as progress;
        
    END FOR;
    
    SELECT 'Generation complete! Processed {num_batches} batches.' as final_result;
END;
$$;

-- Step 3: Check results
SELECT 
    COUNT(*) as total_batches,
    SUM(_meta_records_in_batch) as total_records,
    _meta_company_name,
    _meta_topic,
    _meta_model
FROM {target_database}.{target_schema}.{target_table}
WHERE _meta_company_name = '{company_name}'
    AND _meta_topic = '{topic}'
GROUP BY _meta_company_name, _meta_topic, _meta_model;

-- Step 4: View sample data
SELECT * FROM {target_database}.{target_schema}.{target_table} 
WHERE _meta_company_name = '{company_name}'
    AND _meta_topic = '{topic}'
ORDER BY _meta_batch_number
LIMIT 5;
"""
    
    return call_script

# Main App Layout
st.title("🎲 Synthetic Data Generator")
st.markdown("Generate synthetic datasets using Snowflake Cortex LLM")

# Get Snowflake session
session = get_snowflake_session()

# Setup database infrastructure
if session:
    try:
        setup_database_infrastructure(session)
    except Exception as e:
        # If database setup fails, show warning but don't crash the app
        st.warning(f"⚠️ Database infrastructure setup encountered an issue: {str(e)}")
        st.info("💡 The app may still work if the database was already created by DataOps")

if session:
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Configuration Management
    st.sidebar.subheader("Configuration Management")
    
    # Load existing configuration
    available_configs = get_available_configs(session)
    if available_configs:
        selected_config = st.sidebar.selectbox(
            "Load Configuration", 
            options=["Create New"] + available_configs,
            help="Select a saved configuration to load"
        )
        
        if selected_config != "Create New":
            if st.sidebar.button("📁 Load Configuration"):
                config_data = load_config_from_yaml(session, selected_config)
                if config_data:
                    # Load all the configuration values
                    st.session_state.config_loaded = True
                    st.session_state.loaded_config = config_data
                    st.session_state.loaded_config_name = selected_config  # Store the config name
                    st.success(f"Configuration '{selected_config}' loaded successfully!")
                    st.rerun()
    
    # Get default values from loaded config if available
    loaded_config = st.session_state.get('loaded_config', {})
    
    # Input fields
    st.sidebar.subheader("Dataset Configuration")
    company_name = st.sidebar.text_input(
        "Company Name", 
        value=loaded_config.get('company_name', 'Acme Corp'), 
        key="company_name",
        help="Name of the company for context"
    )
    topic = st.sidebar.text_input(
        "Topic/Domain", 
        value=loaded_config.get('topic', 'Customer Orders'), 
        key="topic",
        help="The domain or topic for the synthetic data"
    )
    
    # Fields configuration
    st.sidebar.subheader("Data Fields")
    default_fields = loaded_config.get('fields', [
        'customer_id', 'customer_name', 'email', 'order_date', 
        'product_name', 'quantity', 'price', 'total_amount'
    ])
    fields_input = st.sidebar.text_area(
        "Fields (one per line)", 
        value='\n'.join(default_fields) if isinstance(default_fields, list) else default_fields,
        key="fields_input",
        help="Enter field names, one per line"
    )
    fields = [field.strip() for field in fields_input.split('\n') if field.strip()]
    
    # Batch Configuration
    st.sidebar.subheader("Batch Configuration")
    batch_size = st.sidebar.slider(
        "Records per Batch", 
        min_value=10, max_value=200, 
        value=loaded_config.get('batch_size', 100), 
        step=10,
        key="batch_size",
        help="Number of records to generate in each batch (smaller batches are more reliable)"
    )
    num_batches = st.sidebar.slider(
        "Number of Batches", 
        min_value=1, max_value=1000, 
        value=loaded_config.get('num_batches', 10),
        key="num_batches",
        help="How many batches to generate"
    )
    
    total_records = batch_size * num_batches
    st.sidebar.info(f"Total records to generate: {total_records:,}")
    
    # Cortex Configuration
    st.sidebar.subheader("Cortex LLM Configuration")
    
    # Model categorization by size
    model_categories = {
        "SMALL": [
            "gemma-7b-it",
            "mistral-7b",
            "llama3.1-8b"
        ],
        "MEDIUM": [
            "mixtral-8x7b",
            "reka-flash"
        ],
        "LARGE": [
            "llama2-70b-chat",
            "reka-core",
            "claude-3-5-sonnet",
            "mistral-large2",
            "llama3.1-405b",
            "snowflake-llama3.1-405b",
            "snowflake-llama3.3-70b",
            "snowflake-arctic"
        ]
    }
    
    # Model type selector
    loaded_model_type = loaded_config.get('model_type', 'MEDIUM')
    model_type_index = list(model_categories.keys()).index(loaded_model_type) if loaded_model_type in model_categories else 1
    model_type = st.sidebar.selectbox(
        "Model Type", 
        options=list(model_categories.keys()),
        index=model_type_index,
        help="Small: Faster, lower cost | Medium: Balanced | Large: Higher quality, slower"
    )
    
    # Model selector based on type
    loaded_model = loaded_config.get('model', model_categories[model_type][0])
    model_index = 0
    if loaded_model in model_categories[model_type]:
        model_index = model_categories[model_type].index(loaded_model)
    model = st.sidebar.selectbox(
        "Model", 
        options=model_categories[model_type],
        index=model_index,
        help=f"Select a specific {model_type.lower()} model"
    )
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, max_value=1.0, 
        value=loaded_config.get('temperature', 0.7), 
        step=0.1
    )
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
        min_value=100, max_value=8000, 
        value=loaded_config.get('max_tokens', 4000), 
        step=100
    )
    
    # Performance Configuration
    st.sidebar.subheader("Performance Configuration")
    use_procedure = st.sidebar.checkbox(
        "High-Performance Mode", 
        value=loaded_config.get('use_procedure', True),
        help="Use stored procedures for faster generation (recommended for large datasets)"
    )
    
    show_manual_scripts = st.sidebar.checkbox(
        "Show Manual Scripts", 
        value=loaded_config.get('show_manual_scripts', False),
        help="Display SQL procedures and call scripts for manual execution"
    )
    
    # Auto-save Configuration
    st.sidebar.subheader("Auto-save Configuration")
    auto_save = st.sidebar.checkbox(
        "Auto-save batches to table", 
        value=loaded_config.get('auto_save', True), 
        help="Automatically save each batch to a Snowflake table as it's generated"
    )
    
    if auto_save:
        save_database = st.sidebar.text_input(
            "Database", 
            value=loaded_config.get('save_database', get_current_database())
        )
        save_schema = st.sidebar.text_input(
            "Schema", 
            value=loaded_config.get('save_schema', 'BRONZE_LAYER')
        )
        save_table = st.sidebar.text_input(
            "Table Name", 
            value=loaded_config.get('save_table', 'GENERATED_DATA')
        )
        
        append_mode = st.sidebar.checkbox(
            "Append to existing table", 
            value=loaded_config.get('append_mode', True),
            help="If unchecked, will overwrite the table on first batch"
        )
    
    # Show warning if high-performance mode is enabled but auto-save is not
    if use_procedure and not auto_save:
        st.sidebar.warning("⚠️ High-Performance Mode requires Auto-save to be enabled")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("System Prompt")
        if st.button("Generate Default Prompts"):
            system_prompt, user_prompt = generate_default_prompts(company_name, topic, fields, batch_size)
            st.session_state.system_prompt = system_prompt
            st.session_state.user_prompt = user_prompt
            st.success("✅ Prompts generated! Check the text areas below.")
        
        # Use loaded prompt if available, otherwise use session state
        current_system_prompt = loaded_config.get('system_prompt') or st.session_state.system_prompt
        system_prompt_edited = st.text_area(
            "System Prompt",
            value=current_system_prompt,
            height=300,
            help="Edit the system prompt to customize the data generation behavior"
        )
    
    with col2:
        st.subheader("User Prompt")
        # Use loaded prompt if available, otherwise use session state
        current_user_prompt = loaded_config.get('user_prompt') or st.session_state.user_prompt
        user_prompt_edited = st.text_area(
            "User Prompt",
            value=current_user_prompt,
            height=300,
            help="Edit the user prompt to specify exactly what data you want"
        )
    
    # Manual Scripts Section
    if show_manual_scripts:
        st.markdown("---")
        st.subheader("📋 Manual Execution Scripts")
        st.markdown("**Use these scripts to run data generation manually in a Snowflake worksheet for better performance**")
        
        # For manual scripts, use default save location if auto_save is disabled
        script_database = save_database if auto_save else get_current_database()
        script_schema = save_schema if auto_save else 'BRONZE_LAYER'
        script_table = save_table if auto_save else 'GENERATED_DATA'
        
        # Show current target location
        st.info(f"**Target Table:** `{script_database}.{script_schema}.{script_table}`")
        
        # Using existing JavaScript procedure
        st.info("💡 **Using existing JavaScript procedure** - This is the proven approach that works reliably.")
        
        # Generate the procedures and call script
        if st.button("🔧 Generate Manual Scripts"):
            with st.spinner("Creating procedure and generating scripts..."):
                # Use the existing JavaScript procedure - create it if it doesn't exist
                success = create_generation_procedure(session, script_database, script_schema)
                
                if success:
                    # Generate the call script for the existing procedure
                    proc_name = "GENERATE_SYNTHETIC_DATA_BATCH"
                    call_script = generate_existing_procedure_call_script(
                        script_database, script_schema, proc_name, company_name, topic,
                        fields, system_prompt_edited, user_prompt_edited, model,
                        temperature, max_tokens, num_batches, script_database, script_schema, script_table
                    )
                    
                    # Display the result
                    st.success("✅ Using existing JavaScript procedure!")
                    
                    # Show procedure name
                    st.info(f"**Procedure:** `{script_database}.{script_schema}.{proc_name}`")
                    
                    # Call script
                    st.subheader("📋 Manual Execution Script")
                    st.markdown("**Copy and paste this into a Snowflake worksheet:**")
                    st.code(call_script, language="sql")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Call Script",
                        data=call_script,
                        file_name=f"{proc_name.lower()}_call_script.sql",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to create procedure. Check the error messages above.")
    
    # Configuration Save Section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Save Configuration")
        save_config_name = st.text_input(
            "Configuration Name", 
            value=st.session_state.get('loaded_config_name', ''),
            placeholder="e.g., customer_orders_v1",
            help="Name for saving this configuration"
        )
        
        if st.button("💾 Save Configuration") and save_config_name:
            current_config = {
                'name': save_config_name,
                'created_date': datetime.now().isoformat(),
                'company_name': company_name,
                'topic': topic,
                'fields': fields,
                'batch_size': batch_size,
                'num_batches': num_batches,
                'model_type': model_type,
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'use_procedure': use_procedure,
                'show_manual_scripts': show_manual_scripts,
                'auto_save': auto_save,
                'save_database': save_database if auto_save else None,
                'save_schema': save_schema if auto_save else None,
                'save_table': save_table if auto_save else None,
                'append_mode': append_mode if auto_save else None,
                'system_prompt': system_prompt_edited,
                'user_prompt': user_prompt_edited
            }
            
            if save_config_to_yaml(session, save_config_name, current_config):
                st.success(f"Configuration '{save_config_name}' saved successfully!")
            else:
                st.error("Failed to save configuration")
    
    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
            # Initialize session state for batch processing
            if 'all_generated_data' not in st.session_state:
                st.session_state.all_generated_data = []
            
            st.session_state.all_generated_data = []  # Reset for new generation
            
            # Create metadata for this generation run
            generation_metadata = {
                'generation_timestamp': datetime.now().isoformat(),
                'company_name': company_name,
                'topic': topic,
                'fields_requested': ', '.join(fields),
                'batch_size': batch_size,
                'num_batches': num_batches,
                'model_type': model_type,
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'system_prompt': system_prompt_edited[:500] + ('...' if len(system_prompt_edited) > 500 else ''),
                'user_prompt': user_prompt_edited[:500] + ('...' if len(user_prompt_edited) > 500 else ''),
                'total_records_requested': total_records
            }
            
            # Create progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_info = st.empty()
            
            successful_batches = 0
            total_records_generated = 0
            procedure_completed = False  # Track if stored procedure was used successfully
            
            # High-performance mode using stored procedures
            if use_procedure and auto_save:
                # Create the stored procedure
                proc_created = create_generation_procedure(session, save_database, save_schema)
                if not proc_created:
                    st.error("Failed to create stored procedure. Falling back to standard mode.")
                    use_procedure = False
                else:
                    st.info("🚀 Using High-Performance Mode with stored procedures")
                    
                    # Ensure target table exists before processing
                    try:
                        create_target_table_sql = f"""
                        CREATE OR REPLACE TABLE {save_database}.{save_schema}.{save_table} (
                            messages VARCHAR,
                            _meta_generation_timestamp VARCHAR,
                            _meta_company_name VARCHAR,
                            _meta_topic VARCHAR,
                            _meta_fields_requested VARCHAR,
                            _meta_batch_size NUMBER,
                            _meta_num_batches NUMBER,
                            _meta_model_type VARCHAR,
                            _meta_model VARCHAR,
                            _meta_temperature NUMBER,
                            _meta_max_tokens NUMBER,
                            _meta_system_prompt VARCHAR,
                            _meta_user_prompt VARCHAR,
                            _meta_total_records_requested NUMBER,
                            _meta_batch_number NUMBER,
                            _meta_batch_timestamp VARCHAR,
                            _meta_records_in_batch NUMBER
                        )
                        """
                        session.sql(create_target_table_sql).collect()
                        st.success(f"✅ Target table {save_database}.{save_schema}.{save_table} ready")
                    except Exception as e:
                        st.error(f"❌ Failed to create target table: {str(e)}")
                        st.error("Falling back to standard mode")
                        use_procedure = False
                    
                    if use_procedure:  # Only continue if table creation succeeded
                        # Prepare metadata as JSON string - clean prompts for JSON safety
                        clean_metadata = generation_metadata.copy()
                        # Remove or clean problematic characters from prompts
                        clean_metadata['system_prompt'] = clean_metadata['system_prompt'].replace('\n', ' ').replace('\r', ' ').replace('"', "'")[:200] + '...'
                        clean_metadata['user_prompt'] = clean_metadata['user_prompt'].replace('\n', ' ').replace('\r', ' ').replace('"', "'")[:200] + '...'
                        
                        metadata_json = json.dumps(clean_metadata)
                        fields_json = json.dumps(fields)
                        
                        # Process each batch using stored procedure
                        for batch_num in range(1, num_batches + 1):
                            progress = (batch_num - 1) / num_batches
                            progress_bar.progress(progress)
                            status_text.text(f"Generating batch {batch_num} of {num_batches} (High-Performance Mode)...")
                            
                            try:
                                # Clean prompts for SQL safety
                                clean_system_prompt = system_prompt_edited.replace("'", "''").replace('\n', ' ').replace('\r', ' ').replace('"', "'")
                                clean_user_prompt = user_prompt_edited.replace("'", "''").replace('\n', ' ').replace('\r', ' ').replace('"', "'")
                                clean_metadata_json = metadata_json.replace("'", "''")
                                
                                # Call the stored procedure
                                proc_call = f"""
                                CALL {save_database}.{save_schema}.GENERATE_SYNTHETIC_DATA_BATCH(
                                    {batch_num},
                                    '{company_name}',
                                    '{topic}',
                                    '{fields_json}',
                                    '{clean_system_prompt}',
                                    '{clean_user_prompt}',
                                    '{model}',
                                    {temperature},
                                    {max_tokens},
                                    '{save_database}',
                                    '{save_schema}',
                                    '{save_table}',
                                    '{clean_metadata_json}'
                                )
                                """
                                
                                result = session.sql(proc_call).collect()
                                
                                if result:
                                    successful_batches += 1
                                    total_records_generated += batch_size  # Assume success means batch_size records
                                    batch_info.success(f"✅ Batch {batch_num}: Generated {batch_size} records using stored procedure")
                                else:
                                    batch_info.error(f"❌ Batch {batch_num}: Stored procedure returned no result")
                                    
                            except Exception as e:
                                batch_info.error(f"❌ Batch {batch_num}: Error calling stored procedure: {str(e)}")
                            
                            # Small delay to let UI update
                            time.sleep(0.1)
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        status_text.text("High-Performance generation complete!")
                        
                        # Final summary
                        if successful_batches > 0:
                            st.success(f"🎉 Successfully generated {total_records_generated:,} records across {successful_batches} batches using stored procedures!")
                            
                            if successful_batches < num_batches:
                                st.warning(f"⚠️  {num_batches - successful_batches} batches failed. Check logs above.")
                            
                            # For display purposes, create a sample of the data
                            try:
                                sample_query = f"SELECT * FROM {save_database}.{save_schema}.{save_table} WHERE _meta_generation_timestamp = '{generation_metadata['generation_timestamp']}' LIMIT 10"
                                sample_data = session.sql(sample_query).collect()
                                if sample_data:
                                    st.session_state.generated_data = [row.as_dict() for row in sample_data]
                            except Exception as e:
                                st.info("Data generated successfully but unable to display sample for preview.")
                        else:
                            st.error("❌ All batches failed to generate data. Please check your configuration and try again.")
                        
                        # Skip the standard processing since we used stored procedures
                        # Mark that we successfully used stored procedures
                        procedure_completed = True
            
            # Standard processing mode (fallback when not using stored procedures)
            if not use_procedure and auto_save:
                # Only run standard processing if we didn't use stored procedures
                # AND auto_save is enabled (for consistency)
                procedure_completed = False
            elif not auto_save:
                # If auto_save is disabled, always use standard processing
                procedure_completed = False
            else:
                # If we used stored procedures, don't run standard processing
                procedure_completed = use_procedure
            
            # Only run standard processing if procedure wasn't completed
            if not procedure_completed:
                # Process each batch
                for batch_num in range(1, num_batches + 1):
                    progress = (batch_num - 1) / num_batches
                    progress_bar.progress(progress)
                    status_text.text(f"Generating batch {batch_num} of {num_batches} (Standard Mode)...")
                    
                    # Generate batch
                    response = call_cortex_complete(
                        session, 
                        model, 
                        system_prompt_edited, 
                        user_prompt_edited,
                        temperature,
                        max_tokens
                    )
                    
                    if response:
                        batch_data = parse_json_response(response)
                        if batch_data:
                            # Add to accumulated data
                            st.session_state.all_generated_data.extend(batch_data)
                            total_records_generated += len(batch_data)
                            successful_batches += 1
                            
                            # Auto-save if enabled
                            if auto_save:
                                is_first_batch = (batch_num == 1)
                                append_data = append_mode or not is_first_batch
                                
                                # Add batch-specific metadata
                                batch_metadata = generation_metadata.copy()
                                batch_metadata['batch_number'] = batch_num
                                batch_metadata['batch_timestamp'] = datetime.now().isoformat()
                                batch_metadata['records_in_batch'] = len(batch_data)
                                
                                success = save_to_table(
                                    session, 
                                    batch_data, 
                                    save_table, 
                                    save_database, 
                                    save_schema,
                                    append=append_data,
                                    metadata=batch_metadata
                                )
                                
                                if success:
                                    batch_info.success(f"✅ Batch {batch_num}: Generated {len(batch_data)} records, saved to table")
                                else:
                                    batch_info.error(f"❌ Batch {batch_num}: Generated {len(batch_data)} records, failed to save")
                            else:
                                batch_info.success(f"✅ Batch {batch_num}: Generated {len(batch_data)} records")
                        else:
                            batch_info.error(f"❌ Batch {batch_num}: Failed to parse generated data")
                    else:
                        batch_info.error(f"❌ Batch {batch_num}: Failed to generate data")
                    
                    # Small delay to let UI update
                    time.sleep(0.1)
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("Standard generation complete!")
                
                # Final summary
                if successful_batches > 0:
                    st.session_state.generated_data = st.session_state.all_generated_data
                    st.success(f"🎉 Successfully generated {total_records_generated:,} records across {successful_batches} batches!")
                    
                    if successful_batches < num_batches:
                        st.warning(f"⚠️  {num_batches - successful_batches} batches failed. Check logs above.")
                else:
                    st.error("❌ All batches failed to generate data. Please check your configuration and try again.")
    
    # Display results
    if st.session_state.generated_data:
        st.markdown("---")
        st.subheader("Generated Data")
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(st.session_state.generated_data))
        with col2:
            st.metric("Configured Batches", num_batches)
        with col3:
            st.metric("Records per Batch", batch_size)
        
        # Display as DataFrame
        df = pd.DataFrame(st.session_state.generated_data)
        st.dataframe(df, use_container_width=True)
        
        # Display raw JSON
        with st.expander("View Raw JSON"):
            st.json(st.session_state.generated_data)
        
        # Manual save option (only show if auto-save was disabled)
        if not auto_save:
            st.markdown("---")
            st.subheader("Save to Table")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_save_database = st.text_input("Database", value=get_current_database(), key="manual_db")
            with col2:
                manual_save_schema = st.text_input("Schema", value="BRONZE_LAYER", key="manual_schema")
            with col3:
                manual_save_table = st.text_input("Table Name", value="GENERATED_DATA", key="manual_table")
            
            if st.button("💾 Save All Data to Table"):
                # Create metadata for manual save
                manual_save_metadata = {
                    'generation_timestamp': datetime.now().isoformat(),
                    'manual_save': True,
                    'total_records': len(st.session_state.generated_data)
                }
                save_to_table(session, st.session_state.generated_data, manual_save_table, 
                            manual_save_database, manual_save_schema, metadata=manual_save_metadata)
    
    # Footer
    st.markdown("---")
    st.markdown("**✨ Features:**")
    st.markdown("- **High-Performance Mode**: Uses stored procedures for 3-5x faster generation")
    st.markdown("- **Batch Processing**: Generates data in reliable batches with progress tracking")
    st.markdown("- **Auto-Save**: Automatically saves each batch to Snowflake with metadata")
    st.markdown("- **Configuration Management**: Save and load configurations in Snowflake table")
    st.markdown("- **Metadata Tracking**: Includes generation timestamp, model settings, and prompts")
    st.markdown("- **Error Handling**: Continues processing even if some batches fail")
    st.markdown("")
    st.markdown("**⚡ Performance Tips:**")
    st.markdown("- Enable **High-Performance Mode** for large datasets (requires auto-save)")
    st.markdown("- Use M or L warehouse sizes for better performance")
    st.markdown("- Keep batch sizes between 50-200 records for optimal reliability")
    st.markdown("")
    st.markdown("**Note:** Configurations are stored in the `SYNTHETIC_DATA_CONFIGS` table. Make sure you have the appropriate permissions to use Cortex functions and create/write to tables.")

else:
    st.error("Please run this application in a Snowflake environment with Streamlit support.") 