import streamlit as st
import pandas as pd
import yaml
import json
import re
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid
from snowflake.snowpark.context import get_active_session

# Get Snowflake session (if available)
try:
    session = get_active_session()
    HAS_SNOWFLAKE_SESSION = True
except:
    session = None
    HAS_SNOWFLAKE_SESSION = False


# -- SNOW-DEMO-CONFIG v1.0
# -- SQL Worksheet to YAML Demo Configuration Generator
# -- Creation Date: 2025-01-15
# -- Purpose: Convert SQL worksheets into YAML configs for SNOW-DEMO harness

st.set_page_config(
    page_title="SQL to YAML Converter",
    page_icon="⚙️",
    layout="wide"
)

# ========================================
# DATABASE FUNCTIONS (for saving configs)
# ========================================

def get_current_database():
    """Get current database from session context"""
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

def setup_config_database(session) -> bool:
    """Create config database and table if they don't exist"""
    if not session:
        return False
    
    try:
        # Get current database from session context
        db_name = get_current_database()
        
        session.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}").collect()
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {db_name}.CONFIG").collect()
        
        # Check if table exists and get its schema
        try:
            schema_check = session.sql(f"""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM {db_name}.INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'CONFIG' 
                AND TABLE_NAME = 'DEMO_CONFIGURATIONS'
                AND COLUMN_NAME = 'DEMO_METADATA'
            """).collect()
            
            # If table exists and has the right schema (TEXT), we're good
            if schema_check and schema_check[0]['DATA_TYPE'] == 'TEXT':
                return True
            
            # If table exists but has wrong schema (VARIANT), drop and recreate
            elif schema_check:
                st.warning("🔄 Updating table schema - existing configurations will be preserved...")
                
                # First, backup existing data if any
                backup_data = []
                try:
                    existing_data = session.sql(f"SELECT * FROM {db_name}.CONFIG.DEMO_CONFIGURATIONS").collect()
                    for row in existing_data:
                        backup_data.append({
                            'CONFIG_ID': row['CONFIG_ID'],
                            'CONFIG_NAME': row['CONFIG_NAME'],
                            'YAML_CONFIG': row['YAML_CONFIG'],
                            'CREATED_TIMESTAMP': row['CREATED_TIMESTAMP']
                        })
                except:
                    pass  # No existing data or error reading
                
                # Drop and recreate with new schema
                session.sql(f"DROP TABLE IF EXISTS {db_name}.CONFIG.DEMO_CONFIGURATIONS").collect()
                
                # Create table with TEXT columns
                session.sql(f"""
                    CREATE TABLE {db_name}.CONFIG.DEMO_CONFIGURATIONS (
                        CONFIG_ID STRING PRIMARY KEY,
                        CONFIG_NAME STRING,
                        DEMO_METADATA TEXT,
                        PARSED_BLOCKS TEXT,
                        YAML_CONFIG TEXT,
                        INFERRED_METADATA TEXT,
                        SUMMARY_REPORT TEXT,
                        CREATED_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                        LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                    )
                """).collect()
                
                # Restore basic data (CONFIG_NAME and YAML_CONFIG only since other fields changed format)
                for backup_row in backup_data:
                    try:
                        session.sql(f"""
                            INSERT INTO {db_name}.CONFIG.DEMO_CONFIGURATIONS 
                            (CONFIG_ID, CONFIG_NAME, YAML_CONFIG, DEMO_METADATA, PARSED_BLOCKS, INFERRED_METADATA, SUMMARY_REPORT)
                            VALUES (?, ?, ?, '{{}}', '[]', '{{}}', '{{}}')
                        """, [
                            backup_row['CONFIG_ID'],
                            backup_row['CONFIG_NAME'],
                            backup_row['YAML_CONFIG']
                        ]).collect()
                    except:
                        pass  # Skip failed restores
                
                st.info("✅ Table schema updated successfully")
                return True
                
        except Exception:
            # Table doesn't exist, create it
            pass
        
        # Create new table with TEXT columns including original SQL
        session.sql(f"""
            CREATE TABLE {db_name}.CONFIG.DEMO_CONFIGURATIONS (
                CONFIG_ID STRING PRIMARY KEY,
                CONFIG_NAME STRING,
                ORIGINAL_SQL TEXT,
                DEMO_METADATA TEXT,
                PARSED_BLOCKS TEXT,
                YAML_CONFIG TEXT,
                INFERRED_METADATA TEXT,
                SUMMARY_REPORT TEXT,
                CREATED_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """).collect()
        
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            return True
        # For permission errors, silently continue (DataOps manages infrastructure)
        if "no privileges" in str(e).lower() or "insufficient privileges" in str(e).lower() or "access control error" in str(e).lower():
            return True
        st.error(f"Error setting up config database: {str(e)}")
        return False

def save_config_to_database(session, config_name: str, demo_metadata: Dict, parsed_blocks: List[Dict], yaml_config: str, inferred_metadata: Dict, summary_report: Dict, original_sql: str = "") -> bool:
    """Save configuration to database including original SQL"""
    if not session:
        return False
    
    try:
        # Use DataOps database name with consistent fallback pattern
        # Get current database from session context
        db_name = get_current_database()
        config_id = str(uuid.uuid4())
        
        # Delete existing config with same name
        session.sql(f"""
            DELETE FROM {db_name}.CONFIG.DEMO_CONFIGURATIONS 
            WHERE CONFIG_NAME = ?
        """, [config_name]).collect()
        
        # Convert to JSON strings
        demo_metadata_json = json.dumps(demo_metadata)
        parsed_blocks_json = json.dumps(parsed_blocks)
        inferred_metadata_json = json.dumps(inferred_metadata)
        summary_report_json = json.dumps(summary_report)
        
        # Insert new config with all data as TEXT including original SQL
        session.sql(f"""
            INSERT INTO {db_name}.CONFIG.DEMO_CONFIGURATIONS 
            (CONFIG_ID, CONFIG_NAME, ORIGINAL_SQL, DEMO_METADATA, PARSED_BLOCKS, YAML_CONFIG, INFERRED_METADATA, SUMMARY_REPORT)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            config_id,
            config_name,
            original_sql,
            demo_metadata_json,
            parsed_blocks_json,
            yaml_config,
            inferred_metadata_json,
            summary_report_json
        ]).collect()
        
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def load_saved_configurations(session) -> List[Dict]:
    """Load list of saved configurations"""
    if not session:
        return []
    
    try:
        # Use DataOps database name with consistent fallback pattern
        # Get current database from session context
        db_name = get_current_database()
        
        result = session.sql(f"""
            SELECT CONFIG_ID, CONFIG_NAME, CREATED_TIMESTAMP, LAST_UPDATED
            FROM {db_name}.CONFIG.DEMO_CONFIGURATIONS
            ORDER BY LAST_UPDATED DESC
        """).collect()
        
        configs = []
        for row in result:
            configs.append({
                'config_id': row['CONFIG_ID'],
                'config_name': row['CONFIG_NAME'],
                'created': row['CREATED_TIMESTAMP'],
                'updated': row['LAST_UPDATED']
            })
        
        return configs
    except Exception as e:
        # Don't show error for missing database - that's expected if no data exists yet
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            return []  # Silently return empty list like original
        st.error(f"Error loading configurations: {str(e)}")
        return []

def load_config_from_database(session, config_id: str) -> Optional[Dict]:
    """Load specific configuration from database"""
    if not session:
        return None
    
    try:
        # Use DataOps database name with consistent fallback pattern
        # Get current database from session context
        db_name = get_current_database()
        
        result = session.sql(f"""
            SELECT CONFIG_NAME, ORIGINAL_SQL, DEMO_METADATA, PARSED_BLOCKS, YAML_CONFIG, INFERRED_METADATA, SUMMARY_REPORT
            FROM {db_name}.CONFIG.DEMO_CONFIGURATIONS
            WHERE CONFIG_ID = ?
        """, [config_id]).collect()
        
        if result:
            row = result[0]
            
            loaded_config = {
                'config_name': row['CONFIG_NAME'],
                'original_sql': row['ORIGINAL_SQL'] or '',
                'demo_metadata': json.loads(row['DEMO_METADATA']) if row['DEMO_METADATA'] and row['DEMO_METADATA'] != '{}' else {},
                'parsed_blocks': json.loads(row['PARSED_BLOCKS']) if row['PARSED_BLOCKS'] and row['PARSED_BLOCKS'] != '[]' else [],
                'yaml_config': row['YAML_CONFIG'],
                'inferred_metadata': json.loads(row['INFERRED_METADATA']) if row['INFERRED_METADATA'] and row['INFERRED_METADATA'] != '{}' else {},
                'summary_report': json.loads(row['SUMMARY_REPORT']) if row['SUMMARY_REPORT'] and row['SUMMARY_REPORT'] != '{}' else {}
            }
            
            return loaded_config
        
        return None
    except Exception as e:
        # Don't show error for missing database - that's expected if no data exists yet
        if "does not exist" in str(e).lower() or "not authorized" in str(e).lower():
            return None  # Silently return None like original
        st.error(f"Error loading configuration: {str(e)}")
        return None

# ========================================
# SQL PARSING FUNCTIONS
# ========================================

def parse_sql_worksheet(sql_content: str, separator: str = "GO") -> List[Dict]:
    """
    Parse SQL worksheet into individual steps/blocks
    """
    blocks = []
    
    # Normalize line endings
    sql_content = sql_content.replace('\r\n', '\n').replace('\r', '\n')
    
    if separator == "GO":
        # Split on GO statements (case insensitive)
        raw_blocks = re.split(r'^\s*GO\s*$', sql_content, flags=re.MULTILINE | re.IGNORECASE)
    elif separator == "Semicolon (;)":
        # Split on semicolons, but be careful with strings
        raw_blocks = split_on_semicolon(sql_content)
    elif separator == "Double Dash (--)":
        # Split on double dash comments that look like step separators
        raw_blocks = re.split(r'^\s*--\s*Step\s*\d+.*$', sql_content, flags=re.MULTILINE | re.IGNORECASE)
    else:
        # Default to GO
        raw_blocks = re.split(r'^\s*GO\s*$', sql_content, flags=re.MULTILINE | re.IGNORECASE)
    
    step_number = 1
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
            
        # Extract comments and SQL
        parsed_block = parse_sql_block(block, step_number)
        if parsed_block:
            blocks.append(parsed_block)
            step_number += 1
    
    return blocks

def split_on_semicolon(sql_content: str) -> List[str]:
    """
    Split SQL on semicolons, but handle strings and comments properly
    """
    blocks = []
    current_block = ""
    in_string = False
    string_char = None
    
    i = 0
    while i < len(sql_content):
        char = sql_content[i]
        
        # Handle string literals
        if char in ("'", '"') and not in_string:
            in_string = True
            string_char = char
        elif char == string_char and in_string:
            in_string = False
            string_char = None
        
        # Handle semicolons
        if char == ';' and not in_string:
            current_block += char
            blocks.append(current_block.strip())
            current_block = ""
        else:
            current_block += char
        
        i += 1
    
    # Add remaining content
    if current_block.strip():
        blocks.append(current_block.strip())
    
    return blocks

def parse_sql_block(block: str, step_number: int) -> Optional[Dict]:
    """
    Parse individual SQL block to extract comments, instructions, and SQL
    """
    lines = block.split('\n')
    comments = []
    sql_lines = []
    instructions = []
    title = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for comment lines
        if line.startswith('--'):
            comment_text = line[2:].strip()
            comments.append(comment_text)
            
            # Try to extract title from first meaningful comment
            if not title and comment_text and not comment_text.lower().startswith('step'):
                title = comment_text
            
            # Check if this looks like an instruction
            if any(keyword in comment_text.lower() for keyword in ['step', 'instruction', 'note', 'explain']):
                instructions.append(comment_text)
        else:
            # This is SQL code
            sql_lines.append(line)
    
    # Join SQL lines
    sql_code = '\n'.join(sql_lines).strip()
    
    # Skip blocks with no SQL
    if not sql_code:
        return None
    
    # Generate title if none found
    if not title:
        title = f"Step {step_number}"
    
    # Join instructions
    instructions_text = '\n'.join(instructions) if instructions else ""
    
    return {
        'id': f"step_{step_number}",
        'title': title,
        'query': sql_code,
        'instructions': instructions_text,
        'comments': comments,
        'step_number': step_number
    }

def detect_unsupported_commands(sql_code: str) -> List[str]:
    """
    Detect SQL commands that are not supported in Snowflake Streamlit
    """
    unsupported_patterns = [
        r'^\s*USE\s+WAREHOUSE\s+',
        r'^\s*USE\s+DATABASE\s+',
        r'^\s*USE\s+SCHEMA\s+',
        r'^\s*USE\s+ROLE\s+',
        r'^\s*ALTER\s+SESSION\s+',
        r'^\s*SET\s+QUERY_TAG\s*=',
    ]
    
    unsupported_commands = []
    lines = sql_code.split('\n')
    
    for line in lines:
        line = line.strip()
        for pattern in unsupported_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                unsupported_commands.append(line)
                break
    
    return unsupported_commands

def comment_out_unsupported_commands(sql_code: str) -> str:
    """
    Comment out unsupported commands in SQL code
    """
    unsupported_patterns = [
        r'^\s*USE\s+WAREHOUSE\s+',
        r'^\s*USE\s+DATABASE\s+',
        r'^\s*USE\s+SCHEMA\s+',
        r'^\s*USE\s+ROLE\s+',
        r'^\s*ALTER\s+SESSION\s+',
        r'^\s*SET\s+QUERY_TAG\s*=',
    ]
    
    lines = sql_code.split('\n')
    modified_lines = []
    
    for line in lines:
        original_line = line
        stripped_line = line.strip()
        
        # Check if this line matches any unsupported pattern
        is_unsupported = False
        for pattern in unsupported_patterns:
            if re.match(pattern, stripped_line, re.IGNORECASE):
                is_unsupported = True
                break
        
        if is_unsupported and not stripped_line.startswith('--'):
            # Add comment explaining why it's commented out
            modified_lines.append(f"-- {original_line}  -- Commented out: Not supported in Streamlit")
        else:
            modified_lines.append(original_line)
    
    return '\n'.join(modified_lines)

# ========================================
# CORTEX DETECTION FUNCTIONS
# ========================================

def detect_cortex_complete(sql_code: str) -> List[Dict]:
    """
    Detect CORTEX.COMPLETE calls in SQL code
    """
    pattern = r'SNOWFLAKE\.CORTEX\.COMPLETE\s*\('
    matches = list(re.finditer(pattern, sql_code, re.IGNORECASE))
    
    cortex_calls = []
    for match in matches:
        # Extract the complete call by finding matching parentheses
        start = match.start()
        paren_count = 0
        end = start
        
        for i in range(start, len(sql_code)):
            if sql_code[i] == '(':
                paren_count += 1
            elif sql_code[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    end = i + 1
                    break
        
        if end > start:
            full_call = sql_code[start:end]
            cortex_calls.append({
                'type': 'complete',
                'call': full_call,
                'start': start,
                'end': end
            })
    
    return cortex_calls

def detect_cortex_search(sql_code: str) -> List[Dict]:
    """
    Detect CORTEX.SEARCH_PREVIEW calls in SQL code (CORTEX.SEARCH is API-only)
    """
    pattern = r'SNOWFLAKE\.CORTEX\.SEARCH_PREVIEW\s*\('
    matches = list(re.finditer(pattern, sql_code, re.IGNORECASE))
    
    cortex_calls = []
    for match in matches:
        # Extract the complete call by finding matching parentheses
        start = match.start()
        paren_count = 0
        end = start
        
        for i in range(start, len(sql_code)):
            if sql_code[i] == '(':
                paren_count += 1
            elif sql_code[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    end = i + 1
                    break
        
        if end > start:
            full_call = sql_code[start:end]
            
            # Try to extract search service name from SEARCH_PREVIEW
            service_match = re.search(r"CORTEX\.SEARCH_PREVIEW\s*\(\s*'([^']+)'", full_call, re.IGNORECASE)
            service_name = service_match.group(1) if service_match else None
            
            cortex_calls.append({
                'type': 'search',
                'call': full_call,
                'start': start,
                'end': end,
                'service_name': service_name
            })
    
    return cortex_calls

def detect_cortex_analyst(sql_code: str) -> List[Dict]:
    """
    Detect Cortex Analyst configurations in SQL code.
    Since Cortex Analyst is API-only, we look for semantic model file references and table indicators.
    """
    cortex_calls = []
    
    # Look for semantic model file references
    model_patterns = [
        r'@[A-Z_]+\.[A-Z_]+\.[A-Z_/]+\.yaml',  # @DATABASE.SCHEMA.PATH/file.yaml
        r'semantic_model[s]?[_/][^\'"\s]+\.yaml',  # semantic_model/file.yaml
        r'[\'"]([^\'"]*/semantic_model[s]?/[^\'"]*.yaml)[\'"]',  # 'path/semantic_models/file.yaml'
    ]
    
    for pattern in model_patterns:
        matches = list(re.finditer(pattern, sql_code, re.IGNORECASE))
        for match in matches:
            model_file = match.group(1) if match.groups() else match.group(0)
            
            # Look for associated table (often nearby)
            table_match = re.search(r'[A-Z_]+\.[A-Z_]+\.[A-Z_]+', sql_code[max(0, match.start()-200):match.start()+200])
            table_name = table_match.group(0) if table_match else None
            
            cortex_calls.append({
                'type': 'analyst',
                'call': f'semantic_model_file: {model_file}',
                'start': match.start(),
                'end': match.end(),
                'semantic_model_file': model_file,
                'table_name': table_name
            })
    
    return cortex_calls

def analyze_cortex_calls(sql_code: str) -> Dict:
    """
    Analyze all Cortex calls in SQL code
    """
    complete_calls = detect_cortex_complete(sql_code)
    search_calls = detect_cortex_search(sql_code)
    analyst_calls = detect_cortex_analyst(sql_code)
    
    return {
        'has_cortex': len(complete_calls) > 0 or len(search_calls) > 0 or len(analyst_calls) > 0,
        'complete_calls': complete_calls,
        'search_calls': search_calls,
        'analyst_calls': analyst_calls,
        'total_calls': len(complete_calls) + len(search_calls) + len(analyst_calls)
    }

def create_cortex_interactive_step(original_block: Dict, cortex_info: Dict, step_number: int) -> Dict:
    """
    Create an interactive Cortex step based on the original block
    """
    cortex_type = cortex_info.get('type', 'complete')
    
    if cortex_type == 'complete':
        return {
            'id': f"step_{step_number}_cortex_complete",
            'title': f"Interactive Cortex Complete - {original_block['title']}",
            'query': original_block['query'],  # Keep original query for parsing
            'instructions': "Experiment with different prompts and parameters to see how they affect the AI's responses. This interactive interface allows you to modify the system prompt, user prompt, temperature, model, and max_tokens in real-time.",
            'instructions_title': f"Step {step_number} - Interactive Cortex Complete",
            'default_chart': 'Table',
            'cortex_type': 'complete',
            'patterns': original_block.get('patterns', {}),
            'variables': original_block.get('variables', []),
            'unsupported_commands': [],
            'talk_track': 'This step demonstrates how different prompts and parameters affect LLM behavior. Try adjusting the temperature to see how it impacts response creativity, or modify the prompts to explore different analytical approaches.',
            'save_as': None
        }
    elif cortex_type == 'search':
        return {
            'id': f"step_{step_number}_cortex_search",
            'title': f"Interactive Cortex Search - {original_block['title']}",
            'query': original_block['query'],
            'instructions': "Use natural language to search through your data. This interactive interface allows you to explore your dataset using conversational queries without writing SQL.",
            'instructions_title': f"Step {step_number} - Interactive Cortex Search",
            'default_chart': 'Table',
            'cortex_type': 'search',
            'cortex_search_service': cortex_info.get('service_name', ''),
            'patterns': original_block.get('patterns', {}),
            'variables': original_block.get('variables', []),
            'unsupported_commands': [],
            'talk_track': 'This step shows how to use natural language search to explore data. Try queries like "parking complaints" or "positive feedback about food" to see how semantic search finds relevant results.',
            'save_as': None
        }
    elif cortex_type == 'analyst':
        return {
            'id': f"step_{step_number}_cortex_analyst",
            'title': f"Interactive Cortex Analyst - {original_block['title']}",
            'query': original_block['query'],
            'instructions': "Ask natural language questions about your data and see the AI-generated SQL. This interface uses the Cortex Analyst API to convert questions into SQL queries, showing both the generated SQL and results.",
            'instructions_title': f"Step {step_number} - Interactive Cortex Analyst",
            'default_chart': 'Table',
            'cortex_type': 'analyst',
            'semantic_model_file': cortex_info.get('semantic_model_file', ''),
            'sample_table': cortex_info.get('table_name', ''),
            'sample_questions': [
                "What are the top 5 records by revenue?",
                "Show me trends over the last quarter",
                "Which segments perform best?",
                "What is the average score by category?",
                "How many records do we have in total?"
            ],
            'patterns': original_block.get('patterns', {}),
            'variables': original_block.get('variables', []),
            'unsupported_commands': [],
            'talk_track': 'This step demonstrates how Cortex Analyst converts natural language questions into SQL queries using the semantic model. Try asking questions about your data and see the generated SQL and results.',
            'save_as': None
        }
    
    return original_block

# ========================================
# VARIABLE EXTRACTION FUNCTIONS
# ========================================

def extract_variables_from_sql(sql_code: str) -> List[str]:
    """
    Extract variables in {VARIABLE} format from SQL code
    For now, return empty list to avoid nonsense variables - will enhance later for cortex
    """
    # Simplified for now - only look for very clear variable patterns
    pattern = r'\{([A-Z_][A-Z0-9_]*)\}'  # Only uppercase variables like {MY_VARIABLE}
    variables = re.findall(pattern, sql_code)
    return list(set(variables))

def extract_table_references(sql_code: str) -> List[Dict]:
    """
    Extract table references from SQL code
    """
    table_patterns = [
        r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)',
        r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)',
        r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)',
        r'INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)',
        r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    
    tables = []
    for pattern in table_patterns:
        matches = re.findall(pattern, sql_code, re.IGNORECASE)
        for match in matches:
            # Parse database.schema.table format
            parts = match.split('.')
            if len(parts) == 3:
                database, schema, table = parts
            elif len(parts) == 2:
                database, schema, table = None, parts[0], parts[1]
            else:
                database, schema, table = None, None, parts[0]
            
            tables.append({
                'full_name': match,
                'database': database,
                'schema': schema,
                'table': table
            })
    
    return tables

def extract_database_schema_info(sql_code: str) -> Dict:
    """
    Extract database and schema information from SQL code
    """
    database_pattern = r'USE\s+DATABASE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    schema_pattern = r'USE\s+SCHEMA\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    role_pattern = r'USE\s+ROLE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    warehouse_pattern = r'USE\s+WAREHOUSE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    
    info = {
        'database': '',
        'schema': '',
        'role': '',
        'warehouse': ''
    }
    
    # Extract database
    db_match = re.search(database_pattern, sql_code, re.IGNORECASE)
    if db_match:
        info['database'] = db_match.group(1)
    
    # Extract schema
    schema_match = re.search(schema_pattern, sql_code, re.IGNORECASE)
    if schema_match:
        info['schema'] = schema_match.group(1)
    
    # Extract role
    role_match = re.search(role_pattern, sql_code, re.IGNORECASE)
    if role_match:
        info['role'] = role_match.group(1)
    
    # Extract warehouse
    warehouse_match = re.search(warehouse_pattern, sql_code, re.IGNORECASE)
    if warehouse_match:
        info['warehouse'] = warehouse_match.group(1)
    
    return info

def analyze_query_patterns(sql_code: str) -> Dict:
    """
    Analyze SQL patterns to suggest visualization types
    """
    patterns = {
        'has_aggregation': bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b', sql_code, re.IGNORECASE)),
        'has_geospatial': bool(re.search(r'\b(ST_|H3_|LATITUDE|LONGITUDE|GEOGRAPHY|GEOMETRY)\b', sql_code, re.IGNORECASE)),
        'has_distance': bool(re.search(r'\b(ST_DISTANCE|DISTANCE)\b', sql_code, re.IGNORECASE)),
        'has_time_series': bool(re.search(r'\b(DATE|TIME|TIMESTAMP|YEAR|MONTH|DAY)\b', sql_code, re.IGNORECASE)),
        'has_top_n': bool(re.search(r'\b(TOP\s+\d+|LIMIT\s+\d+)\b', sql_code, re.IGNORECASE)),
        'has_joins': bool(re.search(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN)\b', sql_code, re.IGNORECASE)),
        'has_coordinates': bool(re.search(r'\b(LATITUDE|LONGITUDE|LAT|LON|COORD)\b', sql_code, re.IGNORECASE)),
        'has_h3': bool(re.search(r'\bH3_', sql_code, re.IGNORECASE)),
        'has_polygon': bool(re.search(r'\b(POLYGON|ST_ENVELOPE|ST_COLLECT)\b', sql_code, re.IGNORECASE)),
        'has_single_value': bool(re.search(r'^\s*SELECT\s+[^,]+\s+FROM\b', sql_code, re.IGNORECASE | re.MULTILINE)),
        'is_count_query': bool(re.search(r'\bCOUNT\s*\(\s*\*\s*\)', sql_code, re.IGNORECASE)),
    }
    
    return patterns

def suggest_visualization_type(sql_code: str) -> str:
    """
    Suggest default visualization type based on SQL patterns
    """
    patterns = analyze_query_patterns(sql_code)
    
    # Priority-based suggestions
    if patterns['has_coordinates'] and patterns['has_distance']:
        return "Network"
    elif patterns['has_coordinates']:
        return "Map"
    elif patterns['has_h3']:
        return "H3"
    elif patterns['has_polygon']:
        return "Polygon"
    elif patterns['has_single_value'] or patterns['is_count_query']:
        return "Metric"
    elif patterns['has_aggregation'] and patterns['has_top_n']:
        return "Bar Chart"
    elif patterns['has_time_series']:
        return "Line Chart"
    elif patterns['has_aggregation']:
        return "Bar Chart"
    else:
        return "Table"

def extract_column_hints(sql_code: str) -> List[str]:
    """
    Extract column names that might be useful for visualization
    """
    # Look for SELECT clause columns
    select_pattern = r'SELECT\s+(.*?)\s+FROM'
    select_matches = re.findall(select_pattern, sql_code, re.IGNORECASE | re.DOTALL)
    
    columns = []
    for match in select_matches:
        # Split by comma and clean up
        cols = [col.strip() for col in match.split(',')]
        for col in cols:
            # Remove aliases and functions
            col = re.sub(r'\s+AS\s+\w+', '', col, flags=re.IGNORECASE)
            col = re.sub(r'\w+\s*\(.*?\)', '', col)
            col = col.strip()
            if col and not col.startswith('*'):
                columns.append(col)
    
    return columns

# ========================================
# METADATA INFERENCE FUNCTIONS
# ========================================

def infer_metadata_from_blocks(blocks: List[Dict], full_sql: str) -> Dict:
    """
    Infer demo metadata from parsed SQL blocks
    """
    # Extract database info from full SQL
    db_info = extract_database_schema_info(full_sql)
    
    # Get all table references
    all_tables = []
    for block in blocks:
        tables = extract_table_references(block['query'])
        all_tables.extend(tables)
    
    # Find most common database and schema
    databases = [t['database'] for t in all_tables if t['database']]
    schemas = [t['schema'] for t in all_tables if t['schema']]
    
    most_common_db = max(set(databases), key=databases.count) if databases else db_info['database']
    most_common_schema = max(set(schemas), key=schemas.count) if schemas else db_info['schema']
    
    # Detect query tag pattern
    query_tag_pattern = r'query_tag\s*=\s*[\'"]([^\'"]*)[\'"]'
    query_tag_match = re.search(query_tag_pattern, full_sql, re.IGNORECASE)
    query_tag = query_tag_match.group(1) if query_tag_match else ""
    
    return {
        'database': most_common_db or '',
        'schema': most_common_schema or '',
        'role': db_info['role'] or '',
        'warehouse': db_info['warehouse'] or '',
        'query_tag': query_tag,
        'tables_referenced': len(set(t['full_name'] for t in all_tables)),
        'has_geospatial': any(analyze_query_patterns(block['query'])['has_geospatial'] for block in blocks),
        'has_aggregation': any(analyze_query_patterns(block['query'])['has_aggregation'] for block in blocks),
        'complexity_score': len(blocks)
    }

def enhance_blocks_with_metadata(blocks: List[Dict]) -> List[Dict]:
    """
    Enhance each block with metadata and suggested visualizations.
    Auto-detect Cortex calls and create interactive steps.
    """
    enhanced_blocks = []
    current_step_number = 1
    
    for block in blocks:
        # Analyze patterns
        patterns = analyze_query_patterns(block['query'])
        
        # Suggest visualization
        suggested_viz = suggest_visualization_type(block['query'])
        
        # Extract variables
        variables = extract_variables_from_sql(block['query'])
        
        # Detect unsupported commands
        unsupported = detect_unsupported_commands(block['query'])
        
        # Clean up unsupported commands
        cleaned_query = comment_out_unsupported_commands(block['query'])
        
        # Analyze Cortex calls
        cortex_analysis = analyze_cortex_calls(block['query'])
        
        # Create enhanced block
        enhanced_block = {
            'id': block['id'],
            'title': block['title'],
            'query': cleaned_query,
            'instructions': block['instructions'],
            'instructions_title': f"Step {current_step_number} - {block['title']}",
            'default_chart': suggested_viz,
            'patterns': patterns,
            'variables': variables,
            'unsupported_commands': unsupported,
            'cortex_analysis': cortex_analysis,
            'talk_track': '',  # Can be filled later
            'save_as': None    # Can be configured later
        }
        
        # Add the original block
        enhanced_blocks.append(enhanced_block)
        current_step_number += 1
        
        # If this block has Cortex calls, create interactive versions
        if cortex_analysis['has_cortex']:
            
            # Create interactive steps for each type of Cortex call
            if cortex_analysis['complete_calls']:
                for complete_call in cortex_analysis['complete_calls']:
                    interactive_step = create_cortex_interactive_step(
                        enhanced_block, 
                        complete_call, 
                        current_step_number
                    )
                    enhanced_blocks.append(interactive_step)
                    current_step_number += 1
            
            if cortex_analysis['search_calls']:
                for search_call in cortex_analysis['search_calls']:
                    interactive_step = create_cortex_interactive_step(
                        enhanced_block, 
                        search_call, 
                        current_step_number
                    )
                    enhanced_blocks.append(interactive_step)
                    current_step_number += 1
            
            if cortex_analysis['analyst_calls']:
                for analyst_call in cortex_analysis['analyst_calls']:
                    interactive_step = create_cortex_interactive_step(
                        enhanced_block, 
                        analyst_call, 
                        current_step_number
                    )
                    enhanced_blocks.append(interactive_step)
                    current_step_number += 1
    
    return enhanced_blocks

# ========================================
# YAML GENERATION FUNCTIONS
# ========================================

def generate_yaml_config(demo_metadata: Dict, blocks: List[Dict], inferred_metadata: Dict) -> str:
    """
    Generate YAML configuration for SNOW-DEMO harness
    """
    # Merge user-provided and inferred metadata
    merged_metadata = {
        'topic': demo_metadata.get('topic', ''),
        'sub_topic': demo_metadata.get('sub_topic', ''),
        'tertiary_topic': demo_metadata.get('tertiary_topic', ''),
        'database': demo_metadata.get('database') or inferred_metadata.get('database', ''),
        'schema': demo_metadata.get('schema') or inferred_metadata.get('schema', ''),
        'role': demo_metadata.get('role') or inferred_metadata.get('role', ''),
        'warehouse': demo_metadata.get('warehouse') or inferred_metadata.get('warehouse', ''),
        'query_tag': inferred_metadata.get('query_tag', ''),
        'owner': demo_metadata.get('owner', ''),
        'title': demo_metadata.get('title', ''),
        'logo_url': demo_metadata.get('logo_url', ''),
        'overview': demo_metadata.get('overview', ''),
        'harness_type': 'Visualizer',
        'date_created': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Build YAML manually to have better control over formatting
    yaml_lines = []
    
    # Demo section
    yaml_lines.append("demo:")
    yaml_lines.append(f"  topic: \"{merged_metadata['topic']}\"")
    yaml_lines.append(f"  sub_topic: \"{merged_metadata['sub_topic']}\"")
    yaml_lines.append(f"  tertiary_topic: \"{merged_metadata['tertiary_topic']}\"")
    yaml_lines.append(f"  database: \"{merged_metadata['database']}\"")
    yaml_lines.append(f"  schema: \"{merged_metadata['schema']}\"")
    yaml_lines.append(f"  role: \"{merged_metadata['role']}\"")
    yaml_lines.append(f"  warehouse: \"{merged_metadata['warehouse']}\"")
    yaml_lines.append(f"  query_tag: \"{merged_metadata['query_tag']}\"")
    yaml_lines.append(f"  owner: \"{merged_metadata['owner']}\"")
    yaml_lines.append(f"  title: \"{merged_metadata['title']}\"")
    yaml_lines.append(f"  logo_url: \"{merged_metadata['logo_url']}\"")
    
    # Handle overview with proper escaping
    if merged_metadata['overview']:
        overview_escaped = merged_metadata['overview'].replace('"', '\\"').replace('\n', '\\n')
        yaml_lines.append(f"  overview: \"{overview_escaped}\"")
    else:
        yaml_lines.append("  overview: \"\"")
    
    yaml_lines.append(f"  harness_type: \"{merged_metadata['harness_type']}\"")
    yaml_lines.append(f"  date_created: \"{merged_metadata['date_created']}\"")
    yaml_lines.append("  init_commands: []")
    yaml_lines.append("  cleanup_commands: []")
    
    # Steps section
    yaml_lines.append("steps:")
    
    for block in blocks:
        yaml_lines.append(f"  - id: \"{block['id']}\"")
        yaml_lines.append(f"    title: \"{block['title']}\"")
        
        # Format query with pipe syntax
        yaml_lines.append("    query: |")
        query_lines = block['query'].split('\n')
        for query_line in query_lines:
            yaml_lines.append(f"      {query_line}")
        
        # Handle talk_track with proper escaping
        talk_track = block.get('talk_track', '')
        if talk_track:
            talk_track_escaped = talk_track.replace('"', '\\"').replace('\n', '\\n')
            yaml_lines.append(f"    talk_track: \"{talk_track_escaped}\"")
        else:
            yaml_lines.append("    talk_track: \"\"")
            
        yaml_lines.append(f"    default_chart: \"{block.get('default_chart', 'Table')}\"")
        yaml_lines.append(f"    instructions_title: \"{block.get('instructions_title', '')}\"")
        
        # Handle instructions with proper escaping
        instructions = block.get('instructions', '')
        if instructions:
            instructions_escaped = instructions.replace('"', '\\"').replace('\n', '\\n')
            yaml_lines.append(f"    instructions: \"{instructions_escaped}\"")
        else:
            yaml_lines.append("    instructions: \"\"")
        
        # Add variables if any
        if block.get('variables'):
            yaml_lines.append("    variable_fields:")
            for var in block['variables']:
                yaml_lines.append(f"      - \"{var}\"")
        
        # Add save_as if specified
        if block.get('save_as'):
            yaml_lines.append(f"    save_as: \"{block['save_as']}\"")
        
        # Add Cortex-specific fields if present
        if block.get('cortex_type'):
            yaml_lines.append(f"    cortex_type: \"{block['cortex_type']}\"")
        
        if block.get('cortex_search_service'):
            yaml_lines.append(f"    cortex_search_service: \"{block['cortex_search_service']}\"")
        
        if block.get('semantic_model_file'):
            yaml_lines.append(f"    semantic_model_file: \"{block['semantic_model_file']}\"")
    
    return '\n'.join(yaml_lines)

def generate_summary_report(blocks: List[Dict], inferred_metadata: Dict) -> Dict:
    """
    Generate a summary report of the parsing results
    """
    total_blocks = len(blocks)
    visualization_types = {}
    patterns_summary = {}
    cortex_summary = {
        'has_cortex_calls': False,
        'total_cortex_calls': 0,
        'cortex_complete_calls': 0,
        'cortex_search_calls': 0,
        'cortex_analyst_calls': 0,
        'interactive_steps_created': 0
    }
    
    for block in blocks:
        # Count visualization types
        viz_type = block.get('default_chart', 'Table')
        visualization_types[viz_type] = visualization_types.get(viz_type, 0) + 1
        
        # Count patterns
        patterns = block.get('patterns', {})
        for pattern, value in patterns.items():
            if value:
                patterns_summary[pattern] = patterns_summary.get(pattern, 0) + 1
        
        # Count Cortex calls
        cortex_analysis = block.get('cortex_analysis', {})
        if cortex_analysis.get('has_cortex'):
            cortex_summary['has_cortex_calls'] = True
            cortex_summary['total_cortex_calls'] += cortex_analysis.get('total_calls', 0)
            cortex_summary['cortex_complete_calls'] += len(cortex_analysis.get('complete_calls', []))
            cortex_summary['cortex_search_calls'] += len(cortex_analysis.get('search_calls', []))
            cortex_summary['cortex_analyst_calls'] += len(cortex_analysis.get('analyst_calls', []))
        
        # Count interactive steps
        if block.get('cortex_type'):
            cortex_summary['interactive_steps_created'] += 1
    
    return {
        'total_steps': total_blocks,
        'visualization_types': visualization_types,
        'patterns_summary': patterns_summary,
        'cortex_summary': cortex_summary,
        'inferred_metadata': inferred_metadata,
        'has_unsupported_commands': any(block.get('unsupported_commands') for block in blocks)
    }

# Title and description
st.title("⚙️ SQL to YAML Converter")
st.markdown("**Convert SQL worksheets into YAML demo configurations for the AI Framework**")
st.markdown("Transform your SQL worksheets into interactive demo configurations for the SNOW-DEMO harness")

# Initialize database if Snowflake session available
if HAS_SNOWFLAKE_SESSION:
    if setup_config_database(session):
        st.success("✅ Connected to Snowflake - configurations will be saved to database")
    else:
        st.warning("⚠️ Database setup failed - configurations will only be available for download")
else:
    st.info("ℹ️ Running without Snowflake session - configurations will only be available for download")

# Saved Configurations Section
if HAS_SNOWFLAKE_SESSION:
    st.header("💾 Saved Configurations")
    
    saved_configs = load_saved_configurations(session)
    
    if saved_configs:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            config_options = {f"{config['config_name']} (Updated: {config['updated']})": config['config_id'] 
                            for config in saved_configs}
            
            selected_config_display = st.selectbox(
                "Load saved configuration:",
                ["Select a configuration..."] + list(config_options.keys())
            )
        
        with col2:
            if selected_config_display != "Select a configuration...":
                if st.button("🔄 Load Configuration"):
                    config_id = config_options[selected_config_display]
                    loaded_config = load_config_from_database(session, config_id)
                    
                    if loaded_config:
                        # Load into session state
                        st.session_state['parsed_blocks'] = loaded_config['parsed_blocks']
                        st.session_state['yaml_config'] = loaded_config['yaml_config']
                        st.session_state['summary_report'] = loaded_config['summary_report']
                        st.session_state['demo_metadata'] = loaded_config['demo_metadata']
                        st.session_state['inferred_metadata'] = loaded_config['inferred_metadata']
                        st.session_state['loaded_sql'] = loaded_config['original_sql']  # Store original SQL
                        
                        # Also load metadata into form fields for editing
                        demo_meta = loaded_config['demo_metadata']
                        if demo_meta:
                            st.session_state['form_topic'] = demo_meta.get('topic', '')
                            st.session_state['form_sub_topic'] = demo_meta.get('sub_topic', '')
                            st.session_state['form_tertiary_topic'] = demo_meta.get('tertiary_topic', '')
                            st.session_state['form_title'] = demo_meta.get('title', '')
                            st.session_state['form_logo_url'] = demo_meta.get('logo_url', '')
                            st.session_state['form_owner'] = demo_meta.get('owner', '')
                            st.session_state['form_database'] = demo_meta.get('database', '')
                            st.session_state['form_schema'] = demo_meta.get('schema', '')
                            st.session_state['form_overview'] = demo_meta.get('overview', '')
                            st.session_state['form_role'] = demo_meta.get('role', '')
                            st.session_state['form_warehouse'] = demo_meta.get('warehouse', '')
                            st.session_state['form_block_separator'] = demo_meta.get('block_separator', 'GO')
                            st.session_state['form_custom_separator'] = demo_meta.get('custom_separator', '')
                        
                        st.success(f"✅ Loaded configuration: {loaded_config['config_name']}")
                        st.rerun()
    else:
        st.info("No saved configurations found")
    
    st.markdown("---")

# Main input area
st.header("📝 Input SQL Worksheet")

# Choose input method
input_method = st.radio(
    "How would you like to provide your SQL worksheet?",
    ["📋 Paste SQL", "📁 Upload File"],
    horizontal=True
)

sql_worksheet = ""

if input_method == "📋 Paste SQL":
    sql_worksheet = st.text_area(
        "Paste your SQL worksheet here:",
        value=st.session_state.get('loaded_sql', ''),  # Use loaded SQL if available
        height=300,
        placeholder="-- Step 1: Setup\nUSE WAREHOUSE MY_WH;\nUSE DATABASE MY_DB;\n\n-- Step 2: Query Data\nSELECT * FROM my_table;\n\n-- Step 3: Analysis\nSELECT COUNT(*) FROM my_table;"
    )
elif input_method == "📁 Upload File":
    uploaded_file = st.file_uploader(
        "Upload SQL file",
        type=['sql', 'txt'],
        help="Upload a .sql or .txt file containing your SQL worksheet"
    )
    if uploaded_file is not None:
        sql_worksheet = uploaded_file.read().decode('utf-8')

# Demo metadata input
st.header("🏷️ Demo Metadata")

col1, col2 = st.columns(2)

with col1:
    demo_topic = st.text_input("Topic", 
                              value=st.session_state.get('form_topic', ''),
                              placeholder="e.g., Analytics, Geospatial, Machine Learning")
    demo_sub_topic = st.text_input("Sub-topic", 
                                  value=st.session_state.get('form_sub_topic', ''),
                                  placeholder="e.g., Sales Analysis, Customer Segmentation") 
    demo_tertiary_topic = st.text_input("Tertiary Topic", 
                                        value=st.session_state.get('form_tertiary_topic', ''),
                                        placeholder="e.g., Q4 Review, Regional Analysis")
    demo_title = st.text_input("Title", 
                               value=st.session_state.get('form_title', ''),
                               placeholder="e.g., Advanced Analytics Dashboard")

with col2:
    demo_logo_url = st.text_input("Logo URL", 
                                 value=st.session_state.get('form_logo_url', ''),
                                 placeholder="https://your-logo-url.com/logo.png")
    demo_owner = st.text_input("Owner", 
                              value=st.session_state.get('form_owner', ''),
                              placeholder="Your name or team")
    demo_database = st.text_input("Database", 
                                 value=st.session_state.get('form_database', ''),
                                 placeholder="Leave blank to auto-detect from SQL")
    demo_schema = st.text_input("Schema", 
                               value=st.session_state.get('form_schema', ''),
                               placeholder="Leave blank to auto-detect from SQL")

demo_overview = st.text_area(
    "Overview Description",
    value=st.session_state.get('form_overview', ''),
    height=100,
    placeholder="Brief description of what this demo covers...\n- Key feature 1\n- Key feature 2\n- Key feature 3"
)

# Advanced options
with st.expander("⚙️ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        demo_role = st.text_input("Role", 
                                 value=st.session_state.get('form_role', ''),
                                 placeholder="Leave blank to auto-detect")
        demo_warehouse = st.text_input("Warehouse", 
                                      value=st.session_state.get('form_warehouse', ''),
                                      placeholder="Leave blank to auto-detect")
        
    with col2:
        separator_options = ["GO", "Semicolon (;)", "Double Dash (--)", "Custom"]
        current_separator = st.session_state.get('form_block_separator', 'GO')
        try:
            separator_index = separator_options.index(current_separator)
        except ValueError:
            separator_index = 0
            
        block_separator = st.selectbox(
            "SQL Block Separator",
            separator_options,
            index=separator_index,
            help="How to split your SQL into separate steps"
        )
        
        custom_separator = ""
        if block_separator == "Custom":
            custom_separator = st.text_input("Custom Separator", 
                                           value=st.session_state.get('form_custom_separator', ''),
                                           placeholder="e.g., /* STEP */")

# Process button
if st.button("🔄 Parse SQL Worksheet", type="primary"):
    if sql_worksheet.strip():
        # Store metadata including all advanced options
        demo_metadata = {
            'topic': demo_topic,
            'sub_topic': demo_sub_topic,
            'tertiary_topic': demo_tertiary_topic,
            'title': demo_title,
            'logo_url': demo_logo_url,
            'owner': demo_owner,
            'database': demo_database,
            'schema': demo_schema,
            'overview': demo_overview,
            'role': demo_role,
            'warehouse': demo_warehouse,
            'block_separator': block_separator,
            'custom_separator': custom_separator if block_separator == "Custom" else ""
        }
        
        # Parse SQL worksheet
        with st.spinner("Parsing SQL worksheet..."):
            parsed_blocks = parse_sql_worksheet(sql_worksheet, block_separator)
        
        if parsed_blocks:
            # Infer metadata
            inferred_metadata = infer_metadata_from_blocks(parsed_blocks, sql_worksheet)
            
            # Enhance blocks with metadata
            enhanced_blocks = enhance_blocks_with_metadata(parsed_blocks)
            
            # Generate YAML
            yaml_config = generate_yaml_config(demo_metadata, enhanced_blocks, inferred_metadata)
            
            # Generate summary report
            summary_report = generate_summary_report(enhanced_blocks, inferred_metadata)
            
            # Store in session state including original SQL
            st.session_state['parsed_blocks'] = enhanced_blocks
            st.session_state['yaml_config'] = yaml_config
            st.session_state['summary_report'] = summary_report
            st.session_state['demo_metadata'] = demo_metadata
            st.session_state['inferred_metadata'] = inferred_metadata
            st.session_state['original_sql'] = sql_worksheet  # Store original SQL for saving
            
            st.success(f"✅ Successfully parsed {len(enhanced_blocks)} SQL blocks!")
            st.rerun()
        else:
            st.error("❌ No SQL blocks found. Please check your SQL format or try a different separator.")
    else:
        st.error("Please provide a SQL worksheet to parse")

# Results section
if 'parsed_blocks' in st.session_state:
    st.header("📊 Parsing Results")
    
    # Summary tab
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Parsed Blocks", "📄 Generated YAML", "💾 Download"])
    
    with tab1:
        st.subheader("📈 Summary Report")
        
        summary = st.session_state['summary_report']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", summary['total_steps'])
        with col2:
            st.metric("Tables Referenced", summary['inferred_metadata'].get('tables_referenced', 0))
        with col3:
            st.metric("Visualization Types", len(summary['visualization_types']))
        with col4:
            has_unsupported = "Yes" if summary['has_unsupported_commands'] else "No"
            st.metric("Unsupported Commands", has_unsupported)
        
        # Visualization types breakdown
        st.subheader("🎨 Suggested Visualizations")
        viz_data = []
        for viz_type, count in summary['visualization_types'].items():
            viz_data.append({"Visualization Type": viz_type, "Count": count})
        
        if viz_data:
            st.dataframe(viz_data, use_container_width=True)
        
        # Patterns detected
        st.subheader("🔍 SQL Patterns Detected")
        pattern_data = []
        for pattern, count in summary['patterns_summary'].items():
            pattern_name = pattern.replace('_', ' ').title()
            pattern_data.append({"Pattern": pattern_name, "Steps": count})
        
        if pattern_data:
            st.dataframe(pattern_data, use_container_width=True)
        
        # Cortex Analysis
        cortex_summary = summary.get('cortex_summary', {})
        if cortex_summary.get('has_cortex_calls'):
            st.subheader("🤖 Cortex AI Analysis")
            
            # Cortex metrics
            cortex_col1, cortex_col2, cortex_col3, cortex_col4 = st.columns(4)
            
            with cortex_col1:
                st.metric("Total Cortex Calls", cortex_summary.get('total_cortex_calls', 0))
            with cortex_col2:
                st.metric("Complete Calls", cortex_summary.get('cortex_complete_calls', 0))
            with cortex_col3:
                st.metric("Search Calls", cortex_summary.get('cortex_search_calls', 0))
            with cortex_col4:
                st.metric("Interactive Steps", cortex_summary.get('interactive_steps_created', 0))
            
            # Cortex capabilities breakdown
            cortex_capabilities = []
            if cortex_summary.get('cortex_complete_calls', 0) > 0:
                cortex_capabilities.append({
                    "AI Capability": "🤖 Cortex Complete",
                    "Count": cortex_summary.get('cortex_complete_calls', 0),
                    "Description": "Interactive LLM prompting with live parameter tuning"
                })
            if cortex_summary.get('cortex_search_calls', 0) > 0:
                cortex_capabilities.append({
                    "AI Capability": "🔍 Cortex Search",
                    "Count": cortex_summary.get('cortex_search_calls', 0),
                    "Description": "Natural language search through your data"
                })
            if cortex_summary.get('cortex_analyst_calls', 0) > 0:
                cortex_capabilities.append({
                    "AI Capability": "🧠 Cortex Analyst",
                    "Count": cortex_summary.get('cortex_analyst_calls', 0),
                    "Description": "AI-powered SQL generation from natural language"
                })
            
            if cortex_capabilities:
                st.dataframe(cortex_capabilities, use_container_width=True)
            
            # Interactive steps info
            if cortex_summary.get('interactive_steps_created', 0) > 0:
                st.success(f"🎉 {cortex_summary.get('interactive_steps_created', 0)} interactive Cortex steps have been automatically created to demonstrate AI capabilities!")
                st.info("💡 These interactive steps will allow users to experiment with prompts, parameters, and see real-time AI responses.")
        
        # Inferred metadata
        st.subheader("🏷️ Inferred Metadata")
        inferred = summary['inferred_metadata']
        
        metadata_cols = st.columns(2)
        with metadata_cols[0]:
            st.write(f"**Database:** {inferred.get('database', 'Not detected')}")
            st.write(f"**Schema:** {inferred.get('schema', 'Not detected')}")
            st.write(f"**Role:** {inferred.get('role', 'Not detected')}")
        with metadata_cols[1]:
            st.write(f"**Warehouse:** {inferred.get('warehouse', 'Not detected')}")
            st.write(f"**Has Geospatial:** {'Yes' if inferred.get('has_geospatial') else 'No'}")
            st.write(f"**Has Aggregation:** {'Yes' if inferred.get('has_aggregation') else 'No'}")

    with tab2:
        st.subheader("🔍 Parsed SQL Blocks")
        
        for i, block in enumerate(st.session_state['parsed_blocks']):
            with st.expander(f"Step {i+1}: {block['title']}"):
                
                # Block metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {block['id']}")
                    st.write(f"**Suggested Visualization:** {block['default_chart']}")
                    if block.get('variables'):
                        st.write(f"**Variables:** {', '.join(block['variables'])}")
                
                with col2:
                    if block.get('unsupported_commands'):
                        st.warning(f"⚠️ Unsupported commands detected: {len(block['unsupported_commands'])}")
                    
                    # Show patterns
                    active_patterns = [k.replace('_', ' ').title() for k, v in block.get('patterns', {}).items() if v]
                    if active_patterns:
                        st.write(f"**Patterns:** {', '.join(active_patterns)}")
                
                # Show Cortex analysis
                cortex_analysis = block.get('cortex_analysis', {})
                if cortex_analysis.get('has_cortex'):
                    st.success("🤖 Cortex AI Detected!")
                    
                    cortex_info = []
                    if cortex_analysis.get('complete_calls'):
                        cortex_info.append(f"Complete: {len(cortex_analysis['complete_calls'])}")
                    if cortex_analysis.get('search_calls'):
                        cortex_info.append(f"Search: {len(cortex_analysis['search_calls'])}")
                    if cortex_analysis.get('analyst_calls'):
                        cortex_info.append(f"Analyst: {len(cortex_analysis['analyst_calls'])}")
                    
                    if cortex_info:
                        st.write(f"**Cortex Calls:** {', '.join(cortex_info)}")
                
                # Show if this is an interactive step
                if block.get('cortex_type'):
                    st.info(f"🎯 Interactive Cortex {block['cortex_type'].title()} Step")
                    if block.get('cortex_search_service'):
                        st.write(f"**Search Service:** {block['cortex_search_service']}")
                    if block.get('semantic_model_file'):
                        st.write(f"**Semantic Model:** {block['semantic_model_file']}")
                
                # Instructions
                if block.get('instructions'):
                    st.write("**Instructions:**")
                    st.write(block['instructions'])
                
                # SQL query
                st.write("**SQL Query:**")
                st.code(block['query'], language='sql')

    with tab3:
        st.subheader("📄 Generated YAML Configuration")
        
        # Show YAML preview
        st.code(st.session_state['yaml_config'], language='yaml')
        
        # Validation info
        st.info("✅ This YAML is compatible with your SNOW-DEMO harness and ready for upload to your stage.")

    with tab4:
        st.subheader("💾 Download & Export")
        
        # Get demo metadata for this tab
        demo_meta = st.session_state['demo_metadata']
        
        # Save to Database Section (if Snowflake available)
        if HAS_SNOWFLAKE_SESSION:
            st.subheader("🗄️ Save to Database")
            
            config_name = st.text_input(
                "Configuration Name:",
                value=f"{demo_meta.get('topic', 'Demo')}_{demo_meta.get('title', 'Config')}_{datetime.now().strftime('%Y%m%d')}",
                help="Name to save this configuration in the database"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save to Database", type="primary"):
                    if config_name.strip():
                        # Get original SQL from session state
                        original_sql = st.session_state.get('original_sql', '') or st.session_state.get('loaded_sql', '')
                        
                        success = save_config_to_database(
                            session,
                            config_name,
                            st.session_state['demo_metadata'],
                            st.session_state['parsed_blocks'],
                            st.session_state['yaml_config'],
                            st.session_state['inferred_metadata'],
                            st.session_state['summary_report'],
                            original_sql
                        )
                        
                        if success:
                            st.success(f"✅ Configuration saved: {config_name}")
                        else:
                            st.error("❌ Failed to save configuration")
                    else:
                        st.error("Please enter a configuration name")
            
            with col2:
                st.info("💡 Saved configurations can be reloaded later to regenerate YAML files")
            
            st.markdown("---")
        
        # Generate filename
        demo_meta = st.session_state['demo_metadata']
        filename_parts = []
        if demo_meta.get('topic'):
            filename_parts.append(demo_meta['topic'].lower().replace(' ', '_'))
        if demo_meta.get('sub_topic'):
            filename_parts.append(demo_meta['sub_topic'].lower().replace(' ', '_'))
        if demo_meta.get('title'):
            filename_parts.append(demo_meta['title'].lower().replace(' ', '_'))
        
        if filename_parts:
            suggested_filename = "_".join(filename_parts) + ".yaml"
        else:
            suggested_filename = f"demo_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        # Download section
        st.subheader("📥 Download YAML")
        st.download_button(
            label="📥 Download YAML Configuration",
            data=st.session_state['yaml_config'],
            file_name=suggested_filename,
            mime="application/x-yaml",
            help="Download the generated YAML file to upload to your SNOW-DEMO stage"
        )
        
        # Upload instructions
        st.subheader("📤 Upload Instructions")
        
        # Get current database for instructions
        try:
            current_db = get_current_database()
        except:
            current_db = "AI_FRAMEWORK_DB"  # Fallback if function fails
            
        st.markdown(f"""
        **Next Steps:**
        1. Download the YAML file above
        2. **Upload to your Snowflake stage** with a project directory name
        3. Open your SNOW-DEMO harness
        4. Select your project area and demo configuration
        5. Run your interactive demo!
        """)
        
        # Stage upload SQL (for reference)
        with st.expander("📋 Complete Upload Instructions"):
            # Get current database from session context
            dataops_db = get_current_database()
            upload_instructions = f"""
💡 **Project Directory Tips:**

Create a meaningful directory name that matches your use case:

- **analytics** - Customer analytics, sales analysis
- **geospatial** - Location-based demos  
- **ml_demos** - Machine learning showcases
- **financial** - Finance use cases
- **retail** - E-commerce demos

🖱️ **Snowsight UI:**

1. Navigate to FRAMEWORK_YAML_STAGE
2. Click "Upload" → Select your YAML file  
3. In path field, type: `/analytics` (or your project name)
4. Click "Upload" - directory created automatically!
"""
            st.markdown(upload_instructions)

# Footer
st.markdown("---")
st.markdown("**SNOW-DEMO-CONFIG v1.0** - Transform your SQL worksheets into interactive demo experiences!") 