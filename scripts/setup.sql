-- Cortex AI Demo Framework - Initial Setup Script
-- Run this script BEFORE using the notebook
-- This creates the role, grants privileges, then creates databases, schemas, warehouses, and stages

USE ROLE accountadmin;

-- Step 1: Create role for Cortex AI Demo Framework data scientists
CREATE OR REPLACE ROLE cortex_ai_demo_data_scientist;

-- Step 2: Grant system-level privileges to the role
-- Grant Cortex AI privileges (required for AI functions)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE cortex_ai_demo_data_scientist;

-- Grant database creation privilege to the role
GRANT CREATE DATABASE ON ACCOUNT TO ROLE cortex_ai_demo_data_scientist;

-- Grant additional necessary privileges
GRANT CREATE WAREHOUSE ON ACCOUNT TO ROLE cortex_ai_demo_data_scientist;
GRANT APPLY MASKING POLICY ON ACCOUNT TO ROLE cortex_ai_demo_data_scientist;
GRANT APPLY ROW ACCESS POLICY ON ACCOUNT TO ROLE cortex_ai_demo_data_scientist;

-- Grant role to current user
SET my_user_var = (SELECT '"' || CURRENT_USER() || '"');
GRANT ROLE cortex_ai_demo_data_scientist TO USER identifier($my_user_var);

-- Step 3: Create warehouses and grant privileges (keep as ACCOUNTADMIN for stability)
CREATE OR REPLACE WAREHOUSE cortex_ai_demo_wh
    WAREHOUSE_SIZE = 'small'
    WAREHOUSE_TYPE = 'standard'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
COMMENT = 'Main warehouse for Cortex AI Demo Framework';

CREATE OR REPLACE WAREHOUSE cortex_ai_synthetic_data_wh
    WAREHOUSE_SIZE = 'medium'
    WAREHOUSE_TYPE = 'standard'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
COMMENT = 'Warehouse for synthetic data generation';

-- Grant warehouse privileges to cortex_ai_demo_data_scientist
GRANT USAGE ON WAREHOUSE cortex_ai_demo_wh TO ROLE cortex_ai_demo_data_scientist;
GRANT OPERATE ON WAREHOUSE cortex_ai_demo_wh TO ROLE cortex_ai_demo_data_scientist;
GRANT MONITOR ON WAREHOUSE cortex_ai_demo_wh TO ROLE cortex_ai_demo_data_scientist;

GRANT USAGE ON WAREHOUSE cortex_ai_synthetic_data_wh TO ROLE cortex_ai_demo_data_scientist;
GRANT OPERATE ON WAREHOUSE cortex_ai_synthetic_data_wh TO ROLE cortex_ai_demo_data_scientist;
GRANT MONITOR ON WAREHOUSE cortex_ai_synthetic_data_wh TO ROLE cortex_ai_demo_data_scientist;

-- Step 4: Switch to cortex_ai_demo_data_scientist role to create databases as owner
USE ROLE cortex_ai_demo_data_scientist;
USE WAREHOUSE cortex_ai_demo_wh;

-- Verify role switch was successful
SELECT CURRENT_ROLE() AS active_role, CURRENT_USER() AS current_user;

-- Create main database (now owned by cortex_ai_demo_data_scientist)
CREATE OR REPLACE DATABASE AI_FRAMEWORK_DB;

-- Create all schemas in AI_FRAMEWORK_DB
USE DATABASE AI_FRAMEWORK_DB;
CREATE OR REPLACE SCHEMA BRONZE_LAYER
    COMMENT = 'Raw and synthetic data generation layer';
CREATE OR REPLACE SCHEMA SILVER_LAYER
    COMMENT = 'Structured and processed data layer';
CREATE OR REPLACE SCHEMA APPS
    COMMENT = 'All applications including Streamlit apps and dashboards';
CREATE OR REPLACE SCHEMA CONFIGS
    COMMENT = 'Configuration files, YAML templates, and settings';
CREATE OR REPLACE SCHEMA FRAMEWORK
    COMMENT = 'SQL delivery framework and core utilities';
CREATE OR REPLACE SCHEMA UTILITIES
    COMMENT = 'Shared functions, transformations, and helper utilities';

-- Step 5: Create stages and file formats (matching original)
USE SCHEMA CONFIGS;

-- Framework YAML stage for configuration files
CREATE OR REPLACE STAGE AI_FRAMEWORK_DB.CONFIGS.FRAMEWORK_YAML_STAGE
    COMMENT = 'Stage for YAML configuration files';

-- Single stage for all Streamlit applications and environment file
USE SCHEMA APPS;
CREATE OR REPLACE STAGE AI_FRAMEWORK_DB.APPS.AI_FRAMEWORK_APPS
    COMMENT = 'Single stage for all Streamlit applications and environment.yml';

-- File formats (matching original)
CREATE OR REPLACE FILE FORMAT AI_FRAMEWORK_DB.CONFIGS.YAML_CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = '~'
    MULTI_LINE = TRUE
    SKIP_HEADER = 0
    FIELD_OPTIONALLY_ENCLOSED_BY = NONE
    COMMENT = 'File format for YAML configuration files';

CREATE OR REPLACE FILE FORMAT AI_FRAMEWORK_DB.CONFIGS.STANDARD_CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '')
    EMPTY_FIELD_AS_NULL = TRUE
    COMMENT = 'Standard CSV file format';

CREATE OR REPLACE FILE FORMAT AI_FRAMEWORK_DB.CONFIGS.JSON_FORMAT
    TYPE = 'JSON'
    COMPRESSION = 'AUTO'
    COMMENT = 'JSON file format for configuration files';

-- Final verification and status
SELECT CURRENT_ROLE() AS current_role, CURRENT_DATABASE() AS current_database;
SELECT 'Cortex AI Demo Framework setup complete! Now upload the Streamlit files to their respective stages and run the notebook.' AS status;

-- Verify databases are owned by cortex_ai_demo_data_scientist
SHOW DATABASES LIKE 'AI_FRAMEWORK_DB';

-- Instructions for next steps:
-- 1. Upload all files to AI_FRAMEWORK_APPS stage:
--    - 01_ai_framework_synthetic_data_generator.py
--    - 02_ai_framework_structured_tables.py
--    - 03_ai_framework_sql_to_yaml_converter.py
--    - 04_ai_framework_snow_demo.py
--    - 05_ai_framework_snow_viz.py
--    - environment.yml
-- 2. Upload ai_framework_semantic_model.yaml to the FRAMEWORK_YAML_STAGE
-- 3. Download and import cortex_ai_demo_framework_setup.ipynb using Snowsight's Import .ipynb file feature
-- 4. Run the imported notebook to create the Streamlit applications and complete the setup

-- ============================================================================
-- TEARDOWN SCRIPT (Uncomment lines below to clean up all resources)
-- ============================================================================

-- Cortex AI Demo Framework Teardown Script
-- Uncomment and run these lines to remove all objects created during the quickstart

-- USE ROLE ACCOUNTADMIN;

-- USE DATABASE SNOWFLAKE;
-- USE SCHEMA INFORMATION_SCHEMA;

-- DROP DATABASE IF EXISTS AI_FRAMEWORK_DB;
-- DROP ROLE IF EXISTS cortex_ai_demo_data_scientist;

-- DROP WAREHOUSE IF EXISTS cortex_ai_demo_wh;
-- DROP WAREHOUSE IF EXISTS cortex_ai_synthetic_data_wh;
