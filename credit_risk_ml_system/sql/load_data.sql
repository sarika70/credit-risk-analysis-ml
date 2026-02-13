-- Create the database
CREATE DATABASE IF NOT EXISTS credit_risk_db;
USE credit_risk_db;

-- Create the HMEQ table
CREATE TABLE IF NOT EXISTS hmeq_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    target INT,         -- Originally 'BAD'
    LOAN FLOAT,
    MORTDUE FLOAT,
    VALUE FLOAT,
    REASON VARCHAR(50),
    JOB VARCHAR(50),
    YOJ FLOAT,
    DEROG FLOAT,
    DELINQ FLOAT,
    CLAGE FLOAT,
    NINQ FLOAT,
    CLNO FLOAT,
    DEBTINC FLOAT
);

-- Note: You can import the CSV directly into this table 
-- using the MySQL Import Wizard in Workbench.

-- How to load a CSV file into the table via SQL:
-- 1. Ensure the CSV file is in a directory accessible by MySQL (check 'secure_file_priv' variable)
-- 2. Use the following command (replace the path with your actual file path)

/*
LOAD DATA INFILE 'C:/path/to/your/hmeq.csv' 
INTO TABLE hmeq_data 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(target, LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC);
*/

-- Alternative: Use the MySQL Workbench Import Wizard:
-- 1. Right-click 'hmeq_data' in the Schemas panel.
-- 2. Select 'Table Data Import Wizard'.
-- 3. Follow the prompts to map your CSV columns to the table fields.