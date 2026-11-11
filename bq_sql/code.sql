-- split data 80:20 train:test

CREATE OR REPLACE TABLE `bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw_split` AS
    SELECT 
        *,
        -- Generates a random float between 0.0 and 1.0. 
        -- If it's under 0.20, it becomes TRUE (Test Set).
        (RAND() < 0.20) AS is_test 
    FROM `bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw`;

-- count % target column by data split

SELECT 
    is_test,
    treatment,
    COUNT(*) AS row_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY is_test), 2) AS percentage
FROM bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw_split
GROUP BY is_test, treatment
ORDER BY is_test DESC, treatment;

-- is_test	treatment	row_count	percentage
-- true	    false	    124	        44.44
-- true	    true	    155	        55.56
-- false	false	    498	        50.82
-- false	true	    482	        49.18


CREATE OR REPLACE MODEL `bq-sandbox-491513.mh_tech_survey.mental_health_rf_model`
  TRANSFORM(
        CASE 
            WHEN Age >= 15 AND Age <= 100 THEN Age 
            ELSE NULL 
        END AS age_clean,
        
        CASE 
            WHEN REGEXP_CONTAINS(LOWER(Gender), r'^(m|male|man|cis male|mail|make)$') THEN 'Male'
            WHEN REGEXP_CONTAINS(LOWER(Gender), r'^(f|female|woman|cis female|femail)$') THEN 'Female'
            ELSE 'Other' 
        END AS gender_clean,
        
        COALESCE(work_interfere, 'Unknown') AS work_interfere_clean,
        COALESCE(self_employed, 'No') AS self_employed_clean,
        
        family_history,
        no_employees,
        remote_work,
        tech_company,
        benefits,
        care_options,
        wellness_program,
        seek_help,
        anonymity,
        leave,
        mental_health_consequence,
        phys_health_consequence,
        coworkers,
        supervisor,
        

        treatment,

        is_test)

  OPTIONS (
        model_type = 'LOGISTIC_REG', 
        input_label_cols = ['treatment'],
      
        auto_class_weights = TRUE,
        --num_parallel_tree = 100,
        --tree_method = 'HIST',
        l2_reg = 0.1,
        data_split_method = 'CUSTOM',
        data_split_col = 'is_test',
        enable_global_explain = TRUE)
AS
SELECT
  *
FROM
  `bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw_split`


/*
If you do a SELECT *, do you still exclude columns not present in the TRANSFORM block?
Yes. When a TRANSFORM clause is present, BQML strictly limits the model's features to only the 
columns output by the TRANSFORM block. Any extra columns pulled in by the SELECT * (like comments or state) 
are completely ignored by the model and will not affect training.(Note: The column you specify for 
data_split_col must be present in the underlying SELECT statement, but it does not need to be included in the 
TRANSFORM block. BQML handles it automatically).
*/

/*
Is it better to carry out the transformations of the features in the TRANSFORM block?
Yes, this is the recommended best practice. Embedding transformations inside the TRANSFORM 
clause bakes the preprocessing logic directly into the saved model artifact. When you run ML.PREDICT later 
on new data, you only need to feed it the raw, uncleaned data. The model will automatically execute the regex 
and case statements internally before making the prediction.
*/


/*
Are the transformations calculated only on the training portion of the data?
It depends on the type of transformation:
- Stateless transformations: Row-level logic (like your CASE WHEN or regex statements) is applied independently 
to every row across the entire dataset before the split.
- Stateful transformations: Data-dependent calculations (like scaling with ML.STANDARD_SCALER
 or the automatic mean-imputation BQML applies to the NULL values in your age_clean column) are computed 
 only on the training set. BigQuery learns the statistical mean from the training data and freezes it, 
 then applies that exact same mean to the test set to prevent data leakage.
*/

/*
What if I pass transformed columns in the SELECT statement instead?
If you perform the data cleaning in the SELECT statement at the bottom instead of the TRANSFORM block, 
the model will still train successfully, but the preprocessing logic will not be saved with the model. 
Whenever you use ML.PREDICT or ML.EXPLAIN_PREDICT in the future, you will be forced to rewrite and execute 
the exact same SQL cleaning steps in your prediction query before passing the data to the model. 
This creates a high risk of "training-serving skew," where a slight typo in your future prediction pipeline 
breaks the model.
*/



-- The input data has NULL values in one or more columns: age_clean. 
-- BQML automatically handles null values (See https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-auto-preprocessing#imputation). 
-- If null values represent a special value in the data, replace them with the desired value 
-- before training and then retry.

CREATE OR REPLACE MODEL `bq-sandbox-491513.mh_tech_survey.mental_health_logreg_model_v1`
TRANSFORM(
    CASE
        WHEN Age >= 15 AND Age <= 100 THEN Age
        ELSE NULL
    END AS age_clean,
    COALESCE(work_interfere, 'Unknown') AS work_interfere_clean,

    treatment,
    is_test
)

OPTIONS(
    MODEL_TYPE = 'LOGISTIC_REG',
    INPUT_LABEL_COLS = ['treatment'],
    AUTO_CLASS_WEIGHTS= TRUE,
    L2_REG=0.1,
    DATA_SPLIT_METHOD='CUSTOM',
    DATA_SPLIT_COL='is_test',
    ENABLE_GLOBAL_EXPLAIN=TRUE
    )
AS SELECT * FROM `bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw_split`

;


CREATE OR REPLACE MODEL `bq-sandbox-491513.mh_tech_survey.mental_health_logreg_model_v2`
TRANSFORM(
    CASE
        WHEN Age >= 15 AND Age <= 100 THEN Age
        ELSE NULL
    END AS age_clean,
    CASE
        WHEN TRIM(LOWER(Gender)) IN ('m', 'male', 'man', 'cis male', 'mail', 'make') THEN 'Male'
        WHEN TRIM(LOWER(Gender)) IN ('f', 'female', 'woman', 'cis female', 'femail') THEN 'Female'
        ELSE 'Other'
    END AS gender_clean,
    COALESCE(work_interfere, 'Unknown') AS work_interfere_clean,
        COALESCE(self_employed, 'No') AS self_employed_clean,
        family_history,
        no_employees,
        remote_work,
        tech_company,
        benefits,
        care_options,
        wellness_program,
        seek_help,
        anonymity,
        `leave`,
        mental_health_consequence,
        phys_health_consequence,
        coworkers,
        supervisor,


    treatment,
    is_test
)

OPTIONS(
    MODEL_TYPE = 'LOGISTIC_REG',
    INPUT_LABEL_COLS = ['treatment'],
    AUTO_CLASS_WEIGHTS= TRUE,
    L2_REG=0.1,
    DATA_SPLIT_METHOD='CUSTOM',
    DATA_SPLIT_COL='is_test',
    ENABLE_GLOBAL_EXPLAIN=TRUE
    )
AS SELECT * FROM `bq-sandbox-491513.mh_tech_survey.mh_tech_survey_raw_split`

--- bq ml transform doesn't handle regexp!