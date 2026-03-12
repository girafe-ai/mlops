-- =============================================================
-- 01. Basic queries: SELECT, WHERE, ORDER BY, LIMIT
-- =============================================================


-- ---- SELECT ------------------------------------------------

-- all columns
SELECT * FROM users LIMIT 5;

-- specific columns with aliases
SELECT
    username   AS name,
    email      AS contact,
    country    AS geo
FROM users
LIMIT 5;


-- ---- WHERE -------------------------------------------------

-- simple comparison
SELECT * FROM user_logs WHERE status_code = 404;

-- AND / OR
SELECT * FROM user_logs
WHERE status_code = 200
  AND response_time_ms > 1500;

SELECT * FROM user_logs
WHERE status_code = 500
   OR status_code = 404;

-- IN
SELECT * FROM users
WHERE country IN ('US', 'DE', 'FR');

-- BETWEEN
SELECT * FROM user_logs
WHERE response_time_ms BETWEEN 100 AND 300;

-- date range
SELECT * FROM user_logs
WHERE created_at BETWEEN now() - interval '7 days' AND now();

-- LIKE / ILIKE (pattern matching)
SELECT * FROM pages WHERE path LIKE '/blog%';
SELECT * FROM users WHERE email ILIKE '%EXAMPLE%';


-- ---- ORDER BY ----------------------------------------------

-- ascending (default)
SELECT * FROM user_logs ORDER BY response_time_ms ASC LIMIT 10;

-- descending
SELECT * FROM user_logs ORDER BY response_time_ms DESC LIMIT 10;

-- multiple columns
SELECT * FROM user_logs
ORDER BY status_code DESC, response_time_ms ASC
LIMIT 20;


-- ---- LIMIT & OFFSET (pagination) --------------------------

-- first page (10 rows)
SELECT * FROM user_logs ORDER BY id LIMIT 10 OFFSET 0;

-- second page
SELECT * FROM user_logs ORDER BY id LIMIT 10 OFFSET 10;


-- ---- DISTINCT ----------------------------------------------

-- unique actions
SELECT DISTINCT action FROM user_logs;

-- unique country list
SELECT DISTINCT country FROM users ORDER BY country;

-- unique (action, status_code) combinations
SELECT DISTINCT action, status_code
FROM user_logs
ORDER BY action, status_code;


-- ---- NULL handling -----------------------------------------

-- PostgreSQL doesn't generate NULLs in our data, but the syntax is:
-- SELECT * FROM users WHERE some_column IS NULL;
-- SELECT * FROM users WHERE some_column IS NOT NULL;

-- COALESCE: fallback for NULLs
SELECT
    id,
    COALESCE(ip_address::text, 'unknown') AS ip
FROM user_logs
LIMIT 5;
