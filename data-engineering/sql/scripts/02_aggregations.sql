-- =============================================================
-- 02. Aggregations: GROUP BY, HAVING, COUNT, AVG, SUM
-- =============================================================


-- ---- Simple aggregates (whole table) -----------------------

SELECT count(*)               AS total_logs        FROM user_logs;
SELECT count(DISTINCT user_id) AS unique_users      FROM user_logs;
SELECT avg(response_time_ms)  AS avg_response       FROM user_logs;
SELECT min(response_time_ms)  AS fastest,
       max(response_time_ms)  AS slowest            FROM user_logs;


-- ---- GROUP BY single column --------------------------------

-- how many logs per action type
SELECT
    action,
    count(*) AS cnt
FROM user_logs
GROUP BY action
ORDER BY cnt DESC;

-- average response time per status code
SELECT
    status_code,
    round(avg(response_time_ms)) AS avg_ms,
    count(*)                     AS cnt
FROM user_logs
GROUP BY status_code
ORDER BY status_code;


-- ---- GROUP BY multiple columns -----------------------------

-- requests breakdown: action x status_code
SELECT
    action,
    status_code,
    count(*) AS cnt
FROM user_logs
GROUP BY action, status_code
ORDER BY action, status_code;


-- ---- HAVING (filter on aggregated values) ------------------

-- users who generated more than 80 log entries
SELECT
    user_id,
    count(*) AS total_requests
FROM user_logs
GROUP BY user_id
HAVING count(*) > 80
ORDER BY total_requests DESC;

-- pages with average response time above 1 second
SELECT
    page_id,
    round(avg(response_time_ms)) AS avg_ms
FROM user_logs
GROUP BY page_id
HAVING avg(response_time_ms) > 1000
ORDER BY avg_ms DESC;


-- ---- Practical examples ------------------------------------

-- top-10 most active users
SELECT
    user_id,
    count(*) AS requests
FROM user_logs
GROUP BY user_id
ORDER BY requests DESC
LIMIT 10;

-- busiest hours of the day
SELECT
    extract(hour FROM created_at) AS hour_of_day,
    count(*)                      AS requests
FROM user_logs
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- daily request volume over the last 30 days
SELECT
    created_at::date AS day,
    count(*)         AS requests
FROM user_logs
WHERE created_at >= now() - interval '30 days'
GROUP BY day
ORDER BY day;

-- error rate per day (% of non-200 responses)
SELECT
    created_at::date AS day,
    count(*)         AS total,
    count(*) FILTER (WHERE status_code >= 400) AS errors,
    round(
        100.0 * count(*) FILTER (WHERE status_code >= 400) / count(*), 2
    ) AS error_rate_pct
FROM user_logs
GROUP BY day
ORDER BY day;
