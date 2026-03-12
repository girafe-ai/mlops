-- =============================================================
-- 04. Window functions
-- =============================================================


-- ---- ROW_NUMBER --------------------------------------------

-- number each log entry per user, ordered by time
SELECT
    user_id,
    id AS log_id,
    action,
    created_at,
    row_number() OVER (PARTITION BY user_id ORDER BY created_at) AS rn
FROM user_logs
WHERE user_id <= 5
ORDER BY user_id, rn;

-- pick only the FIRST action of each user (rn = 1)
SELECT * FROM (
    SELECT
        user_id,
        action,
        created_at,
        row_number() OVER (PARTITION BY user_id ORDER BY created_at) AS rn
    FROM user_logs
) sub
WHERE rn = 1
ORDER BY user_id
LIMIT 20;


-- ---- RANK & DENSE_RANK ------------------------------------

-- rank users by total number of requests
SELECT
    user_id,
    count(*)       AS requests,
    rank()       OVER (ORDER BY count(*) DESC) AS rnk,
    dense_rank() OVER (ORDER BY count(*) DESC) AS dense_rnk
FROM user_logs
GROUP BY user_id
ORDER BY rnk
LIMIT 20;


-- ---- LAG / LEAD -------------------------------------------

-- time gap between consecutive requests of the same user
SELECT
    user_id,
    id AS log_id,
    created_at,
    lag(created_at)  OVER w AS prev_time,
    created_at - lag(created_at) OVER w AS gap_since_prev,
    lead(created_at) OVER w AS next_time
FROM user_logs
WHERE user_id = 1
WINDOW w AS (PARTITION BY user_id ORDER BY created_at)
ORDER BY created_at
LIMIT 20;


-- ---- Running totals with SUM() OVER -----------------------

-- cumulative request count per user over time
SELECT
    user_id,
    created_at::date AS day,
    count(*)         AS daily_cnt,
    sum(count(*))    OVER (PARTITION BY user_id ORDER BY created_at::date) AS running_total
FROM user_logs
WHERE user_id <= 3
GROUP BY user_id, day
ORDER BY user_id, day;

-- cumulative response time: shows total "load" a user placed on the server
SELECT
    user_id,
    id AS log_id,
    response_time_ms,
    sum(response_time_ms) OVER (
        PARTITION BY user_id ORDER BY created_at
    ) AS cumulative_ms
FROM user_logs
WHERE user_id = 1
ORDER BY created_at
LIMIT 20;


-- ---- PARTITION BY for per-page analytics -------------------

-- average response time per page alongside each row's own time
SELECT
    l.id AS log_id,
    p.path,
    l.response_time_ms,
    round(avg(l.response_time_ms) OVER (PARTITION BY l.page_id))::int AS page_avg_ms,
    l.response_time_ms - round(avg(l.response_time_ms) OVER (PARTITION BY l.page_id))::int AS diff_from_avg
FROM user_logs l
JOIN pages p ON p.id = l.page_id
ORDER BY l.page_id, l.id
LIMIT 30;


-- ---- NTILE (percentile bucketing) --------------------------

-- split all logs into 4 buckets by response time (quartiles)
SELECT
    id,
    response_time_ms,
    ntile(4) OVER (ORDER BY response_time_ms) AS quartile
FROM user_logs
ORDER BY response_time_ms
LIMIT 30;

-- p50 / p90 / p99 response times (approximate via percentile_cont)
SELECT
    percentile_cont(0.50) WITHIN GROUP (ORDER BY response_time_ms) AS p50,
    percentile_cont(0.90) WITHIN GROUP (ORDER BY response_time_ms) AS p90,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99
FROM user_logs;
