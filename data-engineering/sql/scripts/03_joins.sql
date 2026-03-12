-- =============================================================
-- 03. JOINs: INNER, LEFT, multi-table
-- =============================================================


-- ---- INNER JOIN (only matching rows) -----------------------

-- logs enriched with username
SELECT
    l.id        AS log_id,
    u.username,
    l.action,
    l.status_code,
    l.created_at
FROM user_logs l
INNER JOIN users u ON u.id = l.user_id
LIMIT 20;

-- logs enriched with page info
SELECT
    l.id AS log_id,
    p.path,
    p.title,
    l.action,
    l.response_time_ms
FROM user_logs l
INNER JOIN pages p ON p.id = l.page_id
LIMIT 20;


-- ---- LEFT JOIN (keep all rows from the left table) ---------

-- find users who have ZERO log entries
SELECT
    u.id,
    u.username,
    u.email
FROM users u
LEFT JOIN user_logs l ON l.user_id = u.id
WHERE l.id IS NULL;

-- every user with their request count (including 0)
SELECT
    u.id,
    u.username,
    count(l.id) AS request_count
FROM users u
LEFT JOIN user_logs l ON l.user_id = u.id
GROUP BY u.id, u.username
ORDER BY request_count ASC
LIMIT 20;


-- ---- Multi-table JOIN --------------------------------------

-- full picture: username + page path + log details
SELECT
    u.username,
    u.country,
    p.path       AS page_path,
    p.section,
    l.action,
    l.status_code,
    l.response_time_ms,
    l.created_at
FROM user_logs l
JOIN users u ON u.id = l.user_id
JOIN pages p ON p.id = l.page_id
ORDER BY l.created_at DESC
LIMIT 20;

-- which countries visit which sections the most
SELECT
    u.country,
    p.section,
    count(*) AS visits
FROM user_logs l
JOIN users u ON u.id = l.user_id
JOIN pages p ON p.id = l.page_id
GROUP BY u.country, p.section
ORDER BY u.country, visits DESC;


-- ---- Self-join example -------------------------------------

-- find pairs of logs from the SAME user that happened
-- within 1 minute of each other (potential bot behavior)
SELECT
    a.id   AS log_a,
    b.id   AS log_b,
    a.user_id,
    a.created_at AS time_a,
    b.created_at AS time_b,
    b.created_at - a.created_at AS gap
FROM user_logs a
JOIN user_logs b
    ON  a.user_id = b.user_id
    AND b.id > a.id
    AND b.created_at - a.created_at < interval '1 minute'
ORDER BY gap ASC
LIMIT 20;


-- ---- CROSS JOIN (cartesian product) ------------------------

-- every possible (action, section) combination
SELECT
    a.action,
    s.section
FROM (SELECT DISTINCT action FROM user_logs)  a
CROSS JOIN (SELECT DISTINCT section FROM pages) s
ORDER BY a.action, s.section;
