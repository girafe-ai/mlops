-- =============================================================
-- 05. DML: INSERT, UPDATE, DELETE
-- =============================================================


-- ---- INSERT single row -------------------------------------

INSERT INTO users (username, email, country)
VALUES ('new_user', 'new_user@example.com', 'US');

-- verify
SELECT * FROM users WHERE username = 'new_user';


-- ---- INSERT multiple rows ----------------------------------

INSERT INTO users (username, email, country) VALUES
    ('batch_user_1', 'batch1@example.com', 'DE'),
    ('batch_user_2', 'batch2@example.com', 'FR'),
    ('batch_user_3', 'batch3@example.com', 'JP');


-- ---- INSERT ... SELECT (copy data from a query) -----------

-- create a log entry for every user who has never had a "submit"
INSERT INTO user_logs (user_id, page_id, action, status_code, response_time_ms, ip_address, created_at)
SELECT
    u.id,
    1,
    'view',
    200,
    100,
    '10.0.0.1'::inet,
    now()
FROM users u
WHERE u.id NOT IN (
    SELECT DISTINCT user_id FROM user_logs WHERE action = 'submit'
);


-- ---- UPDATE with WHERE -------------------------------------

-- fix the country for our test user
UPDATE users
SET country = 'UK'
WHERE username = 'new_user';

SELECT * FROM users WHERE username = 'new_user';


-- ---- UPDATE with subquery ----------------------------------

-- set response_time_ms = 0 for all logs of the most active user
UPDATE user_logs
SET response_time_ms = 0
WHERE user_id = (
    SELECT user_id
    FROM user_logs
    GROUP BY user_id
    ORDER BY count(*) DESC
    LIMIT 1
);


-- ---- UPDATE with JOIN (PostgreSQL syntax) ------------------

-- mark all logs hitting "auth" pages with status 403
UPDATE user_logs
SET status_code = 403
FROM pages p
WHERE user_logs.page_id = p.id
  AND p.section = 'auth'
  AND user_logs.status_code = 404;


-- ---- DELETE with conditions --------------------------------

-- remove all 500-error logs
DELETE FROM user_logs WHERE status_code = 500;

-- remove logs older than 60 days
DELETE FROM user_logs
WHERE created_at < now() - interval '60 days';


-- ---- RETURNING (get back what you changed) -----------------

-- insert and immediately get the assigned id
INSERT INTO users (username, email, country)
VALUES ('returning_demo', 'returning@example.com', 'CA')
RETURNING id, username;

-- delete and see what was removed
DELETE FROM user_logs
WHERE response_time_ms = 0
RETURNING id, user_id, action;


-- ---- UPSERT with ON CONFLICT ------------------------------

-- try to insert a user that already exists -> update country instead
INSERT INTO users (username, email, country)
VALUES ('new_user', 'new_user@example.com', 'DE')
ON CONFLICT (username)
DO UPDATE SET country = EXCLUDED.country;

SELECT * FROM users WHERE username = 'new_user';

-- ON CONFLICT DO NOTHING (silently skip duplicates)
INSERT INTO pages (path, title, section)
VALUES ('/home', 'Home Page', 'main')
ON CONFLICT (path) DO NOTHING;
