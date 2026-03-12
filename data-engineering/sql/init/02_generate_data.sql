-- =============================================================
-- Generate ~200 users
-- =============================================================
INSERT INTO users (username, email, created_at, country)
SELECT
    'user_' || i,
    'user_' || i || '@example.com',
    now() - (random() * interval '730 days'),
    (ARRAY['US','UK','DE','FR','RU','CN','JP','BR','IN','CA',
           'AU','ES','IT','NL','SE','KR','MX','AR','PL','TR']
    )[floor(random() * 20 + 1)::int]
FROM generate_series(1, 200) AS i;

-- =============================================================
-- Generate 20 pages
-- =============================================================
INSERT INTO pages (path, title, section) VALUES
    ('/home',          'Home Page',          'main'),
    ('/about',         'About Us',           'main'),
    ('/pricing',       'Pricing Plans',      'main'),
    ('/contact',       'Contact Us',         'main'),
    ('/blog',          'Blog',               'content'),
    ('/blog/post-1',   'Intro to SQL',       'content'),
    ('/blog/post-2',   'Advanced Joins',     'content'),
    ('/blog/post-3',   'Window Functions',   'content'),
    ('/docs',          'Documentation',      'docs'),
    ('/docs/getting-started', 'Getting Started', 'docs'),
    ('/docs/api',      'API Reference',      'docs'),
    ('/docs/faq',      'FAQ',                'docs'),
    ('/login',         'Login',              'auth'),
    ('/register',      'Register',           'auth'),
    ('/profile',       'User Profile',       'account'),
    ('/settings',      'Settings',           'account'),
    ('/dashboard',     'Dashboard',          'account'),
    ('/search',        'Search Results',     'main'),
    ('/cart',          'Shopping Cart',       'shop'),
    ('/checkout',      'Checkout',           'shop');

-- =============================================================
-- Generate ~10 000 user log entries
-- =============================================================
INSERT INTO user_logs (user_id, page_id, action, status_code, response_time_ms, ip_address, created_at)
SELECT
    floor(random() * 200 + 1)::int                          AS user_id,
    floor(random() * 20  + 1)::int                          AS page_id,

    (ARRAY['view','view','view','click','click',
           'submit','scroll','scroll']
    )[floor(random() * 8 + 1)::int]                         AS action,

    -- weighted towards 200
    (ARRAY[200,200,200,200,200,200,
           301,302,404,404,500]
    )[floor(random() * 11 + 1)::int]                        AS status_code,

    floor(random() * 2000 + 10)::int                        AS response_time_ms,

    ('10.' || floor(random() * 255)::int
          || '.' || floor(random() * 255)::int
          || '.' || floor(random() * 254 + 1)::int)::inet   AS ip_address,

    now() - (random() * interval '90 days')                 AS created_at

FROM generate_series(1, 10000);
