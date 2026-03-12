CREATE TABLE users (
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    email      VARCHAR(120) NOT NULL UNIQUE,
    created_at TIMESTAMP    NOT NULL DEFAULT now(),
    country    VARCHAR(3)   NOT NULL
);

CREATE TABLE pages (
    id      SERIAL      PRIMARY KEY,
    path    VARCHAR(120) NOT NULL UNIQUE,
    title   VARCHAR(200) NOT NULL,
    section VARCHAR(50)  NOT NULL
);

CREATE TABLE user_logs (
    id               SERIAL    PRIMARY KEY,
    user_id          INT       NOT NULL REFERENCES users(id),
    page_id          INT       NOT NULL REFERENCES pages(id),
    action           VARCHAR(20) NOT NULL,
    status_code      SMALLINT  NOT NULL,
    response_time_ms INT       NOT NULL,
    ip_address       INET      NOT NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_logs_user_id    ON user_logs(user_id);
CREATE INDEX idx_user_logs_page_id    ON user_logs(page_id);
CREATE INDEX idx_user_logs_created_at ON user_logs(created_at);
CREATE INDEX idx_user_logs_action     ON user_logs(action);
