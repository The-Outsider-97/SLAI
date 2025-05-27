CREATE TABLE IF NOT EXISTS evaluation_issues (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    issue_type VARCHAR(50) NOT NULL,
    severity FLOAT CHECK (severity >= 0 AND severity <= 1),
    context JSONB,
    metrics JSONB,
    resolution_status VARCHAR(20) DEFAULT 'unresolved'
);

CREATE INDEX idx_issue_type ON evaluation_issues(issue_type);
CREATE INDEX idx_timestamp ON evaluation_issues(timestamp);
