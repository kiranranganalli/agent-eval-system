BEGIN;

CREATE TABLE IF NOT EXISTS agents (
    agent_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    agent_key TEXT NOT NULL UNIQUE,
    agent_name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    framework TEXT,
    model_name TEXT,
    version TEXT NOT NULL DEFAULT 'v1',
    configuration JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    benchmark_name TEXT,
    benchmark_version TEXT,
    domain TEXT,
    description TEXT,
    configuration JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_task_id TEXT,
    source_name TEXT,
    source_version TEXT,
    domain TEXT,
    task_type TEXT NOT NULL,
    input_text TEXT,
    input_payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    expected_result JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        input_text IS NOT NULL
        OR input_payload <> '{}'::JSONB
    )
);

CREATE TABLE IF NOT EXISTS runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    agent_id BIGINT NOT NULL
        REFERENCES agents(agent_id)
        ON DELETE RESTRICT,
    experiment_id BIGINT
        REFERENCES experiments(experiment_id)
        ON DELETE SET NULL,
    task_id BIGINT
        REFERENCES tasks(task_id)
        ON DELETE SET NULL,
    trial_number INTEGER NOT NULL DEFAULT 1
        CHECK (trial_number > 0),
    status TEXT NOT NULL DEFAULT 'running',
    verdict TEXT,
    primary_score DOUBLE PRECISION,
    final_output TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    latency_ms BIGINT CHECK (latency_ms IS NULL OR latency_ms >= 0),
    input_tokens BIGINT CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens IS NULL OR output_tokens >= 0),
    cost_usd NUMERIC(14, 8) CHECK (cost_usd IS NULL OR cost_usd >= 0),
    parse_succeeded BOOLEAN,
    error_message TEXT,
    raw_result JSONB NOT NULL DEFAULT '{}'::JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        completed_at IS NULL
        OR completed_at >= started_at
    )
);

CREATE TABLE IF NOT EXISTS events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES runs(run_id)
        ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL
        CHECK (sequence_number > 0),
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    content_text TEXT,
    tool_name TEXT,
    tool_input JSONB,
    tool_output JSONB,
    success BOOLEAN,
    error_message TEXT,
    latency_ms BIGINT CHECK (latency_ms IS NULL OR latency_ms >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES runs(run_id)
        ON DELETE CASCADE,
    event_id BIGINT
        REFERENCES events(event_id)
        ON DELETE SET NULL,
    metric_group TEXT NOT NULL DEFAULT 'evaluation',
    metric_name TEXT NOT NULL,
    numeric_value DOUBLE PRECISION,
    boolean_value BOOLEAN,
    text_value TEXT,
    json_value JSONB,
    unit TEXT,
    scale_min DOUBLE PRECISION,
    scale_max DOUBLE PRECISION,
    explanation TEXT,
    evaluator_name TEXT,
    evaluator_version TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        NUM_NONNULLS(
            numeric_value,
            boolean_value,
            text_value,
            json_value
        ) = 1
    ),
    CHECK (
        scale_min IS NULL
        OR scale_max IS NULL
        OR scale_min <= scale_max
    )
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES runs(run_id)
        ON DELETE CASCADE,
    event_id BIGINT
        REFERENCES events(event_id)
        ON DELETE SET NULL,
    issue_category TEXT,
    issue_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'medium',
    description TEXT NOT NULL,
    detected_by TEXT,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_agent_created
    ON runs (agent_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_experiment
    ON runs (experiment_id);

CREATE INDEX IF NOT EXISTS idx_runs_task
    ON runs (task_id);

CREATE INDEX IF NOT EXISTS idx_events_run_sequence
    ON events (run_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_events_tool_name
    ON events (tool_name)
    WHERE tool_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_metrics_run_name
    ON metrics (run_id, metric_name);

CREATE INDEX IF NOT EXISTS idx_issues_run_severity
    ON issues (run_id, severity);

CREATE INDEX IF NOT EXISTS idx_tasks_external_id
    ON tasks (external_task_id)
    WHERE external_task_id IS NOT NULL;

-- Calibration results: compares your judge's verdict against tau2-bench ground truth
CREATE TABLE calibration_results (
    id SERIAL PRIMARY KEY,
    source_file TEXT UNIQUE NOT NULL,
    task_purpose TEXT,
    ground_truth_reward FLOAT,
    ground_truth_verdict TEXT,
    judge_verdict TEXT,
    agreed BOOLEAN,
    correctness_score INT,
    faithfulness_score INT,
    relevance_score INT,
    tool_selection_score INT,
    efficiency_score INT,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMIT;
