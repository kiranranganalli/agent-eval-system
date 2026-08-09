from __future__ import annotations

import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class EvaluationStore:
    def __init__(self, database_url: str | None = None) -> None:
        load_dotenv()

        self.database_url = database_url or os.getenv("DATABASE_URL")

        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is missing. Add it to your .env file."
            )

        self.connection: psycopg.Connection | None = None

    def __enter__(self) -> "EvaluationStore":
        self.connection = psycopg.connect(
            self.database_url,
            row_factory=dict_row,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.connection is None:
            return

        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()
            self.connection = None

    def _get_connection(self) -> psycopg.Connection:
        if self.connection is None:
            raise RuntimeError(
                "EvaluationStore must be used inside a with block."
            )

        return self.connection

    def upsert_agent(
        self,
        *,
        agent_key: str,
        agent_name: str,
        agent_type: str,
        framework: str | None = None,
        model_name: str | None = None,
        version: str = "v1",
        configuration: dict[str, Any] | None = None,
    ) -> int:
        query = """
            INSERT INTO agents (
                agent_key,
                agent_name,
                agent_type,
                framework,
                model_name,
                version,
                configuration
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_key)
            DO UPDATE SET
                agent_name = EXCLUDED.agent_name,
                agent_type = EXCLUDED.agent_type,
                framework = EXCLUDED.framework,
                model_name = EXCLUDED.model_name,
                version = EXCLUDED.version,
                configuration = EXCLUDED.configuration,
                updated_at = CURRENT_TIMESTAMP
            RETURNING agent_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    agent_key,
                    agent_name,
                    agent_type,
                    framework,
                    model_name,
                    version,
                    Jsonb(configuration or {}),
                ),
            )

            return int(cursor.fetchone()["agent_id"])

    def create_experiment(
        self,
        *,
        experiment_name: str,
        benchmark_name: str | None = None,
        benchmark_version: str | None = None,
        domain: str | None = None,
        description: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> int:
        query = """
            INSERT INTO experiments (
                experiment_name,
                benchmark_name,
                benchmark_version,
                domain,
                description,
                configuration
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING experiment_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    experiment_name,
                    benchmark_name,
                    benchmark_version,
                    domain,
                    description,
                    Jsonb(configuration or {}),
                ),
            )

            return int(cursor.fetchone()["experiment_id"])

    def create_task(
        self,
        *,
        task_type: str,
        input_text: str | None = None,
        input_payload: dict[str, Any] | None = None,
        external_task_id: str | None = None,
        source_name: str | None = None,
        source_version: str | None = None,
        domain: str | None = None,
        expected_result: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if input_text is None and not input_payload:
            raise ValueError(
                "A task requires input_text or input_payload."
            )

        query = """
            INSERT INTO tasks (
                external_task_id,
                source_name,
                source_version,
                domain,
                task_type,
                input_text,
                input_payload,
                expected_result,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING task_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    external_task_id,
                    source_name,
                    source_version,
                    domain,
                    task_type,
                    input_text,
                    Jsonb(input_payload or {}),
                    Jsonb(expected_result)
                    if expected_result is not None
                    else None,
                    Jsonb(metadata or {}),
                ),
            )

            return int(cursor.fetchone()["task_id"])

    def start_run(
        self,
        *,
        agent_id: int,
        experiment_id: int | None = None,
        task_id: int | None = None,
        trial_number: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
            INSERT INTO runs (
                agent_id,
                experiment_id,
                task_id,
                trial_number,
                status,
                metadata
            )
            VALUES (%s, %s, %s, %s, 'running', %s)
            RETURNING run_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    agent_id,
                    experiment_id,
                    task_id,
                    trial_number,
                    Jsonb(metadata or {}),
                ),
            )

            return int(cursor.fetchone()["run_id"])

    def add_event(
        self,
        *,
        run_id: int,
        sequence_number: int,
        actor: str,
        event_type: str,
        content_text: str | None = None,
        tool_name: str | None = None,
        tool_input: Any | None = None,
        tool_output: Any | None = None,
        success: bool | None = None,
        error_message: str | None = None,
        latency_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
            INSERT INTO events (
                run_id,
                sequence_number,
                actor,
                event_type,
                content_text,
                tool_name,
                tool_input,
                tool_output,
                success,
                error_message,
                latency_ms,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            RETURNING event_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    run_id,
                    sequence_number,
                    actor,
                    event_type,
                    content_text,
                    tool_name,
                    Jsonb(tool_input)
                    if tool_input is not None
                    else None,
                    Jsonb(tool_output)
                    if tool_output is not None
                    else None,
                    success,
                    error_message,
                    latency_ms,
                    Jsonb(metadata or {}),
                ),
            )

            return int(cursor.fetchone()["event_id"])

    def add_metric(
        self,
        *,
        run_id: int,
        metric_name: str,
        value: Any,
        metric_group: str = "evaluation",
        event_id: int | None = None,
        unit: str | None = None,
        scale_min: float | None = None,
        scale_max: float | None = None,
        explanation: str | None = None,
        evaluator_name: str | None = None,
        evaluator_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        numeric_value = None
        boolean_value = None
        text_value = None
        json_value = None

        if isinstance(value, bool):
            boolean_value = value
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            text_value = value
        else:
            json_value = Jsonb(value)

        query = """
            INSERT INTO metrics (
                run_id,
                event_id,
                metric_group,
                metric_name,
                numeric_value,
                boolean_value,
                text_value,
                json_value,
                unit,
                scale_min,
                scale_max,
                explanation,
                evaluator_name,
                evaluator_version,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            RETURNING metric_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    run_id,
                    event_id,
                    metric_group,
                    metric_name,
                    numeric_value,
                    boolean_value,
                    text_value,
                    json_value,
                    unit,
                    scale_min,
                    scale_max,
                    explanation,
                    evaluator_name,
                    evaluator_version,
                    Jsonb(metadata or {}),
                ),
            )

            return int(cursor.fetchone()["metric_id"])

    def add_issue(
        self,
        *,
        run_id: int,
        issue_type: str,
        description: str,
        issue_category: str | None = None,
        severity: str = "medium",
        event_id: int | None = None,
        detected_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
            INSERT INTO issues (
                run_id,
                event_id,
                issue_category,
                issue_type,
                severity,
                description,
                detected_by,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING issue_id;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    run_id,
                    event_id,
                    issue_category,
                    issue_type,
                    severity,
                    description,
                    detected_by,
                    Jsonb(metadata or {}),
                ),
            )

            return int(cursor.fetchone()["issue_id"])

    def complete_run(
        self,
        *,
        run_id: int,
        status: str,
        verdict: str | None = None,
        primary_score: float | None = None,
        final_output: str | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        parse_succeeded: bool | None = None,
        error_message: str | None = None,
        raw_result: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query = """
            UPDATE runs
            SET
                status = %s,
                verdict = %s,
                primary_score = %s,
                final_output = %s,
                completed_at = CURRENT_TIMESTAMP,
                latency_ms = %s,
                input_tokens = %s,
                output_tokens = %s,
                cost_usd = %s,
                parse_succeeded = %s,
                error_message = %s,
                raw_result = %s,
                metadata = metadata || %s
            WHERE run_id = %s;
        """

        with self._get_connection().cursor() as cursor:
            cursor.execute(
                query,
                (
                    status,
                    verdict,
                    primary_score,
                    final_output,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    parse_succeeded,
                    error_message,
                    Jsonb(raw_result or {}),
                    Jsonb(metadata or {}),
                    run_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(f"Run {run_id} was not found.")
