from storage import EvaluationStore


def main() -> None:
    with EvaluationStore() as store:
        agent_id = store.upsert_agent(
            agent_key="langchain_research_agent",
            agent_name="LangChain Research Agent",
            agent_type="research",
            framework="LangChain",
            model_name="claude-sonnet",
            version="v1",
            configuration={
                "tools": [
                    "search",
                    "wikipedia",
                    "save_text_to_file",
                ]
            },
        )

        experiment_id = store.create_experiment(
            experiment_name="Storage Smoke Test",
            benchmark_name="custom_research_eval",
            benchmark_version="v1",
            domain="research",
        )

        task_id = store.create_task(
            task_type="research_query",
            input_text="What were the consequences of World War II?",
            source_name="manual",
            domain="history",
        )

        run_id = store.start_run(
            agent_id=agent_id,
            experiment_id=experiment_id,
            task_id=task_id,
        )

        store.add_event(
            run_id=run_id,
            sequence_number=1,
            actor="user",
            event_type="user_message",
            content_text=(
                "What were the consequences of World War II?"
            ),
        )

        store.add_event(
            run_id=run_id,
            sequence_number=2,
            actor="agent",
            event_type="tool_call",
            tool_name="search",
            tool_input={
                "query": "consequences of World War II"
            },
            success=True,
        )

        scores = {
            "correctness": 5,
            "faithfulness": 4,
            "relevance": 5,
            "tool_selection": 4,
            "efficiency": 3,
        }

        for metric_name, score in scores.items():
            store.add_metric(
                run_id=run_id,
                metric_name=metric_name,
                value=score,
                metric_group="judge_score",
                scale_min=1,
                scale_max=5,
                evaluator_name="llm_judge",
                evaluator_version="v1",
            )

        store.add_metric(
            run_id=run_id,
            metric_name="total_steps",
            value=7,
            metric_group="path_analysis",
        )

        store.complete_run(
            run_id=run_id,
            status="completed",
            verdict="PASS",
            primary_score=4.2,
            final_output=(
                "World War II caused major human, political, "
                "economic, and social changes."
            ),
            parse_succeeded=True,
            raw_result={
                "test_record": True
            },
        )

        print(f"Database test succeeded. Run ID: {run_id}")


if __name__ == "__main__":
    main()
