from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool
from detectors import run_all_detectors
from storage import EvaluationStore

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Connect to the Claude model
llm = ChatAnthropic(model="claude-sonnet-4-6")

# Define the expected final response structure
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Instructions given to the research agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.

            Wrap the final output in this format and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools available to the agent
tools = [search_tool, wiki_tool, save_tool]

# Create the tool-calling agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

# Run the agent and keep all tool execution details
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)

# Get the research question
query = input("What can I help you research? ")

# Run the agent (catch tool crashes so one bad tool call doesn't kill everything)
import time

print("\n Agent is researching (this takes a moment)...", flush=True)
start_time = time.time()
try:
    raw_response = agent_executor.invoke({"query": query})
except Exception as e:
    print(f"\n⚠️  Agent run crashed: {e}")
    print("This usually means a tool (like Wikipedia) failed. Try running again.")
    raise SystemExit
print("Agent finished. Analyzing...", flush=True)

end_time = time.time()
latency_ms = int((end_time - start_time) * 1000)

# Extract token usage — dedupe by message id since one Claude turn
# can appear across multiple tool-call steps in intermediate_steps
seen_message_ids = set()
total_input_tokens = 0
total_output_tokens = 0

for action, result in raw_response.get("intermediate_steps", []):
    msg = action.message_log[0] if action.message_log else None
    if msg is None:
        continue
    msg_id = getattr(msg, "id", None)
    usage = getattr(msg, "usage_metadata", None)
    if msg_id and usage and msg_id not in seen_message_ids:
        seen_message_ids.add(msg_id)
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

# Claude sonnet-4-6 pricing: $3/M input tokens, $15/M output tokens
cost_usd = (total_input_tokens * 3 / 1_000_000) + (total_output_tokens * 15 / 1_000_000)


# Show each tool call cleanly
print("\n========== TOOL EXECUTION STEPS ==========")

steps = raw_response.get("intermediate_steps", [])

if not steps:
    print("The agent did not use any tools.")

for i, (action, result) in enumerate(steps, start=1):
    result_text = " ".join(str(result).split())
    if len(result_text) > 200:
        result_text = result_text[:200] + "..."

    print(f"\nStep {i} | Tool: {action.tool}")
    print(f"  Input:  {action.tool_input}")
    print(f"  Output: {result_text}")


# Extract text from the Anthropic response
output = raw_response.get("output")
if isinstance(output, list):
    output_text = "".join(
        block.get("text", "") for block in output if isinstance(block, dict)
    )
else:
    output_text = output


# Try to parse the final output (and record whether it worked)
try:
    structured_response = parser.parse(output_text)
    parse_succeeded = True

    print("\n========== STRUCTURED RESPONSE ==========")
    print("Topic:", structured_response.topic)
    print("Summary:", structured_response.summary)

    print("\nSources:")
    for source in structured_response.sources:
        print("-", source)

    print("\nTools used:")
    for tool in structured_response.tools_used:
        print("-", tool)

except Exception as error:
    parse_succeeded = False

    print("\n========== PARSING ERROR ==========")
    print("Error:", error)
    print("Output that could not be parsed:")
    print(output_text[:500] + "..." if len(output_text) > 500 else output_text)

# Build the step list
steps_with_output = [(action.tool, action.tool_input, str(result)) for action, result in steps]

# Run detectors + judge + path analysis
print("⚖️  Running evaluation (judge is scoring the run)...", flush=True)
issues, evaluation, path = run_all_detectors(query, steps_with_output, parse_succeeded, output_text)
print("✅ Evaluation complete.\n", flush=True)

# ---- 1. DETECTED ISSUES ----
print("\n========== DETECTED ISSUES ==========")
if not issues:
    print("No rule-based issues detected.")
else:
    for issue in issues:
        print(f"- {issue}")

# ---- 2. EVALUATION SCORECARD ----
print("\n========== EVALUATION SCORECARD ==========")
if "error" in evaluation:
    print("Judge evaluation failed:", evaluation["error"])
    print("Raw judge output:", evaluation.get("raw", "")[:300])
else:
    dimensions = ["correctness", "faithfulness", "relevance", "tool_selection", "efficiency"]
    for dim in dimensions:
        d = evaluation.get(dim, {})
        score = d.get("score", "?")
        reason = d.get("reason", "")
        bar = "█" * int(score) + "░" * (5 - int(score)) if isinstance(score, int) else ""
        print(f"{dim.replace('_', ' ').title():<16} {score}/5  {bar}")
        print(f"                 └─ {reason}")
    print(f"\nOVERALL VERDICT: {evaluation.get('overall_verdict', '?')}")

    # ---- 3. AGENT PROCESS ANALYSIS ----
    print("\n========== AGENT PROCESS ANALYSIS ==========")
    print(evaluation.get("process_analysis", "No analysis available."))

# ---- 4. PATH ANALYSIS (measured facts) ----
print("\n========== PATH ANALYSIS ==========")
print(f"Total steps:      {path['total_steps']}")

tool_summary = ", ".join(f"{tool} ×{count}" for tool, count in path["tool_counts"].items())
print(f"Tool usage:       {tool_summary}")

if path["duplicate_count"] > 0:
    dup_list = ", ".join(f"step {i} ({tool})" for i, tool in path["duplicates"])
    print(f"Duplicate calls:  {path['duplicate_count']}  ⚠️  ({dup_list})")
else:
    print("Duplicate calls:  0")

if path["over_reliance_tool"]:
    print(f"Over-reliance:    heavily used '{path['over_reliance_tool']}'  ⚠️")
else:
    print("Over-reliance:    balanced tool usage")

# ---- SAVE EVERYTHING TO POSTGRES ----
print("\n💾 Saving to database...", flush=True)

with EvaluationStore() as store:
    agent_id = store.upsert_agent(
        agent_key="langchain_research_agent",
        agent_name="LangChain Research Agent",
        agent_type="research",
        framework="LangChain",
        model_name="claude-sonnet-4-6",
        version="v1",
        configuration={"tools": ["search", "wikipedia", "save_text_to_file"]},
    )

    task_id = store.create_task(
        task_type="research_query",
        input_text=query,
        source_name="manual",
    )

    import os

    def rename_saved_file(query, task_id):
        files = sorted(
            [f for f in os.listdir(".") if f.startswith("research_") and f.endswith(".txt")],
            key=os.path.getmtime,
            reverse=True
        )
        if not files:
            return None
        latest_file = files[0]
        words = query.lower().replace("?", "").split()
        stopwords = {"of", "the", "a", "an", "is", "are", "what", "why", "how", "does", "do"}
        slug = "_".join(w for w in words if w not in stopwords)[:50]
        new_name = f"research_{slug}__task{task_id}.txt"
        os.rename(latest_file, new_name)
        return new_name

    saved_filename = rename_saved_file(query, task_id)
    if saved_filename:
        print(f"📁 Saved research file renamed to: {saved_filename}")

    run_id = store.start_run(agent_id=agent_id, task_id=task_id)

    for i, (tool, tool_input, tool_output) in enumerate(steps_with_output, start=1):
        store.add_event(
            run_id=run_id,
            sequence_number=i,
            actor="agent",
            event_type="tool_call",
            tool_name=tool,
            tool_input=tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)},
            tool_output={"result": tool_output[:2000]},
            success=True,
        )

    if "error" not in evaluation:
        for dim in ["correctness", "faithfulness", "relevance", "tool_selection", "efficiency"]:
            d = evaluation.get(dim, {})
            if "score" in d:
                store.add_metric(
                    run_id=run_id,
                    metric_name=dim,
                    value=d["score"],
                    metric_group="judge_score",
                    scale_min=1,
                    scale_max=5,
                    explanation=d.get("reason", ""),
                    evaluator_name="llm_judge",
                    evaluator_version="v1",
                )

    store.add_metric(run_id=run_id, metric_name="total_steps", value=path["total_steps"], metric_group="path_analysis")
    store.add_metric(run_id=run_id, metric_name="duplicate_count", value=path["duplicate_count"], metric_group="path_analysis")

    for issue in issues:
        store.add_issue(
            run_id=run_id,
            issue_type=issue.split(":")[0].strip(),
            description=issue,
            detected_by="rule_based",
        )

    if "error" not in evaluation:
        scores = [evaluation[d]["score"] for d in ["correctness", "faithfulness", "relevance", "tool_selection", "efficiency"] if d in evaluation]
        avg_score = sum(scores) / len(scores) if scores else None
    else:
        avg_score = None

    store.complete_run(
        run_id=run_id,
        status="completed",
        verdict=evaluation.get("overall_verdict", "UNKNOWN"),
        primary_score=avg_score,
        final_output=output_text[:2000],
        latency_ms=latency_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=cost_usd,
        parse_succeeded=parse_succeeded,
        raw_result={"num_issues": len(issues)},
)

print(f"✅ Saved to database. Run ID: {run_id}")