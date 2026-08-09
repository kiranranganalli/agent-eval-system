from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool
from storage import EvaluationStore


load_dotenv()


class Claim(BaseModel):
    claim: str
    verdict: str          # "TRUE", "FALSE", or "UNVERIFIABLE"
    evidence: str


class FactCheckResponse(BaseModel):
    original_topic: str
    claims_checked: list[Claim]
    overall_reliability: str   # "high", "medium", or "low"


llm = ChatAnthropic(model="claude-sonnet-4-6")

parser = PydanticOutputParser(pydantic_object=FactCheckResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a fact-checking assistant. You will be given a piece of research
            text that makes several factual claims.

            Your job:
            1. Extract 3-5 specific, checkable factual claims from the text (names, 
               dates, numbers, events — not vague statements).
            2. For EACH claim, use the search tool to independently verify it.
            3. Give each claim a verdict: TRUE (confirmed by search), FALSE 
               (contradicted by search), or UNVERIFIABLE (search didn't give a 
               clear answer either way).
            4. Base your verdict only on what the search tool actually returns — 
               do not rely on your own prior knowledge to decide true/false.
            5. Give an overall_reliability rating (high/medium/low) based on how 
               many claims came back TRUE vs FALSE/UNVERIFIABLE.

            Wrap the final output in this format and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{research_text}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)


# ---- Load the research file to fact-check ----
import os
import re

search_term = input("What topic do you want to fact-check? (e.g. french fries): ")

words = search_term.lower().split()
candidates = [
    f for f in os.listdir(".")
    if f.startswith("research_") and f.endswith(".txt")
    and all(w in f.lower() for w in words)
]

if not candidates:
    print(f"❌ No research file found matching '{search_term}'.")
    raise SystemExit

if len(candidates) > 1:
    print(f"⚠️  Multiple matches found, using the most recent:")
    for c in candidates:
        print(f"   - {c}")
    candidates.sort(key=os.path.getmtime, reverse=True)

filename = candidates[0]

match = re.search(r"__task(\d+)\.txt$", filename)
linked_task_id = int(match.group(1)) if match else None

with open(filename, "r", encoding="utf-8") as f:
    research_text = f.read()

print(f"📄 Loaded research from {filename} ({len(research_text)} characters)")
if linked_task_id:
    print(f"🔗 Linked to task_id: {linked_task_id}\n")
else:
    print("⚠️  No task_id found in filename — this file predates the linking system.\n")


print("🔎 Fact-checker is verifying claims (this takes a moment)...", flush=True)
try:
    raw_response = agent_executor.invoke({"research_text": research_text})
except Exception as e:
    print(f"\n⚠️  Fact-checker crashed: {e}")
    print("This usually means the search tool failed. Try running again.")
    raise SystemExit
print("✅ Fact-check complete.\n", flush=True)

# Show each tool call cleanly
print("\n========== TOOL EXECUTION STEPS ==========")
steps = raw_response.get("intermediate_steps", [])

if not steps:
    print("The fact-checker did not use any tools.")

for i, (action, result) in enumerate(steps, start=1):
    result_text = " ".join(str(result).split())
    if len(result_text) > 200:
        result_text = result_text[:200] + "..."
    print(f"\nStep {i} | Tool: {action.tool}")
    print(f"  Input:  {action.tool_input}")
    print(f"  Output: {result_text}")

# Extract and parse the final output
output = raw_response.get("output")
if isinstance(output, list):
    output_text = "".join(b.get("text", "") for b in output if isinstance(b, dict))
else:
    output_text = output

try:
    structured_response = parser.parse(output_text)

    print("\n========== FACT-CHECK RESULTS ==========")
    print("Topic:", structured_response.original_topic)
    print(f"\nOverall reliability: {structured_response.overall_reliability.upper()}")

    print("\nClaims checked:")
    for i, c in enumerate(structured_response.claims_checked, start=1):
        print(f"\n{i}. Claim: {c.claim}")
        print(f"   Verdict: {c.verdict}")
        print(f"   Evidence: {c.evidence}")

    # ---- SAVE TO POSTGRES (linked to the same task as the research run) ----
    print("\n💾 Saving fact-check to database...", flush=True)

    with EvaluationStore() as store:
        agent_id = store.upsert_agent(
            agent_key="langchain_fact_checker",
            agent_name="LangChain Fact Checker",
            agent_type="fact_check",
            framework="LangChain",
            model_name="claude-sonnet-4-6",
            version="v1",
            configuration={"tools": ["search"]},
        )

        if linked_task_id:
            task_id_to_use = linked_task_id
        else:
            task_id_to_use = store.create_task(
                task_type="fact_check_query",
                input_text=research_text[:500],
                source_name="manual",
            )

        run_id = store.start_run(agent_id=agent_id, task_id=task_id_to_use)

        for i, (action, result) in enumerate(steps, start=1):
            store.add_event(
                run_id=run_id,
                sequence_number=i,
                actor="agent",
                event_type="tool_call",
                tool_name=action.tool,
                tool_input=action.tool_input if isinstance(action.tool_input, dict) else {"input": str(action.tool_input)},
                tool_output={"result": str(result)[:2000]},
                success=True,
            )

        true_count = sum(1 for c in structured_response.claims_checked if c.verdict.upper() == "TRUE")
        false_count = sum(1 for c in structured_response.claims_checked if c.verdict.upper() == "FALSE")
        unverifiable_count = sum(1 for c in structured_response.claims_checked if c.verdict.upper() == "UNVERIFIABLE")

        store.add_metric(
            run_id=run_id, metric_name="claims_true", value=true_count,
            metric_group="fact_check", evaluator_name="fact_checker_agent",
        )
        store.add_metric(
            run_id=run_id, metric_name="claims_false", value=false_count,
            metric_group="fact_check", evaluator_name="fact_checker_agent",
        )
        store.add_metric(
            run_id=run_id, metric_name="claims_unverifiable", value=unverifiable_count,
            metric_group="fact_check", evaluator_name="fact_checker_agent",
        )

        for c in structured_response.claims_checked:
            if c.verdict.upper() in ("FALSE", "UNVERIFIABLE"):
                store.add_issue(
                    run_id=run_id,
                    issue_type=f"claim_{c.verdict.lower()}",
                    description=f"{c.claim} — {c.evidence}",
                    detected_by="fact_checker_agent",
                    severity="medium" if c.verdict.upper() == "FALSE" else "low",
                )

        store.complete_run(
            run_id=run_id,
            status="completed",
            verdict="RELIABLE" if false_count == 0 else "UNRELIABLE",
            primary_score=(true_count / len(structured_response.claims_checked)) * 5 if structured_response.claims_checked else None,
            final_output=output_text[:2000],
            parse_succeeded=True,
            raw_result={"overall_reliability": structured_response.overall_reliability},
        )

    print(f"✅ Saved to database. Run ID: {run_id} (linked to task_id: {task_id_to_use})")

except Exception as error:
    print("\n========== PARSING ERROR ==========")
    print("Error:", error)
    print("Output that could not be parsed:")
    print(output_text[:500] + "..." if len(output_text) > 500 else output_text)