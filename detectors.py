import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

_judge = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


# ---------- Rule-based detectors (cheap, fast) ----------

def detect_schema_violation(parse_succeeded):
    if not parse_succeeded:
        return "schema_violation: final output did not match the required format"
    return None


def detect_repeated_calls(steps):
    seen = set()
    issues = []
    for tool, inp in steps:
        fp = tool + "::" + json.dumps(inp, sort_keys=True, default=str)
        if fp in seen:
            issues.append(f"repeated_call: '{tool}' called with the same input more than once")
        seen.add(fp)
    return issues


# ---------- Path analysis (rule-based measurement) ----------

def analyze_path(steps_with_output):
    total_steps = len(steps_with_output)

    # tally how many times each tool was used
    tool_counts = {}
    for tool, inp, out in steps_with_output:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    # find duplicate calls (same tool + same input)
    seen = set()
    duplicates = []
    for i, (tool, inp, out) in enumerate(steps_with_output, start=1):
        fp = tool + "::" + json.dumps(inp, sort_keys=True, default=str)
        if fp in seen:
            duplicates.append((i, tool))
        seen.add(fp)

    # detect over-reliance on one tool (>60% of steps)
    most_used = max(tool_counts, key=tool_counts.get) if tool_counts else None
    over_reliance = most_used and total_steps > 0 and (tool_counts[most_used] / total_steps) > 0.6

    return {
        "total_steps": total_steps,
        "tool_counts": tool_counts,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "over_reliance_tool": most_used if over_reliance else None,
    }


# ---------- LLM-as-judge: multi-dimension evaluation ----------

def judge_agent_run(query, steps_with_output, final_answer):
    # build a readable summary of what the agent did
    trajectory = ""
    for i, (tool, inp, output) in enumerate(steps_with_output, start=1):
        snippet = " ".join(str(output).split())[:300]
        trajectory += f"Step {i}: used '{tool}' with input {inp}\n  -> returned: {snippet}\n"

    final_answer_clean = " ".join(str(final_answer).split())

    final_snippet = final_answer_clean

    prompt = f"""You are an expert evaluator monitoring an AI agent, like an LLM-as-judge in a production eval system. Evaluate the agent's run below.

USER'S QUESTION:
"{query}"

WHAT THE AGENT DID (its trajectory):
{trajectory}

THE AGENT'S FINAL ANSWER:
"{final_snippet}"

FAITHFULNESS RULE: Score faithfulness based on whether EVERY specific claim in the final answer
(names, dates, numbers, events) can be traced to something in the trajectory above — not whether
the final answer sounds plausible. If any tool result was clearly off-topic or wrong (e.g.,
returned information about a different subject than asked), faithfulness must be capped at 3/5
even if the final answer recovered by using other sources, because the agent didn't verify or
flag the bad source.

Score each dimension from 1 (very poor) to 5 (excellent). For EACH, give the score and a one-sentence justification. Then give an overall verdict and a short analysis of the agent's thinking process (what it did well and what it did poorly).

Respond in EXACTLY this JSON format and nothing else:
{{
  "correctness": {{"score": <1-5>, "reason": "<why>"}},
  "faithfulness": {{"score": <1-5>, "reason": "<did the answer come from the tool results, or was it made up>"}},
  "relevance": {{"score": <1-5>, "reason": "<did it answer what was asked>"}},
  "tool_selection": {{"score": <1-5>, "reason": "<did it pick appropriate tools>"}},
  "efficiency": {{"score": <1-5>, "reason": "<was the path reasonable, or did it wander/repeat>"}},
  "overall_verdict": "<PASS or FAIL>",
  "process_analysis": "<2-4 sentences: what the agent did well and poorly in its reasoning process>"
}}"""

    response = _judge.invoke(prompt).content.strip()

    # pull the JSON out (judge sometimes wraps it in markdown)
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        return json.loads(response[start:end])
    except Exception as e:
        return {"error": f"could not parse judge response: {e}", "raw": response}


# ---------- Orchestrator ----------

def run_all_detectors(query, steps_with_output, parse_succeeded, final_answer):
    # cheap rule-based issues first
    issues = []
    schema = detect_schema_violation(parse_succeeded)
    if schema:
        issues.append(schema)
    issues.extend(detect_repeated_calls([(t, i) for t, i, o in steps_with_output]))

    # path analysis (rule-based)
    path = analyze_path(steps_with_output)

    # full LLM-judge evaluation
    evaluation = judge_agent_run(query, steps_with_output, final_answer)

    return issues, evaluation, path