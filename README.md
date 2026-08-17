# Agent Eval System

A working observability and evaluation pipeline for AI agents — the same category of problem that Braintrust, Langfuse, and Galileo solve: **capture what an agent did, detect when it silently failed, score it like a human reviewer would, and persist everything for analysis.**

Built end-to-end: a LangChain research agent → a layered detection engine (cheap rules + LLM-as-judge) → a normalized Postgres schema for storing every run, step, score, and issue.

## Why this exists

AI agents fail *silently*. An agent can call the wrong tool, retrieve the wrong information, or claim success while actually failing — and still produce a fluent, confident-looking final answer. A real example this system caught: asked about the consequences of **World War 2**, the agent's Wikipedia tool returned the page for **World War 1** instead. The agent didn't notice. The final answer still read as complete and correct.

Catching failures like this — automatically, in production, before a human has to read every transcript — is the core problem this project solves.

## Architecture

```
Agent runs (LangChain + Claude)
        ↓
Trace Capture — every tool call, input, and output structured cleanly
        ↓
Detection Layer
   ├─ Rule-based (free, instant): schema violations, repeated tool calls
   ├─ Path analysis (free, instant): step count, tool usage, duplicate calls, over-reliance
   └─ LLM-as-Judge (Claude): scores 5 dimensions with written justification
        ↓
Postgres Storage — every run, step, score, and issue persisted for querying
```

The detection layer follows the same division of labor real eval systems use: **cheap deterministic rules catch obvious structural failures for free; an LLM judge is reserved for the failures that require actual understanding** — like noticing a tool result is topically wrong, or that an agent's reasoning process was unsound even when the final answer looks fine.

## What the judge actually evaluates

Every run is scored 1–5 on five dimensions, each with a written justification (not just a number):

| Dimension | What it checks |
|---|---|
| **Correctness** | Is the final answer factually accurate? |
| **Faithfulness** | Does the answer trace back to what the tools actually returned, or was it invented? |
| **Relevance** | Does it actually answer what was asked? |
| **Tool Selection** | Did the agent choose sensible tools for the task? |
| **Efficiency** | Was the path reasonable, or did it wander/repeat? |

The judge also produces a written **process analysis** — a critique of *how* the agent reasoned, not just what it output — and an overall PASS/FAIL verdict.

## Path analysis

Two agents can produce the identical final answer while taking completely different routes to get there — one direct, one wandering through redundant tool calls. The path itself is measured independently of the answer's quality:

- Total steps taken
- Tool usage breakdown
- Duplicate call detection
- Over-reliance on a single tool

This is rule-based, not judged — it's just counting, and it's what makes the "efficiency" score in the scorecard verifiable rather than a guess.

## Storage

Every run is persisted to Postgres under a normalized schema designed to support multiple agents, multiple experiments, and full trace replay:

- **`agents`** — registered agent identities (model, framework, tool config)
- **`experiments`** — optional grouping for batches of runs
- **`tasks`** — the question/input, decoupled from any single run so it can be reused
- **`runs`** — one row per execution: status, verdict, score, timing
- **`events`** — the full step-by-step trace (tool, input, output, sequence)
- **`metrics`** — every judge score and path metric, stored as flexible name/value rows so new metrics never require a schema change
- **`issues`** — every detected problem, tagged by which detector caught it

This shape is what makes the system extensible: querying **average faithfulness score across every run of a given agent**, or **which agent produces the most duplicate tool calls**, is a straightforward join — not a redesign.

## Known limitations (found by using the system, not assumed)

Building this surfaced real calibration problems in the judge — which is itself the point of measuring:

- **Faithfulness leniency.** The judge sometimes scores faithfulness 5/5 even when a step retrieved wrong information, if the agent's *final* answer recovered. It judges outcome more than process.
- **Rules miss semantic duplicates.** The rule-based duplicate detector only catches identical tool inputs. Differently-worded queries that retrieve the same information pass the rule check but get flagged by the judge — proof that rules and judges catch different failure classes, which is why both exist.

These aren't bugs to silently patch — they're the reason ground-truth measurement (comparing judge verdicts against known-correct outcomes) is the planned next phase.

## Judge calibration (ground-truth testing)

**Second finding — order-of-operations blind spot.** Scaling the calibration to 15 tasks brought agreement to 14/15 (93.3%). The one disagreement (task 7) revealed a different kind of judge weakness than the truncation bug: the agent was asked to answer a question ("what's the total cost of my other upcoming flights?") early in the conversation, but instead answered it at the very end — after two other reservations had already been cancelled. The dollar total it gave was correct arithmetic, but calculated from stale, already-changed data, making the final answer wrong. My judge scored the individual response as reasonable in isolation and missed that answering out of the order the user asked deshaped the result. This shows my judge is strong at spot-checking individual answers but weaker at catching an agent that quietly reorders a multi-part request in a way that changes the outcome.

Scoring is only useful if the judge itself can be trusted. To test this, I used [tau2-bench](https://github.com/sierra-research/tau2-bench) — a benchmark with real conversations and a verified pass/fail outcome (checked against actual database state, not opinion).

**Bug found during calibration:** the judge's final-answer input was silently capped at 800 characters. This was invisible on short research answers, but on a full multi-turn customer service conversation, it meant the judge only ever saw the *beginning* of the conversation — never the actual resolution. The judge confidently scored the run as a FAIL, reasoning that it "could not see whether the agent ultimately refused," while the real transcript clearly showed the agent handling it correctly at the end.

**Fix:** raised the limit and replaced the silent cutoff with a visible warning whenever truncation happens, so context loss is never hidden again.

**Result after the fix:** re-run on the same conversation, the judge scored all 5 dimensions correctly and reached the correct PASS verdict — matching tau2-bench's ground truth.

This is a small sample size so far (1 verified case). Scaling this to more tasks to produce a real agreement percentage is the next step (see Roadmap).

**Current result:** After scaling to 24 calibrated tasks across two domains (airline and retail), the judge agrees with tau2-bench's ground truth on 23/24 (95.8%). This number will keep growing as more tasks are added — every run of `calibrate_judge.py` automatically finds new tau2-bench results and adds them to this total.

## Stack

- **Agent framework:** LangChain (tool-calling agent) + Claude (Anthropic API)
- **Detection:** Python rule-based detectors + Claude as LLM-judge
- **Storage:** PostgreSQL, `psycopg`, normalized 7-table schema
- **Tools given to the agent:** web search, Wikipedia, file save

## Running it

```bash
pip install -r requirements.txt
psql -h localhost -U eval_user -d evals -f schema_v1.sql   # one-time schema setup
python main.py
```

You'll be prompted for a research question. The agent runs, its trace prints step by step, the detection layer runs (issues → scorecard → process analysis → path analysis), and the full result is saved to Postgres automatically.

## Live dashboard

A Grafana dashboard reads directly from the same Postgres database — no separate backend or API layer needed, since the schema was already designed to support this. It shows:

- **Judge Calibration Accuracy** — live agreement % between the judge and tau2-bench ground truth
- **Total Tasks Calibrated** — running count, grows automatically as more tasks are calibrated
- **Agreement Rate by Domain** — breaks accuracy down by task domain (airline vs. retail), showing the judge performs more consistently on retail's simpler return/exchange logic than on airline's more complex, multi-step policy checks
- **Average Judge Score by Dimension** — average score per judge dimension (correctness, faithfulness, relevance, tool selection, efficiency), useful for spotting whether the judge is systematically harsher on any one dimension
- **Judge Accuracy by Sector**, **Judge Verdict Breakdown**, **Research Agent Path Efficiency**, and **Avg Cost & Latency per Run** — additional panels covering sector-level accuracy, pass/fail distribution, tool-call efficiency, and real per-run cost/token/latency tracking

**Real cost & performance tracking.** Initially, the dashboard's "Avg Cost & Latency per Run" panel would have shown misleading data — the `cost_usd`, `input_tokens`, `output_tokens`, and `latency_ms` columns existed in the schema but were never actually being populated by the agent code. Rather than ship a panel with fake-looking zeros, I went back and added real tracking: wall-clock timing around the agent's `invoke()` call, and token-usage extraction from LangChain's `usage_metadata` (de-duplicated by message ID, since a single Claude turn can span multiple tool-call steps). Cost is computed from Claude Sonnet 4.6's published per-token pricing. This is now a genuinely accurate cost/performance panel, not a placeholder.

All panels query `calibration_results` directly with plain SQL — adding a new dimension or metric to track is a query away, not a code change.

## Roadmap

- [x] Ground-truth measurement (started) — first calibration test against tau2-bench passed after fixing a truncation bug in the judge; scaling to more tasks for a real agreement percentage is in progress
- [ ] Second agent for cross-agent comparison (different model/config, same tasks)
- [ ] Multi-agent pipeline with handoff tracking (orchestrator → specialist agents)
- [x] Live dashboard reading directly from Postgres (Grafana)