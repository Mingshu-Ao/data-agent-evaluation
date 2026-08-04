from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask

REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    video_evidence_instruction = ""
    if (task.context_dir / "video_evidence.md").is_file():
        video_evidence_instruction = (
            " This task contains locally extracted video evidence. Read "
            "`video_evidence.md` with a sufficiently large `max_chars` value before querying "
            "the structured data. Treat OCR as noisy: confirm the configured field, threshold, "
            "time scope, deduplication rule, and grouping dimension across nearby frames."
        )
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        f"When you have the final table, call the `answer` tool.{video_evidence_instruction}\n"
        "Final answer rules: preserve the requested attribute granularity; do not merge two "
        "requested attributes into one column unless the question explicitly asks for a single "
        "combined value. Prefer exact observed source column names and capitalization when a "
        "requested attribute maps directly to a table field. Before answering, verify that each "
        "output row satisfies all filters in the question and that every row has exactly the "
        "same number of values as the selected columns."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2, default=str)
    return f"Observation:\n{rendered}"


def build_answer_budget_prompt(remaining_steps: int) -> str:
    return (
        f"Execution budget warning: only {remaining_steps} model step(s) remain. "
        "Stop exploring and do not call any non-terminal tool. Use the evidence already "
        "collected to submit the best available table with the `answer` tool now. Preserve "
        "observed source field names and capitalization in `columns` when possible."
    )
