"""Replay audited assistant turns through the tool-pack router, offline.

The input is the JSONL file that ``scripts/export_assistant_eval_data.py``
writes. For each text turn the script runs the router on the stored trigger,
target and user text. It compares the chosen packs with the tools that the
model called, and counts the prompt and schema characters that the packs save.
It reads only the file. It opens no database and calls no model.

Usage:
    uv run python scripts/export_assistant_eval_data.py --output turns.jsonl
    uv run python scripts/assistant_pack_replay.py turns.jsonl
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from financial_dashboard.services.assistant.contracts import PackName
from financial_dashboard.services.assistant.packs import (
    ALL_PACKS,
    PACKS,
    choose_packs,
    pack_response_schema,
    tool_name,
)
from financial_dashboard.services.assistant.prompt import PromptContext, build_prompt

TEXT_TRIGGERS = frozenset({"ask", "reply"})
HISTORY_TURNS = 12
TOOL_PACK: dict[str, PackName] = {
    tool_name(tool): name for name, pack in PACKS.items() for tool in pack.tools
}


class ReplayTurn(NamedTuple):
    """One audited text turn, as the router sees it.

    Attributes:
        trigger: ``ask`` or ``reply``.
        has_target: Whether the turn had a trusted transaction target.
        user_texts: The user message, then the earlier user turns.
        tools: The tool names that the model called.
        calls: The number of model calls of the turn, without pack requests.
    """

    trigger: str
    has_target: bool
    user_texts: list[str]
    tools: set[str]
    calls: int


class TurnScore(NamedTuple):
    """The router result of one turn.

    Attributes:
        trigger: The turn trigger, with ``+target`` when it had a target.
        missed: The packs that the model needed and the router did not load.
        full_chars: The characters that the turn sends with every pack.
        routed_chars: The characters that the turn sends with the router.
    """

    trigger: str
    missed: frozenset[PackName]
    full_chars: int
    routed_chars: int


def step_chars(packs: frozenset[PackName]) -> int:
    """Count the fixed characters of one model step with these packs.

    Args:
        packs: The loaded packs.

    Returns:
        The instruction characters plus the schema characters.
    """
    return len(build_prompt(PromptContext("", packs=packs))) + len(
        json.dumps(pack_response_schema(packs))
    )


def read_turns(lines: Iterable[str]) -> Iterator[ReplayTurn]:
    """Read the text turns that called the model, oldest first.

    Args:
        lines: The JSONL lines of the audit export.

    Returns:
        An iterator of turns, each with the user turns of its conversation.
    """
    prior: dict[int, list[str]] = defaultdict(list)
    for line in lines:
        row = json.loads(line)
        outputs = row["model_output"]
        conversation = row["conversation_id"]
        earlier = prior[conversation] if conversation is not None else []
        if row["trigger"] in TEXT_TRIGGERS and isinstance(outputs, list) and outputs:
            yield ReplayTurn(
                trigger=row["trigger"],
                has_target=row["transaction_id"] is not None,
                user_texts=[row["user_text"] or "", *earlier[-HISTORY_TURNS:]],
                tools={
                    call["name"]
                    for output in outputs
                    for call in output["payload"].get("calls", [])
                },
                calls=sum(
                    output["payload"].get("outcome") != "request_packs"
                    for output in outputs
                ),
            )
        if row["assistant_text"] is not None and row["user_text"]:
            earlier.append(row["user_text"])


def score(turn: ReplayTurn, *, use_history: bool) -> TurnScore:
    """Route one turn and count what the packs save.

    A missed pack costs one more model call, and the later calls carry it.

    Args:
        turn: The audited turn.
        use_history: Whether the router also reads the earlier user turns.

    Returns:
        The missed packs and the characters with and without the router.
    """
    texts = turn.user_texts if use_history else turn.user_texts[:1]
    chosen = choose_packs(texts, has_target=turn.has_target)
    needed = frozenset(TOOL_PACK[tool] for tool in turn.tools if tool in TOOL_PACK)
    missed = needed - chosen
    routed = turn.calls * step_chars(chosen | needed) + (
        step_chars(chosen) if missed else 0
    )
    return TurnScore(
        f"{turn.trigger}{'+target' if turn.has_target else ''}",
        missed,
        turn.calls * step_chars(ALL_PACKS),
        routed,
    )


def report(turns: list[ReplayTurn], *, use_history: bool) -> None:
    """Print the miss rate and the saved characters per trigger.

    Args:
        turns: The audited turns.
        use_history: Whether the router also reads the earlier user turns.
    """
    groups: dict[str, list[TurnScore]] = defaultdict(list)
    for turn in turns:
        result = score(turn, use_history=use_history)
        groups[result.trigger].append(result)
        groups["all"].append(result)
    print(f"router reads history: {use_history}")
    for trigger, results in sorted(groups.items()):
        misses = [result for result in results if result.missed]
        saved = sum(result.full_chars - result.routed_chars for result in results)
        missed_packs = sorted({name for result in misses for name in result.missed})
        print(
            f"  {trigger:14} turns={len(results):4} misses={len(misses):3} "
            f"({len(misses) / len(results):6.1%}) "
            f"chars/turn full={sum(r.full_chars for r in results) / len(results):8.0f} "
            f"saved={saved / len(results):8.0f} missed_packs={missed_packs}"
        )


def main() -> None:
    """Read the export, then print the step sizes and the replay per trigger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="JSONL from the audit export")
    args = parser.parse_args()
    with args.export.open(encoding="utf-8") as lines:
        turns = list(read_turns(lines))
    for name in ("core", *PACKS):
        packs = frozenset() if name == "core" else frozenset({name})
        print(f"step chars with {name:12} {step_chars(packs):6}")
    print(f"step chars with all packs    {step_chars(ALL_PACKS):6}")
    for use_history in (False, True):
        report(turns, use_history=use_history)


if __name__ == "__main__":
    main()
