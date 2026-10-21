"""Claim rules: which provisions and which cross-checks apply to an advertising claim.

The table (claim_rules.json) is versioned in Git so legal reviewers can read and change it.
A rule matches by the claim type the agent assigned or by a pattern in the phrase itself, so a
"clinical" claim with "식약처" in it also picks up the endorsement rule.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    claim_types: tuple[str, ...]
    patterns: tuple[re.Pattern, ...]
    provisions: tuple[dict, ...]
    evidence_rows: dict | None
    checks: tuple[str, ...]
    sweep: bool = False


@dataclass(frozen=True)
class RuleMatch:
    rule: Rule
    matched_by: str
    matched_text: str


@lru_cache(maxsize=1)
def load_rules() -> tuple[str, tuple[Rule, ...]]:
    data = json.loads((Path(__file__).with_name("claim_rules.json")).read_text(encoding="utf-8"))
    rules = tuple(Rule(id=r["id"], label=r["label"], claim_types=tuple(r.get("claim_types", [])),
                       patterns=tuple(re.compile(p) for p in r.get("patterns", [])),
                       provisions=tuple(r.get("provisions", [])), evidence_rows=r.get("evidence_rows"),
                       checks=tuple(r.get("checks", [])), sweep=bool(r.get("sweep"))) for r in data["rules"])
    return data["version"], rules


def match_rules(phrase: str, claim_type: str | None) -> list[RuleMatch]:
    compact = re.sub(r"\s+", "", phrase)
    matches = []
    for rule in load_rules()[1]:
        if claim_type and claim_type in rule.claim_types:
            matches.append(RuleMatch(rule, "claim_type", claim_type))
            continue
        for pattern in rule.patterns:
            found = pattern.search(phrase) or pattern.search(compact)
            if found:
                matches.append(RuleMatch(rule, "pattern", found.group(0)))
                break
    return matches


def section_matches(section: str | None, wanted: str) -> bool:
    """'제13조' matches '제13조 (부당한…)' but not '제13조의2'. '[별표 2] > 1.' matches its continuation chunks."""
    if not section or not section.startswith(wanted):
        return False
    rest = section[len(wanted):]
    return rest == "" or rest[0] in " (>" or wanted[-1] in ".]>"


def item_line(body: str, hint: str) -> tuple[str | None, str | None]:
    """Find the numbered item (호) containing hint. Returns (item label, the exact line)."""
    squeeze = lambda value: re.sub(r"\s+", "", value)
    for line in body.split("\n"):
        if squeeze(hint) in squeeze(line):
            number = re.match(r"^\s*(\d+)\.", line)
            return (f"제1항제{number.group(1)}호" if number else None), line.strip()
    return None, None


def scan_lines(lines: list[str]) -> list[dict]:
    """Lines matching a high-signal rule pattern (sweep=true). Used to catch claims the model did not extract."""
    out = []
    for index, line in enumerate(lines):
        hits = [m for m in match_rules(line, None) if m.rule.sweep]
        if hits:
            out.append({"index": index, "line": line, "rules": [m.rule.id for m in hits],
                        "labels": [m.rule.label for m in hits], "matched": [m.matched_text for m in hits]})
    return out
