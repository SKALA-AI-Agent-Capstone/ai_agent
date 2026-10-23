import ast

from capstone_agent.instructions import PROMPTS, instruction


def _calls_and_system_messages(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "parse" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "provider":
            yield "parse", node.args[0]
        if isinstance(node, ast.Dict):
            pairs = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
            role = pairs.get("role")
            if isinstance(role, ast.Constant) and role.value == "system" and "content" in pairs:
                yield "system message", pairs["content"]


def test_every_model_prompt_is_loaded_from_a_prompt_file():
    files = {p.stem for p in PROMPTS.glob("*.md") if p.name != "README.md"}
    used = set()
    for module in ("review_engine.py", "checks.py", "evidence.py", "ad_reading.py", "judged_basis.py", "evidence_vlm.py"):
        tree = ast.parse((PROMPTS.parent / module).read_text(encoding="utf-8"))
        for kind, prompt in _calls_and_system_messages(tree):
            if kind == "system message" and isinstance(prompt, ast.Name):
                continue  # Provider.parse forwards the prompt it was given
            assert isinstance(prompt, ast.Call) and getattr(prompt.func, "id", None) == "instruction", \
                f"inline prompt in a {kind} in {module}"
            used.add(prompt.args[0].value)
    assert used == files
    for name in files:
        text = instruction(name)
        assert text and "\n" not in text and "  " not in text
