"""Compare a fixed route, model plan, and end-of-plan replanning on sample notes."""

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
SOURCE = HERE.parent / "01-tool-use-react" / "notes"
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
QUESTION = "服务 A 的请求超时是多少秒，失败后重试几次？分别写在哪份笔记里？"
FIELDS = ("timeout", "retry")


def sample_notes(case):
    notes = {p.name: p.read_text(encoding="utf-8") for p in SOURCE.glob("*.md")}
    if case == "missing":
        notes.pop("runbook.md")
    elif case == "changed":
        notes["runbook.md"] = notes["runbook.md"].replace("2 次", "3 次")
    return notes


def call_model(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "只输出 JSON。规划本地笔记搜索和读取。不要编造文件内容。"},
                     {"role": "user", "content": prompt}],
        "format": "json", "stream": False, "think": False,
        "options": {"temperature": 0},
    }
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.loads(json.load(response)["message"]["content"])


def valid_plan(data):
    goals = data.get("goals", []) if isinstance(data, dict) else []
    if not isinstance(goals, list):
        return []
    clean = []
    for item in goals[:4]:
        if not isinstance(item, dict):
            continue
        field, query = item.get("field"), item.get("query")
        if field in FIELDS and isinstance(query, str) and query:
            clean.append({"field": field, "query": query[:80]})
    return clean


def plan(notes, history=None):
    instructions = ("为问题列出资料目标。只返回 JSON 对象，格式为 "
                    "{\"goals\":[{\"field\":\"timeout\",\"query\":\"服务 A\"},"
                    "{\"field\":\"retry\",\"query\":\"服务 A\"}]}。"
                    "field 只能是 timeout 或 retry，query 是用于搜索笔记正文的空格分隔词。"
                    "执行器会为每个目标搜索并读取命中文件，你不必规划具体工具动作。")
    if history is None:
        prompt = instructions + "\n问题：" + QUESTION + "\n可见文件名：" + json.dumps(sorted(notes), ensure_ascii=False)
    else:
        prompt = (instructions + "\n只补还没有依据的目标，不重复已成功的读取。最多补 2 个目标。"
                  "\n问题：" + QUESTION + "\n已有轨迹：" + json.dumps(history, ensure_ascii=False))
    return valid_plan(call_model(prompt))


def fact(field, content):
    if field == "timeout":
        match = re.search(r"请求超时设置为\s*(\d+)\s*秒", content)
    else:
        match = re.search(r"重试次数为\s*(\d+)\s*次", content)
    return match.group(1) if match else None


def execute(goals, notes, history, evidence):
    for goal in goals:
        if len(history) >= 12:
            break
        field, query = goal["field"], goal["query"]
        terms = query.lower().split()
        matches = [name for name, content in sorted(notes.items())
                   if terms and all(term in content.lower() for term in terms)]
        history.append({"goal": goal, "tool": "search_notes", "arg": query,
                        "result": matches, "verdict": "pass" if matches else "missing"})
        found = False
        for name in matches:
            if len(history) >= 12:
                break
            content = notes[name]
            value = fact(field, content)
            verdict = "pass" if value else "missing"
            history.append({"goal": goal, "tool": "read_note", "arg": name,
                            "result": content, "verdict": verdict})
            if value:
                evidence[field] = {"value": value, "file": name}
                found = True
                break
        if not found:
            history.append({"goal": goal, "tool": "verify", "arg": field,
                            "result": "本次未取得所需事实", "verdict": "missing"})


def run(case, route):
    notes = sample_notes(case)
    history, evidence = [], {}
    calls = 0
    start = time.monotonic()
    if route == "fixed":
        steps = [{"field": "timeout", "query": "服务 A"},
                 {"field": "retry", "query": "服务 A"}]
    else:
        steps = plan(notes)
        calls += 1
    execute(steps, notes, history, evidence)
    if route == "adaptive" and len(evidence) < 2:
        extra = plan(notes, history)
        calls += 1
        execute(extra[:2], notes, history, evidence)
    elapsed = round(time.monotonic() - start, 3)
    return {"case": case, "route": route, "question": QUESTION,
            "plan": steps, "history": history, "evidence": evidence,
            "missing": [field for field in FIELDS if field not in evidence],
            "model_calls": calls, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    cases = sys.argv[1:] or ["complete", "missing", "changed"]
    for case in cases:
        if case not in ("complete", "missing", "changed"):
            raise SystemExit(f"unknown case: {case}")
        for route in ("fixed", "planned", "adaptive"):
            print(json.dumps(run(case, route), ensure_ascii=False), flush=True)
