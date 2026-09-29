import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent / 'notes'
MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4:e4b')
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_notes',
            'description': 'Search Markdown notes. All space-separated terms must match. Return file names.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_note',
            'description': 'Read one Markdown note in the allowed notes directory.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        },
    },
]

def search_notes(query):
    terms = query.lower().split()
    if not terms:
        return []
    return [
        p.name for p in sorted(ROOT.glob('*.md'))
        if p.is_file() and not p.is_symlink()
        and all(term in p.read_text(encoding='utf-8').lower() for term in terms)
    ]

def read_note(path):
    if Path(path).name != path or not path.endswith('.md'):
        return 'DENIED: invalid path'
    p = ROOT / path
    if not p.is_file() or p.is_symlink():
        return 'NOT_FOUND'
    return p.read_text(encoding='utf-8')

def chat(messages):
    payload = {'model': MODEL, 'messages': messages, 'tools': TOOLS, 'stream': False, 'think': False, 'options': {'temperature': 0}}
    req = Request('http://127.0.0.1:11434/api/chat', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=240) as resp:
        return json.load(resp)['message']

def run(question):
    messages = [
        {'role': 'system', 'content': 'You answer questions only from local notes. Call search_notes first, then read_note before using a note as evidence. Call one tool at a time. Cite the note file name. If the notes lack the answer, say you cannot find it. Do not invent facts. Answer in Chinese.'},
        {'role': 'user', 'content': question},
    ]
    print('USER', question, flush=True)
    evidence = False
    for step in range(4):
        msg = chat(messages)
        messages.append(msg)
        calls = msg.get('tool_calls') or []
        if not calls:
            print('FINAL', msg.get('content', '') if evidence else '未读取到笔记依据，无法回答。', flush=True)
            return
        if len(calls) != 1:
            print('STOP multiple tool calls', flush=True)
            return
        fn = calls[0]['function']
        name, args = fn['name'], fn['arguments']
        print('CALL', name, json.dumps(args, ensure_ascii=False), flush=True)
        try:
            if name == 'search_notes':
                result = search_notes(args['query'])
            elif name == 'read_note':
                result = read_note(args['path'])
                evidence = evidence or result not in ('NOT_FOUND', 'DENIED: invalid path')
            else:
                result = 'DENIED: unknown tool'
        except (KeyError, TypeError) as exc:
            result = f'ERROR: {type(exc).__name__}'
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, list) else result
        print('RESULT', content, flush=True)
        messages.append({'role': 'tool', 'tool_name': name, 'content': content})
    print('STOP step limit', flush=True)

if __name__ == '__main__':
    run(sys.argv[1])
