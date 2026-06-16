"""Self-check for the one non-trivial path: turning a messy LLM reply into an
ordered [{speaker, line}] list. Run: .venv/bin/python test_parse.py"""
from app import parse_script

names = ["Ava", "Marcus"]

# clean JSON array
a = parse_script('[{"speaker":"Ava","line":"Hi"},{"speaker":"Marcus","line":"Hey"}]', names)
assert a == [{"speaker": "Ava", "line": "Hi", "emotion": "neutral"},
             {"speaker": "Marcus", "line": "Hey", "emotion": "neutral"}], a

# fenced + dict wrapper + "text" key + empty line dropped + unknown speaker round-robins
b = parse_script('```json\n{"chat":[{"speaker":"???","text":"Yo"},{"speaker":"x","line":""}]}\n```', names)
assert b == [{"speaker": "Ava", "line": "Yo", "emotion": "neutral"}], b

# "(pauses)" stage direction becomes an ellipsis, not read-aloud text
c = parse_script('[{"speaker":"Ava","line":"that\'s when... (pauses) ...bad things happen"}]', names)
assert c == [{"speaker": "Ava", "line": "that's when... ... ...bad things happen",
              "emotion": "neutral"}], c

print("parse_script OK")
