import requests
import re
import json

URL = "http://localhost:5067/v1/chat/completions"

tests_data = [
    "اكتب فقرة بالعربية عن الجاذبية",
    "اشرح نظرية النسبية",
    "كيف تعمل الجاذبية؟",
    "اكتب فقرة عن الثقوب السوداء",
    "اشرح كيف يعمل الضوء",
    "ما هي الطاقة المظلمة؟",
    "اشرح ميكانيكا الكم ببساطة",
    "كيف تعمل النجوم؟",
    "ما هو الانفجار العظيم؟",
    "اكتب فقرة عن تكوين المجرات"
]

garbage_pattern = re.compile(r"(UTF8_ERROR|[�ÃÄÂÐĠ]|▁|Ø.|Ù.|Ð.|Â.)")

def has_garbage(text):
    result = bool(garbage_pattern.search(text))
    return result

stats = {"tests": 0, "fail": 0}

for i, prompt in enumerate(tests_data, 1):

    payload = {
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    print(f"\n--- Test {i} (arabic) ---")
    print("Prompt:", prompt)
    print("Reply: ", end="", flush=True)

    full_reply = ""

    try:
        with requests.post(URL, json=payload, stream=True, timeout=120) as r:

            for line in r.iter_lines():

                if line:

                    line = line.decode("utf-8")

                    if line.startswith("data: "):
                        data = line[6:]

                        if data == "[DONE]":
                            break

                        obj = json.loads(data)

                        token = obj["choices"][0]["delta"].get("content", "")

                        print(token, end="", flush=True)

                        full_reply += token

    except Exception as e:
        print("\nRequest failed:", e)
        stats["tests"] += 1
        stats["fail"] += 1
        continue

    print()

    stats["tests"] += 1

    if has_garbage(full_reply):
        print("❌ Garbage detected")
        stats["fail"] += 1
    else:
        print("✅ Clean")

print("\n====================")
print("TEST SUMMARY")

tests = stats["tests"]
fail = stats["fail"]
success = tests - fail

print("ARABIC:")
print("  tests:", tests)
print("  success:", success)
print("  failures:", fail)

print("====================")
