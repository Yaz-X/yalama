#NOTE: this is only for Instruct models, multi-turn chat wont work with non instruct models and last prompt will be taken

import requests
import json
import re

URL = "http://localhost:5067/v1/chat/completions"

messages = [
    {"role": "user", "content": "hello my name is yalama"},
    {"role": "assistant", "content": "nice to meet you"},
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "i am good"},
    {"role": "user", "content": "what is my name"}
]

garbage_pattern = re.compile(r"(UTF8_ERROR|[�ÃÄÂÐĠ]|▁|Ø.|Ù.|Ð.|Â.)")
template_pattern = re.compile(r"(\[INST\]|\[/INST\]|<s>|</s>)")

payload = {
    "stream": True,
    "messages": messages
}

print("MISTRAL MULTI TURN TEST")
print("Reply:", end=" ", flush=True)

full_reply = ""

with requests.post(URL, json=payload, stream=True) as r:

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

print("\n")

if "yalama" in full_reply.lower():
    print("✅ Memory correct")
else:
    print("❌ Memory failed")

if garbage_pattern.search(full_reply):
    print("❌ Garbage detected")
else:
    print("✅ Clean UTF8")

if template_pattern.search(full_reply):
    print("❌ Template leaked")
else:
    print("✅ Template clean")
