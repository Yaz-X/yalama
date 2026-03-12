import requests
import json

BASE = "http://localhost:5067"


def build_conversation(turns=20, words=120):

    large_text = " ".join(["environment"] * words)

    messages = []
    i = 0

    while i < turns:

        messages.append({
            "role": "user",
            "content": f"Question {i}: {large_text}"
        })

        messages.append({
            "role": "assistant",
            "content": f"Answer {i}: {large_text}"
        })

        i += 1

    messages.append({
        "role": "user",
        "content": "Summarize the entire conversation."
    })

    return messages


def stress_chat():

    payload = {
        "messages": build_conversation()
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print("Status:", r.status_code)
    print("Body:", r.text)


if __name__ == "__main__":
    stress_chat()