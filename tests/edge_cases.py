import requests
import json
import time

BASE = "http://localhost:5067"


def unicode_arabic():
    payload = {
        "messages": [
            {"role": "user", "content": "إشرح لي الجاذبية بجملة واحدة"}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print("\nUnicode Arabic")
    print(r.text)


def unicode_emoji():
    payload = {
        "messages": [
            {"role": "user", "content": "Explain gravity 🚀🌍"}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print("\nUnicode Emoji")
    print(r.text)


def long_prompt():
    text = "Explain gravity. " * 500

    payload = {
        "messages": [
            {"role": "user", "content": text}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print("\nLong Prompt")
    print("Status:", r.status_code)


def streaming_disconnect():
    payload = {
        "stream": True,
        "messages": [
            {"role": "user", "content": "Explain gravity in detail."}
        ]
    }

    print("\nStreaming Disconnect Test")

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=True
    )

    for line in r.iter_lines():
        if line:
            decoded = line.decode()
            print(decoded)

            # simulate client abort
            r.close()
            print("\nClient terminated connection early.")
            break


def rapid_requests():
    print("\nRapid Requests")

    for i in range(10):
        payload = {
            "messages": [
                {"role": "user", "content": f"Test request {i}"}
            ]
        }

        r = requests.post(
            f"{BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        print(i, r.status_code)


if __name__ == "__main__":
    unicode_arabic()
    unicode_emoji()
    long_prompt()
    streaming_disconnect()
    rapid_requests()