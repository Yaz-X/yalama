import requests
import json

BASE = "http://localhost:5067"


def print_result(name, r):
    print(f"\n{name}")
    print("Status:", r.status_code)
    print("Body:", r.text)


def health():
    r = requests.get(f"{BASE}/health")
    print_result("Health", r)


def model():
    r = requests.get(f"{BASE}/model")
    print_result("Model", r)


def valid_chat():
    payload = {
        "messages": [
            {"role": "user", "content": "Explain gravity in one sentence."}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Valid Chat", r)


def valid_stream():
    payload = {
        "stream": True,
        "messages": [
            {"role": "user", "content": "Explain gravity in one sentence."}
        ]
    }

    print("\nStreaming Test")

    with requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=True
    ) as r:

        print("Status:", r.status_code)

        for line in r.iter_lines():
            if line:
                decoded = line.decode()

                if decoded.startswith("data: "):
                    data = decoded[6:]

                    if data == "[DONE]":
                        print("\n[STREAM DONE]")
                        break

                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"]["content"]
                        print(token, end="", flush=True)
                    except:
                        pass


def invalid_json():
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data="{ invalid json"
    )

    print_result("Invalid JSON", r)


def empty_messages():
    payload = {"messages": []}

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Empty Messages", r)


def missing_messages():
    payload = {"stream": False}

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Missing Messages", r)


def wrong_role():
    payload = {
        "messages": [
            {"role": "banana", "content": "hello"}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Wrong Role", r)


def missing_content():
    payload = {
        "messages": [
            {"role": "user"}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Missing Content", r)


def wrong_types():
    payload = {
        "messages": [
            {"role": 123, "content": ["hello"]}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print_result("Wrong Types", r)


if __name__ == "__main__":
    health()
    model()

    valid_chat()
    valid_stream()

    invalid_json()
    empty_messages()
    missing_messages()
    wrong_role()
    missing_content()
    wrong_types()