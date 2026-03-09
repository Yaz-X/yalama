import requests
import json
import threading

BASE = "http://localhost:5067"


def worker(i):
    payload = {
        "messages": [
            {"role": "user", "content": f"Explain gravity briefly {i}"}
        ]
    }

    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    print(i, r.status_code)


threads = []

for i in range(50):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("\nLoad test finished")