import requests
import json
import time

BASE = "http://localhost:5067"

payload = {
    "messages": [
        {"role": "user", "content": "Explain gravity briefly"}
    ]
}

start = time.time()
duration = 30  # 5 minutes

count = 0
errors = 0

while time.time() - start < duration:
    try:
        r = requests.post(
            f"{BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if r.status_code == 200:
            count += 1
        else:
            errors += 1

    except Exception:
        errors += 1

    time.sleep(0.1)

print("\n--- Endurance Test Finished ---")
print("Successful requests:", count)
print("Errors:", errors)