# tests/send_many_tickets.py
import requests
import time

BASE_URL = "http://localhost:8000"
TICKET_URL = f"{BASE_URL}/ticket"          # M3 endpoint
QUEUE_URL = f"{BASE_URL}/queue"

payloads = [
    {"description": "Hello good morning team"},                     # low urgency
    {"description": "Good morning"},                                # low
    {"description": "How do I reset my password please?"},          # medium
    {"description": "Billing broken urgent fix ASAP!!!"},           # high
    {"description": "System crashed for everyone right now!!!"},    # high
] * 3  # repeat to make 15 tickets

print("Sending tickets...")
job_ids = []
for i, payload in enumerate(payloads, 1):
    r = requests.post(TICKET_URL, json=payload, timeout=10)
    if r.status_code == 202:
        data = r.json()
        jid = data.get("job_id")
        job_ids.append(jid)
        print(f"Sent {i}/{len(payloads)}, job_id={jid}")
    else:
        print(f"Failed to send {i}: {r.status_code} {r.text}")
    time.sleep(0.3)

print("\nCheck queue:")
r = requests.get(QUEUE_URL, timeout=5)
print(r.json())

print("\nWaiting for processing (watch RQ worker logs)...")
time.sleep(10)  # give worker time to finish