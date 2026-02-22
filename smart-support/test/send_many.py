# tests/send_many_tickets.py
import requests
import time

URL = "http://localhost:8000/ticket"
JOB_URL = "http://localhost:8000/job/{}"

N = 12
payload = {
    "subject": "Production outage",
    "description": "production system completely down — users cannot login"
}

job_ids = []
for i in range(N):
    r = requests.post(URL, json=payload, timeout=5)
    data = r.json()
    job_ids.append(data.get("job_id"))
    print(f"Sent {i+1}/{N}, job_id={data.get('job_id')}")
    time.sleep(0.5)  # pace requests; adjust to fit 5-minute window

# Poll results
for jid in job_ids:
    if not jid:
        continue
    for _ in range(30):
        r = requests.get(JOB_URL.format(jid), timeout=5)
        j = r.json()
        if j.get("status") not in ("queued", "started"):
            print(f"Job {jid} → result: {j.get('result')}")
            break
        time.sleep(1)