import requests
import concurrent.futures
import time

URL = "http://localhost:8000/ticket_m2"
TOTAL_REQUESTS = 20
CONCURRENT_WORKERS = 10

def send_request(i):
    payload = {
        "subject": f"Load Test Ticket {i}",
        "description": f"This is concurrent test request number {i}"
    }
    response = requests.post(URL, json=payload)
    return response.status_code

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
    futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
    results = [f.result() for f in futures]

end_time = time.time()

print("Status Codes:", results)
print("Total Time:", round(end_time - start_time, 2), "seconds")