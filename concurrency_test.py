import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request():
    response = requests.post("http://127.0.0.1:8000/ticket_m2", json={"description": "Test ticket"})
    return response.status_code

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(send_request) for _ in range(1)]
    for future in as_completed(futures):
        print(f"Response: {future.result()}")