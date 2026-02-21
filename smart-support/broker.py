# broker.py
from redis import Redis
from rq import Queue

# Connect to local Redis (Codespaces auto-starts it)
redis_conn = Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Create queue (default name is 'default')
ticket_queue = Queue(connection=redis_conn, name='tickets')