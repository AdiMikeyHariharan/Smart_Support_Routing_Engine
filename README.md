# Smart_Support_Routing_Engine

**Categorizes** tickets (Billing / Technical / Legal)  
**Scores urgency** with transformers & sentiment  
**Detects ticket storms** via semantic deduplication  
**Routes** to agents with skill matching & circuit breakers

Built for the **System Design & NLP Track** hackathon (48-hour challenge)

Deployment Link : https://smart-support-routing-engine-dgtd.onrender.com/


## Implemented Milestones

| Milestone | Description | Status | Key Features |
|-----------|-------------|--------|--------------|
| **1**     | Minimum Viable Router (MVR) | ✅ Complete | REST API, keyword-based classification (Billing/Technical/Legal), regex urgency heuristic, in-memory priority queue (heapq) |
| **2**     | Intelligent Queue | ✅ Complete | 202 Accepted + background processing, Transformer classifier, continuous urgency score [0,1], mock webhook for S > 0.8 |
| **3**     | Autonomous Orchestrator | ✅ Complete | Semantic deduplication (sentence embeddings + cosine similarity), circuit breaker (fallback on high latency), skill-based routing with PuLP |

## Demo (Screenshots / GIFs)

(Add screenshots here — use Swagger UI or curl output)

- **POST /ticket_m1** (Milestone 1 – baseline routing)  
  ![m1-demo](https://via.placeholder.com/800x400?text=Milestone+1+Demo)

- **POST /ticket_m2** (Milestone 2 – async + continuous score)  
  ![m2-demo](https://via.placeholder.com/800x400?text=Milestone+2+Demo)

- **GET /queue** (priority ordering)  
  ![queue-demo](https://via.placeholder.com/800x400?text=Queue+with+Urgency+Ordering)

## Quick Start

Requires **Python 3.12.1** (pinned in `.python-version`)

```bash
# 1. Clone & enter repo
git clone https://github.com/AdiMikeyHariharan/Smart_Support_Routing_Engine.git
cd Smart_Support_Routing_Engine

# 2. Activate virtual environment
python -m venv env
source env/bin/activate          # Linux/macOS/Codespaces
# or env\Scripts\activate        # Windows

# 3. Install exact dependencies
cd smart-engine
pip install -r requirements.txt

# 4. Run redis server on one terminal
sudo apt install redis-server -y
redis-server 
redis-server --daemonize yes
rq worker

# 5. Run the server in a different terminal
cd ..
uvicorn smart-engine.main:app --host 0.0.0.0 --port 8000 --reload
```

## Built By
Aditya Hariharan - 22PC04 
  
Jaishree K - 22PC15

Navya V T - 22PC23 

Nidharshana S - 22PC24
