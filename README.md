# AI Surveillance (MVP)
1. python3 -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. cp .env.example .env   (then put your LINE token + user id)
4. cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
5. New Terminal tab: brew install cloudflared && cloudflared tunnel --url http://localhost:8000
   -> copy the https://xxxx.trycloudflare.com URL
6. Mac browser: <URL>/        (dashboard)
   Phone browser: <URL>/camera (press Start Camera)
