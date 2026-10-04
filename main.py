import asyncio, base64, logging, os, sqlite3, time
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ROOT = Path(__file__).resolve().parent.parent
FRONT = ROOT / "frontend"
DB_PATH = ROOT / "data.db"
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER = os.getenv("LINE_USER_ID", "")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
S = {"conf": 0.4, "line_y": 0.5, "invert": False, "max_occupancy": 10}  # settings
state = {"people": 0, "in": 0, "out": 0, "last_frame": 0.0, "fps": 0.0}
last_y: dict[int, float] = {}
viewers: set[WebSocket] = set()
last_alert = 0.0


# ---------- database (events + snapshots only, never per frame) ----------
def db():
    return sqlite3.connect(DB_PATH)

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS detection_events(
            id INTEGER PRIMARY KEY, camera_id INT DEFAULT 1, event_type TEXT,
            person_id INT, confidence REAL, direction TEXT, detected_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS occupancy_snapshots(
            id INTEGER PRIMARY KEY, camera_id INT DEFAULT 1, people_count INT, recorded_at TEXT)""")

def log_event(etype, pid, conf, direction):
    with db() as c:
        c.execute("INSERT INTO detection_events(event_type,person_id,confidence,direction,detected_at) VALUES(?,?,?,?,?)",
                  (etype, pid, conf, direction, datetime.now().isoformat(timespec="seconds")))


# ---------- LINE ----------
def send_line(text: str):
    if not LINE_TOKEN or LINE_TOKEN.startswith("PUT_") or not LINE_USER:
        log.warning("LINE not configured, skip: %s", text)
        return
    try:
        r = requests.post("https://api.line.me/v2/bot/message/push",
                          headers={"Authorization": f"Bearer {LINE_TOKEN}"},
                          json={"to": LINE_USER, "messages": [{"type": "text", "text": text}]}, timeout=8)
        log.info("LINE status %s", r.status_code)
    except Exception as e:
        log.error("LINE error: %s", e)


# ---------- AI: detect + track + count ----------
def process(frame):
    global last_alert
    h, w = frame.shape[:2]
    r = model.track(frame, persist=True, classes=[0], conf=S["conf"],
                    tracker="bytetrack.yaml", verbose=False)[0]
    ly = int(S["line_y"] * h)
    boxes = r.boxes
    state["people"] = len(boxes)
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    for xyxy, pid, conf in zip(boxes.xyxy.tolist(), ids, boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 2)
        cv2.putText(frame, f"#{pid} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120), 1)
        if pid is None:
            continue
        cy = (y1 + y2) / 2
        prev = last_y.get(pid)
        if prev is not None:
            d = "IN" if prev < ly <= cy else "OUT" if prev >= ly > cy else None
            if d:
                if S["invert"]:
                    d = "OUT" if d == "IN" else "IN"
                state["in" if d == "IN" else "out"] += 1
                log_event("crossing", pid, conf, d)
        last_y[pid] = cy
    cv2.line(frame, (0, ly), (w, ly), (0, 160, 255), 2)

    now = time.time()
    if state["people"] > S["max_occupancy"] and now - last_alert > 300:
        last_alert = now
        log_event("alert_high_occupancy", None, None, None)
        send_line(f"🚨 High Occupancy\nPeople: {state['people']}\nTime: {datetime.now():%H:%M}")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode()


async def snapshot_loop():
    while True:
        await asyncio.sleep(10)
        if time.time() - state["last_frame"] < 5:
            with db() as c:
                c.execute("INSERT INTO occupancy_snapshots(people_count,recorded_at) VALUES(?,?)",
                          (state["people"], datetime.now().isoformat(timespec="seconds")))


# ---------- app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=FRONT), name="static")

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(snapshot_loop())

@app.get("/")
async def dashboard(): return FileResponse(FRONT / "dashboard.html")

@app.get("/camera")
async def camera_page(): return FileResponse(FRONT / "camera.html")

@app.get("/api/status")
async def status():
    online = time.time() - state["last_frame"] < 3
    return {"camera": "online" if online else "offline", "ai": "online",
            "people": state["people"], "in": state["in"], "out": state["out"],
            "passed": state["in"] + state["out"], "fps": round(state["fps"], 1)}

@app.get("/api/settings")
async def get_settings(): return S

@app.post("/api/settings")
async def set_settings(new: dict):
    for k in S:
        if k in new: S[k] = type(S[k])(new[k])
    return S

@app.get("/api/stats")
async def stats():
    today = datetime.now().strftime("%Y-%m-%d")
    with db() as c:
        hourly = c.execute("""SELECT substr(recorded_at,12,2), ROUND(AVG(people_count),1), MAX(people_count)
            FROM occupancy_snapshots WHERE recorded_at LIKE ? GROUP BY 1 ORDER BY 1""", (today + "%",)).fetchall()
        events = c.execute("""SELECT detected_at,event_type,person_id,direction,confidence
            FROM detection_events ORDER BY id DESC LIMIT 15""").fetchall()
    return {"hourly": hourly, "events": events}

async def broadcast(msg):
    for v in list(viewers):
        try: await v.send_json(msg)
        except Exception: viewers.discard(v)

@app.websocket("/ws/view")
async def view_ws(ws: WebSocket):
    await ws.accept(); viewers.add(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        viewers.discard(ws)

@app.websocket("/ws/camera")
async def camera_ws(ws: WebSocket):
    await ws.accept()
    log.info("camera connected")
    t_prev = time.time()
    try:
        while True:
            data = await ws.receive_text()
            try:
                raw = base64.b64decode(data.split(",", 1)[-1])
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: raise ValueError("bad frame")
            except Exception as e:
                log.warning("decode error: %s", e)
                await ws.send_json({"status": "error"}); continue
            img = await asyncio.to_thread(process, frame)
            now = time.time()
            state["fps"] = 1 / max(now - t_prev, 1e-3); t_prev = now
            state["last_frame"] = now
            await ws.send_json({"status": "Camera Connected", "people": state["people"]})
            await broadcast({"image": img, **(await status())})
    except WebSocketDisconnect:
        log.info("camera disconnected")
