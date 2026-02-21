from fastapi import FastAPI
app = FastAPI()

@app.post("/train")
def train(): return {"status": "started"}

@app.get("/metrics")
def metrics(): return {"accuracy": 0.0}
