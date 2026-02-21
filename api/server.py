from fastapi import FastAPI
app = FastAPI()

@app.post("/train")
def train(): return {"status": "started"}
