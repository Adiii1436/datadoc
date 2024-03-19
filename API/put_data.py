from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from API.store_data import store_data
from main import get_llm_response

app = FastAPI()

class StoreRequest(BaseModel):
    path: str

@app.post("/store_data")
def process_directory(request: StoreRequest):
    store_data(request.path)
    return {"message": "Data stored successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
