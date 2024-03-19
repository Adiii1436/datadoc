from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from main import get_llm_response  

app = FastAPI()

class ProcessRequest(BaseModel):
    query: str

@app.post("/process_query")
def process_query(request: ProcessRequest):
    result = get_llm_response(request.query)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
