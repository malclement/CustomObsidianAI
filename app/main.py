# main.py
from fastapi import FastAPI

app = FastAPI(
    title="Custom Obsidian API",
    description="A simple FastAPI application to run custom models on Obsidian Clipper.",
    version="0.1.0",
)

# Run the server with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
