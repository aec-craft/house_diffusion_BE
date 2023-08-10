from app import app
import uvicorn

if __name__ == "__main__":
    if uvicorn.run(app, host="0.0.0.0", port=8080):
        print("Service started")
        pass
