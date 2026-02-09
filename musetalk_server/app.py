from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from musetalk_server.conf import conf as settings
from musetalk_server.core.model_loader import model_loader
from musetalk_server.routers import system, avatars, inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    print("Starting MuseTalk Server...")
    print(f"Configuration: {settings.dict()}")
    try:
        # Pre-load models on startup to avoid latency on first request
        # This can be disabled for faster dev startup if needed
        model_loader.load()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models during startup: {e}")
    
    yield
    
    # Shutdown logic (if any)
    print("Shutting down MuseTalk Server...")

app = FastAPI(
    title="MuseTalk API",
    description="Real-time High-Fidelity Audio-Driven Lip-Syncing Server",
    version=settings.version,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system.router, tags=["System"])
app.include_router(avatars.router, tags=["Avatars"])
app.include_router(inference.router, tags=["Inference"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("musetalk_server.app:app", host="0.0.0.0", port=settings.port, reload=False)
