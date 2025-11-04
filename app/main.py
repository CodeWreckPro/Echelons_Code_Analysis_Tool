from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="EchoLens",
    description="AI-Driven Codebase Intelligence Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Import routers
from app.api.code_evolution import router as code_evolution_router
from app.api.change_storytelling import router as change_storytelling_router
from app.api.hotspot_prediction import router as hotspot_prediction_router
from app.api.refactor_guide import router as refactor_guide_router
from app.api.code_map import router as code_map_router
from app.api.insights import router as insights_router

# Include routers
app.include_router(code_evolution_router, prefix="/api/code-evolution", tags=["Code Evolution"])
app.include_router(change_storytelling_router, prefix="/api/change-storytelling", tags=["Change Storytelling"])
app.include_router(hotspot_prediction_router, prefix="/api/hotspot-prediction", tags=["Hotspot Prediction"])
app.include_router(refactor_guide_router, prefix="/api/refactor-guide", tags=["Refactor Guide"])
app.include_router(code_map_router, prefix="/api/code-map", tags=["Code Map"])
app.include_router(insights_router, prefix="/api/insights", tags=["Insights"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to EchoLens API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }