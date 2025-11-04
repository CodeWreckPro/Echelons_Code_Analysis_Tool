from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.models.insights import ModuleNode, ModuleConnection, CodeMap
from app.services.code_map_service import CodeMapService

router = APIRouter()
code_map_service = CodeMapService()

@router.get("/map", response_model=CodeMap)
async def get_code_map(path: str):
    """
    Generate a 3D visualization map of the codebase.
    """
    try:
        return code_map_service.generate_map(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hotspots", response_model=List[ModuleNode])
async def get_hotspot_nodes(path: str):
    """
    Get nodes representing code hotspots.
    """
    try:
        return code_map_service.identify_hotspots(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dependencies", response_model=List[ModuleConnection])
async def get_module_dependencies(path: str):
    """
    Get dependency relationships between modules.
    """
    try:
        return code_map_service.analyze_dependencies(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/module-info/{module_id}")
async def get_module_info(module_id: str):
    """
    Get detailed information about a specific module.
    """
    try:
        return code_map_service.get_module_details(module_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))