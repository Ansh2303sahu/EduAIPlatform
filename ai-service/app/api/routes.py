from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.infer_student import router as student_router
from app.api.infer_professor import router as professor_router
from app.api.infer_similarity import router as similarity_router
from app.api.infer_tables import router as tables_router
from app.api.infer_professor_multimodal import router as professor_mm_router
api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(student_router)
api_router.include_router(professor_router)
api_router.include_router(similarity_router)
api_router.include_router(tables_router)
api_router.include_router(professor_mm_router)