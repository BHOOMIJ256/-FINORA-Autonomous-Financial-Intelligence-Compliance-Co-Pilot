from fastapi import APIRouter

router = APIRouter()

# TODO: Implement compliance routes
@router.get("/")
async def placeholder():
    return {"message": "compliance route — coming soon"}
