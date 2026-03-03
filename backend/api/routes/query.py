from fastapi import APIRouter

router = APIRouter()

# TODO: Implement query routes
@router.get("/")
async def placeholder():
    return {"message": "query route — coming soon"}
