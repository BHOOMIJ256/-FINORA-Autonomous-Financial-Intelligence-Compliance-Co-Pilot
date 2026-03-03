from fastapi import APIRouter

router = APIRouter()

# TODO: Implement transactions routes
@router.get("/")
async def placeholder():
    return {"message": "transactions route — coming soon"}
