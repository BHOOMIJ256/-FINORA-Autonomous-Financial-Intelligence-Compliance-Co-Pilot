from fastapi import APIRouter

router = APIRouter()

# TODO: Implement audit routes
@router.get("/")
async def placeholder():
    return {"message": "audit route — coming soon"}
