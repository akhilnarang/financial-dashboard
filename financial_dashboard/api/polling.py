from fastapi import APIRouter, Request

from financial_dashboard.schemas.polling import PollStatusResponse

router = APIRouter()


@router.get("/poll/status", response_model=PollStatusResponse)
async def poll_status(request: Request) -> PollStatusResponse:
    service = getattr(request.app.state, "fetch_service", None)
    if service is None:
        return PollStatusResponse(state="idle")
    return PollStatusResponse.model_validate(service.get_poll_status())
