
from fastapi import APIRouter

from app.modules.agent.route import router as AgentRoutes
from app.modules.rfp_analyze.route import router as RfpAnalyzeRoutes


router = APIRouter()

module_routes = [
    {
        "path": "/agent",
        "route": AgentRoutes,
    },
    {
        "path": "/rfp",
        "route": RfpAnalyzeRoutes,
    },
]

for module in module_routes:
    router.include_router(
        module["route"],
        prefix=module["path"],
    )
