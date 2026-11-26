
from fastapi import APIRouter

from app.modules.agent.route import router as AgentRoutes
from app.modules.documents.route import router as DocumentsRoutes
from app.modules.requirements.route import router as RequirementsRoutes
from app.modules.rfp_analyze.route import router as RfpAnalyzeRoutes
from app.modules.drafting.route import router as DraftingRoutes


router = APIRouter()

module_routes = [
    {
        "path": "/agent",
        "route": AgentRoutes,
    },
    {
        "path": "/documents",
        "route": DocumentsRoutes,
    },
    {
        "path": "/rfp",
        "route": RfpAnalyzeRoutes,
    },
    {
        "path": "/requirements",
        "route": RequirementsRoutes,
    },
    {
        "path": "/drafting",
        "route": DraftingRoutes,
    },
]

for module in module_routes:
    router.include_router(
        module["route"],
        prefix=module["path"],
    )
