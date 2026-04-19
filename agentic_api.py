"""
agentic_api.py — FastAPI endpoints for Agentic AI
Integrates with existing NiveshSaathi app.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from agentic_planner import AutoInvestmentPlanner
import os

agentic_router = APIRouter(prefix="/api/agentic", tags=["Agentic AI"])

_planner = None

def get_planner():
    global _planner
    if _planner is None:
        _planner = AutoInvestmentPlanner(
            db_path="agentic_data.db",
            twilio_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            twilio_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
        )
    return _planner


# ── Request models ────────────────────────────────────────────────────────────

class SaveWhatsAppRequest(BaseModel):
    user_id: str
    phone_number: str   # any format: 9876543210 / +919876543210

class ProcessBillRequest(BaseModel):
    user_id: str
    bill_text: str
    monthly_income: float
    phone_number: Optional[str] = None

class ManualExpenseRequest(BaseModel):
    user_id: str
    category: str       # Food & Groceries / Shopping / Utilities / etc.
    amount: float
    description: Optional[str] = ""
    monthly_income: Optional[float] = 15000
    phone_number: Optional[str] = None

class GetTrendsRequest(BaseModel):
    user_id: str
    num_months: int = 3

class GetPlanRequest(BaseModel):
    user_id: str
    monthly_income: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@agentic_router.post("/save-whatsapp")
async def save_whatsapp_number(request: SaveWhatsAppRequest):
    """
    Save user's WhatsApp number for alerts.
    Call this ONCE — all future alerts auto-use this number.

    Accepts any format:
    - "9876543210"
    - "+919876543210"
    - "whatsapp:+919876543210"
    """
    try:
        planner = get_planner()
        result  = planner.save_user_whatsapp(request.user_id, request.phone_number)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.post("/add-manual-expense")
async def add_manual_expense(request: ManualExpenseRequest):
    """
    Add expense manually without scanning a bill.
    User types: category + amount + optional description.

    Example categories:
    - Food & Groceries
    - Shopping
    - Utilities
    - Transport
    - Healthcare
    - Education
    - Telecom
    - Entertainment
    - Others

    Example request:
    {
        "user_id": "user_123",
        "category": "Shopping",
        "amount": 2000,
        "description": "Kapde kharide market se",
        "monthly_income": 15000
    }
    """
    try:
        planner = get_planner()
        result  = planner.add_manual_expense(
            user_id=request.user_id,
            category=request.category,
            amount=request.amount,
            description=request.description or "",
            monthly_income=request.monthly_income or 15000,
            phone_number=request.phone_number,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.post("/process-bill")
async def process_bill(request: ProcessBillRequest):
    """
    Process a scanned bill through the full autonomous pipeline.
    If phone_number not passed, uses the saved number from /save-whatsapp.
    """
    try:
        planner = get_planner()
        result  = planner.process_new_bill(
            user_id=request.user_id,
            bill_text=request.bill_text,
            monthly_income=request.monthly_income,
            phone_number=request.phone_number,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.post("/analyze-trends")
async def analyze_trends(request: GetTrendsRequest):
    try:
        planner = get_planner()
        return planner.trend_analyzer.analyze_trends(request.user_id, request.num_months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.post("/adjust-plan")
async def adjust_plan(request: GetPlanRequest):
    try:
        planner = get_planner()
        trends  = planner.trend_analyzer.analyze_trends(request.user_id, 3)
        return planner.plan_adjuster.adjust_plan(request.user_id, request.monthly_income, trends)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str):
    try:
        return get_planner().get_user_dashboard(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.get("/bills/{user_id}")
async def get_bills(user_id: str, month: Optional[str] = None):
    from datetime import datetime
    if not month:
        month = datetime.now().strftime("%Y-%m")
    try:
        return get_planner().bill_scanner.get_monthly_summary(user_id, month)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@agentic_router.get("/categories")
async def get_categories():
    """Return all valid expense categories for manual entry UI"""
    from agentic_planner import CATEGORIES
    return {
        "categories": list(CATEGORIES.keys()) + ["Rent", "EMI"],
        "examples": {
            "Food & Groceries": "Kirana, restaurant, grocery",
            "Shopping": "Kapde, shoes, Amazon order",
            "Utilities": "Bijli, paani, gas",
            "Transport": "Auto, petrol, bus pass",
            "Healthcare": "Doctor, medicine, lab test",
            "Education": "School fee, tuition, books",
            "Telecom": "Mobile recharge, DTH, internet",
            "Entertainment": "Movie, Netflix, event",
            "Rent": "Ghar ka kiraya",
            "EMI": "Loan EMI payment",
            "Others": "Anything else",
        }
    }


@agentic_router.get("/status")
async def get_status():
    try:
        p = get_planner()
        return {
            "status": "online",
            "whatsapp_enabled": p.alert_agent.whatsapp_enabled,
            "components": {
                "bill_scanner":    "✅ Active",
                "trend_analyzer":  "✅ Active",
                "plan_adjuster":   "✅ Active",
                "alert_agent":     "✅ WhatsApp" if p.alert_agent.whatsapp_enabled else "⚠️ Console only",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))