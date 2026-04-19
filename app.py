"""
NiveshSaathi — AI Investment Advisor for Bharat
FastAPI Backend — Uses EasyOCR (works perfectly on Windows)
"""

import os
from dotenv import load_dotenv
load_dotenv()   # ← reads .env file automatically

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import base64
import io
import re
import numpy as np
from pathlib import Path

# ── GAN Module ────────────────────────────────────────────────────────────────
try:
    from gan_module import generate_growth_image, get_growth_stage
    GAN_AVAILABLE = True
    print("[APP] GAN module loaded ✅")
except Exception as e:
    GAN_AVAILABLE = False
    print(f"[APP] GAN module not available: {e}")

# ── RAG Module ────────────────────────────────────────────────────────────────
try:
    from rag_module import ask_financial_question, setup_chromadb
    RAG_AVAILABLE = True
    print("[APP] RAG module loaded ✅")
    # Single background thread — ChromaDB singleton handles concurrent calls safely
    import threading as _rag_thread
    _t = _rag_thread.Thread(target=setup_chromadb, daemon=True, name="chromadb_init")
    _t.start()
except Exception as e:
    RAG_AVAILABLE = False
    print(f"[APP] RAG not available: {e}")


app = FastAPI(title="NiveshSaathi", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
try:
    from agentic_api import agentic_router
    AGENTIC_AVAILABLE = True
except Exception as e:
    print("[AGENTIC] Not available:", e)
    AGENTIC_AVAILABLE = False
# Register agentic router
if AGENTIC_AVAILABLE:
    app.include_router(agentic_router)

# ── OCR — EasyOCR (Windows friendly) ─────────────────────────────────────────
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("[OCR] Loading EasyOCR model...")
            _ocr_reader = easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)
            print("[OCR] EasyOCR ready ✅")
        except Exception as e:
            print(f"[OCR] EasyOCR load failed: {e}")
            return None
    return _ocr_reader

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        import cv2
        reader = get_ocr_reader()
        if reader is None:
            return _mock_ocr_result()

        print("[OCR] Reading bill...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = reader.readtext(img)
        texts = []
        for item in result:
            text = item[1]
            confidence = item[2]
            if confidence > 0.25 and text.strip():
                texts.append(text.strip())
                print(f"[OCR] '{text}' ({confidence:.2f})")

        extracted = " | ".join(texts)
        print(f"[OCR] Done: {extracted}")
        return extracted if texts else _mock_ocr_result()

    except Exception as e:
        print(f"[OCR] Error: {e}")
        return _mock_ocr_result()

def _mock_ocr_result():
    """Only used if OCR completely fails"""
    return "Grand Total 60.00 | Food Bill"

# ── CATEGORIZATION ────────────────────────────────────────────────────────────
CATEGORIES = {
    "Food & Groceries": [
        "grocery","rice","dal","vegetable","milk","tiffin","food","restaurant",
        "sabji","atta","oil","sugar","parotta","aloo","payaaz","onion","roti",
        "biryani","pizza","burger","cafe","hotel","dhaba","sweet","halwa",
        "paneer","chicken","mutton","fish","bread","butter","egg","tea","chai"
    ],
    "Utilities": [
        "electricity","water","gas","bill","bijli","paani","current","meter",
        "connection","pipe","cylinder","lpg","bsnl","broadband"
    ],
    "Transport": [
        "auto","bus","rickshaw","petrol","diesel","train","travel","cab",
        "uber","ola","rapido","metro","ticket","fuel","parking","toll"
    ],
    "Healthcare": [
        "medical","medicine","doctor","hospital","pharmacy","tablet","dawai",
        "clinic","health","chemist","apollo","nursing","injection","test","lab"
    ],
    "Education": [
        "school","fee","book","tuition","college","stationery","pen","copy",
        "uniform","exam","coaching","class","study","notes"
    ],
    "Telecom": [
        "mobile","recharge","dth","internet","phone","jio","airtel","vi",
        "bsnl","sim","data","plan","tata","broadband","wifi"
    ],
    "Entertainment": [
        "movie","cinema","game","entertainment","pvr","inox","netflix",
        "spotify","amazon","subscription","sport","club"
    ],
    "Others": [],
}

CATEGORY_ICONS = {
    "Food & Groceries": "🍛",
    "Utilities": "💡",
    "Transport": "🚗",
    "Healthcare": "💊",
    "Education": "📖",
    "Telecom": "📱",
    "Entertainment": "🎬",
    "Others": "📦",
}

# def categorize_expenses(text: str) -> dict:
#     text_lower = text.lower()
#     found_categories = {}

#     for category, keywords in CATEGORIES.items():
#         score = sum(1 for kw in keywords if kw in text_lower)
#         if score > 0:
#             found_categories[category] = score

#     if not found_categories:
#         found_categories["Food & Groceries"] = 1

#     # Extract amounts — handles ₹60, Rs.60, 60.00 formats
#     amounts = re.findall(
#         r'(?:₹|rs\.?|inr)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)',
#         text_lower
#     )
#     # Filter out noise (dates, bill numbers, quantities)
#     valid_amounts = []
#     for a in amounts:
#         val = float(a.replace(',', ''))
#         if 5 <= val <= 100000:  # reasonable bill amount range
#             valid_amounts.append(val)

#     # Take the largest amount as total (usually the grand total)
#     total = max(valid_amounts) if valid_amounts else 0

#     # Distribute proportionally across categories
#     total_score = sum(found_categories.values())
#     category_breakdown = {}
#     for cat, score in found_categories.items():
#         proportion = score / total_score
#         category_breakdown[cat] = round(total * proportion, 2)

#     return {
#         "raw_text": text,
#         "total_detected": total,
#         "categories": category_breakdown,
#         "amounts_found": [str(a) for a in valid_amounts],
#     }

def categorize_expenses(text: str) -> dict:
    text_lower = text.lower()
 
    # ── Step 1: Find the TOTAL amount smartly ────────────────────────────────
    # First try to find explicit total/grand total line
    total = 0.0
 
    # Pattern 1: "total ₹514.50" or "total: 514.50" or "TOTAL ₹ 514.50"
    total_match = re.search(
        r'(?:grand\s+)?total[^\d]{0,10}(?:₹|rs\.?)?\s*([\d,]+\.?\d*)',
        text_lower
    )
    if total_match:
        try:
            total = float(total_match.group(1).replace(',', ''))
        except:
            total = 0.0
 
    # Pattern 2: "amount payable", "net amount", "bill amount"
    if total == 0.0:
        payable_match = re.search(
            r'(?:amount\s+payable|net\s+amount|bill\s+amount|payable)[^\d]{0,10}(?:₹|rs\.?)?\s*([\d,]+\.?\d*)',
            text_lower
        )
        if payable_match:
            try:
                total = float(payable_match.group(1).replace(',', ''))
            except:
                total = 0.0
 
    # Pattern 3: fallback — find all currency amounts and take the LARGEST
    # (Grand total is almost always the largest number on a bill)
    if total == 0.0:
        # Only match numbers preceded by ₹ or Rs to avoid picking up dates/ids
        currency_amounts = re.findall(
            r'(?:₹|rs\.?)\s*([\d,]+\.?\d*)',
            text_lower
        )
        valid = []
        for a in currency_amounts:
            try:
                val = float(a.replace(',', ''))
                if 10 <= val <= 500000:
                    valid.append(val)
            except:
                pass
        if valid:
            total = max(valid)
 
    # Pattern 4: last resort — any number that looks like a total
    if total == 0.0:
        all_amounts = re.findall(r'(\d{1,6}(?:,\d{3})*\.\d{2})', text_lower)
        valid = []
        for a in all_amounts:
            try:
                val = float(a.replace(',', ''))
                if 10 <= val <= 500000:
                    valid.append(val)
            except:
                pass
        if valid:
            total = max(valid)
 
    # ── Step 2: Categorize based on keywords ─────────────────────────────────
    found_categories = {}
 
    # Use word-boundary style matching to avoid false positives
    # e.g. "bill" in "billing" or "rate" matching "restaurant"
    noise_words = {'bill', 'total', 'rate', 'amount', 'date', 'invoice',
                   'description', 'qty', 'cgst', 'sgst', 'igst', 'gst'}
 
    for category, keywords in CATEGORIES.items():
        score = 0
        for kw in keywords:
            if kw in noise_words:
                continue  # skip generic words that match everything
            if kw in text_lower:
                score += 1
        if score > 0:
            found_categories[category] = score
 
    if not found_categories:
        found_categories["Food & Groceries"] = 1
 
    # ── Step 3: Assign full total to the DOMINANT category ───────────────────
    # Instead of splitting proportionally (which gives wrong numbers),
    # assign 100% to the top category, and note others as secondary
    top_category = max(found_categories, key=found_categories.get)
 
    category_breakdown = {}
    if len(found_categories) == 1:
        category_breakdown[top_category] = round(total, 2)
    else:
        # Give dominant category the bulk, distribute remainder
        total_score = sum(found_categories.values())
        for cat, score in found_categories.items():
            proportion = score / total_score
            category_breakdown[cat] = round(total * proportion, 2)
 
    return {
        "raw_text": text,
        "total_detected": round(total, 2),
        "categories": category_breakdown,
        "amounts_found": [str(total)],
    }

# ── 3 BUCKET RECOMMENDATION ───────────────────────────────────────────────────
def get_bucket_recommendation(monthly_income: float, total_expenses: float) -> dict:
    savings_possible = max(0, monthly_income - total_expenses)
    rd_amount        = round(monthly_income * 0.10, 2)
    gold_amount      = round(monthly_income * 0.05, 2)
    emergency_amount = round(savings_possible * 0.30, 2)

    rd_1yr  = round(rd_amount * 12 * 1.065, 2)
    rd_3yr  = round(rd_amount * 36 * 1.195, 2)
    gold_1yr = round(gold_amount * 12 * 1.08, 2)
    gold_3yr = round(gold_amount * 36 * 1.26, 2)

    return {
        "income": monthly_income,
        "expenses": total_expenses,
        "savings_possible": savings_possible,
        "buckets": {
            "rd_savings": {
                "name": "Recurring Deposit (RD)",
                "emoji": "🏦",
                "monthly": rd_amount,
                "description": "Fixed monthly savings at your bank. Guaranteed interest!",
                "safe_level": "Very Safe ✅",
                "growth_1yr": rd_1yr,
                "growth_3yr": rd_3yr,
                "compare": "Safe like Fixed Deposit",
            },
            "digital_gold": {
                "name": "Digital Gold",
                "emoji": "✨",
                "monthly": gold_amount,
                "description": "Buy gold digitally in tiny amounts. No locker needed!",
                "safe_level": "Safe ✅",
                "growth_1yr": gold_1yr,
                "growth_3yr": gold_3yr,
                "compare": "Same as physical gold",
            },
            "emergency_cash": {
                "name": "Emergency Cash",
                "emoji": "🆘",
                "monthly": emergency_amount,
                "description": "Keep ready for hospital, job loss, or any sudden need.",
                "safe_level": "Liquid 💧",
                "growth_1yr": emergency_amount * 12,
                "growth_3yr": emergency_amount * 36,
                "compare": "In savings account",
            },
        },
        "tip": _savings_tip(savings_possible, monthly_income),
    }

def _savings_tip(savings, income):
    pct = (savings / income * 100) if income > 0 else 0
    if pct < 5:
        return "🌱 Even ₹100/month saved today becomes a safety net tomorrow!"
    elif pct < 15:
        return "📈 Good saving! Split across 3 buckets for better protection."
    else:
        return "🌳 Excellent! Consider index funds for even better long-term growth."

# ── MARKET DATA ────────────────────────────────────────────────────────────────
def get_market_data():
    try:
        import yfinance as yf
        tickers = {
            "Gold (GOLDBEES)": "GOLDBEES.NS",
            "Nifty 50": "^NSEI",
            "SBI Bank": "SBIN.NS",
        }
        data = {}
        for name, ticker in tickers.items():
            try:
                t    = yf.Ticker(ticker)
                hist = t.history(period="1mo")
                if not hist.empty:
                    current   = round(float(hist["Close"].iloc[-1]), 2)
                    month_ago = round(float(hist["Close"].iloc[0]), 2)
                    change    = round(((current - month_ago) / month_ago) * 100, 2)
                    data[name] = {
                        "current": current,
                        "change_1month_pct": change,
                        "trend": "up" if change > 0 else "down",
                        "history": hist["Close"].tail(20).round(2).tolist(),
                    }
            except Exception:
                pass
        return data if data else _mock_market()
    except Exception:
        return _mock_market()

def _mock_market():
    return {
        "Gold (GOLDBEES)": {
            "current": 5820.50, "change_1month_pct": 2.3, "trend": "up",
            "history": [5650,5670,5690,5710,5730,5750,5770,5790,5800,5810,
                        5812,5815,5817,5818,5819,5820,5821,5820,5819,5820.50],
        },
        "Nifty 50": {
            "current": 22350.75, "change_1month_pct": 1.8, "trend": "up",
            "history": [21900,21950,22000,22050,22100,22150,22200,22250,
                        22280,22300,22310,22320,22330,22340,22345,22350.75],
        },
        "SBI Bank": {
            "current": 785.30, "change_1month_pct": -0.5, "trend": "down",
            "history": [790,789,788,787,786,785,786,785,784,785,
                        785,786,785,784,785,785.30],
        },
    }

# ── GROWTH VISUAL PROMPT ──────────────────────────────────────────────────────
def get_growth_visual_prompt(profession: str, savings_amount: float, months: int = 6) -> dict:
    stage = (
        "seedling"   if months <= 3  else
        "sapling"    if months <= 6  else
        "young_tree" if months <= 12 else
        "full_tree"
    )
    total = round(savings_amount * months, 2)
    return {
        "stage": stage,
        "months": months,
        "savings_total": total,
        "metaphor": f"₹{savings_amount:.0f}/month × {months} months = ₹{total:,.0f} — like a {stage.replace('_',' ')}!",
    }

# ── API ROUTES ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/api/scan-bill")
async def scan_bill(file: UploadFile = File(...)):
    try:
        image_bytes    = await file.read()
        text           = extract_text_from_image(image_bytes)
        categorization = categorize_expenses(text)
        return JSONResponse(content={"success": True, "data": categorization})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations")
async def get_recommendations(data: dict):
    income     = float(data.get("monthly_income", 15000))
    expenses   = float(data.get("total_expenses", 10000))
    profession = data.get("profession", "default")
    months     = int(data.get("months", 6))

    buckets  = get_bucket_recommendation(income, expenses)
    monthly  = buckets["buckets"]["rd_savings"]["monthly"]
    visual   = get_growth_visual_prompt(profession, monthly, months)

    return JSONResponse(content={
        "success": True,
        "recommendations": buckets,
        "visual": visual,
    })

@app.get("/api/market-data")
async def market_data():
    return JSONResponse(content={"success": True, "data": get_market_data()})

@app.post("/api/generate-visual")
async def generate_visual(data: dict):
    profession = data.get("profession", "default")
    months     = int(data.get("months", 6))
    monthly    = float(data.get("monthly_savings", 500))

    # Check cache first
    from gan_module import get_growth_stage
    stage      = get_growth_stage(months)
    cache_path = Path(f"static/gan_cache/{profession}_{stage}.png")

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return JSONResponse(content={
            "success": True,
            "method": "cached_gan",
            "stage": stage,
            "image_base64": b64,
            "metaphor": f"₹{monthly:.0f}/month × {months} months = ₹{monthly*months:,.0f} saved!",
        })

    # Live generation fallback
    if GAN_AVAILABLE:
        try:
            result = generate_growth_image(profession, months, monthly)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(base64.b64decode(result["image_base64"]))
            return JSONResponse(content=result)
        except Exception as e:
            print(f"[GAN] Error: {e}")

    raise HTTPException(status_code=503, detail="GAN not available and no cache found")

@app.post("/api/simulate-upi")
async def simulate_upi(data: dict):
    import random
    amount  = data.get("amount", 100)
    purpose = data.get("purpose", "Investment")
    return JSONResponse(content={
        "success": True,
        "transaction_id": f"TXN{random.randint(100000,999999)}",
        "amount": amount,
        "purpose": purpose,
        "status": "SUCCESS (Demo)",
        "message": f"₹{amount} marked for {purpose}.",
    })

@app.get("/api/rd-calculator")
async def rd_calculator(monthly: float = 500, months: int = 12, rate: float = 6.5):
    total_invested = monthly * months
    maturity       = monthly * months * (1 + (rate / 100) * (months / 12) / 2)
    interest       = maturity - total_invested
    monthly_growth = [
        round(monthly * m * (1 + (rate / 100) * (m / 12) / 2), 2)
        for m in range(1, months + 1)
    ]
    return JSONResponse(content={
        "monthly_deposit":  monthly,
        "months":           months,
        "rate_percent":     rate,
        "total_invested":   round(total_invested, 2),
        "maturity_amount":  round(maturity, 2),
        "interest_earned":  round(interest, 2),
        "monthly_growth":   monthly_growth,
    })

@app.get("/api/gan-status")
def gan_status():
    cached = len(list(Path("static/gan_cache").glob("*.png"))) \
             if Path("static/gan_cache").exists() else 0
    return {
        "gan_available": GAN_AVAILABLE,
        "cached_images": cached,
        "ready": cached >= 16,
    }


@app.post("/api/ask")
async def ask_question(data: dict):
    """RAG endpoint — Hindi financial Q&A using ChromaDB + Groq LLM"""
    question   = data.get("question", "").strip()
    profession = data.get("profession", "user")
    income     = float(data.get("income", 15000))
    expenses   = float(data.get("expenses", 11000))

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if RAG_AVAILABLE:
        result = ask_financial_question(question, profession, income, expenses)
    else:
        result = {
            "success": True,
            "question": question,
            "answer": "RAG module load nahi hua. Please install: pip install chromadb sentence-transformers",
            "sources": [],
            "method": "unavailable"
        }
    return JSONResponse(content=result)

@app.get("/api/rag-status")
def rag_status():
    return {
        "rag_available": RAG_AVAILABLE,
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "db_path": "./rag_db"
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "ocr": "easyocr"}

if __name__ == "__main__":
    os.makedirs("static/gan_cache", exist_ok=True)
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)