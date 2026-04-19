"""
agentic_planner.py — Auto Investment Planner (FIXED)
4 Autonomous Agents:
1. Bill Scanner Agent    — categorizes bills
2. Trend Analyzer Agent  — detects spending patterns
3. Plan Adjuster Agent   — updates investment buckets
4. Alert Agent           — WhatsApp alerts via Twilio

FIXES from original:
- Removed broken LangChain imports (not needed, added complexity)
- Removed broken HuggingFaceHub import
- Fixed WhatsApp number format handling
- Added manual expense entry support
- Fixed ConversationBufferMemory import error
- Simplified to work with zero extra dependencies beyond twilio
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

class AgenticDB:
    def __init__(self, db_path: str = "agentic_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS bills (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    NOT NULL,
                date          TEXT    NOT NULL,
                category      TEXT    NOT NULL,
                amount        REAL    NOT NULL,
                raw_text      TEXT,
                source        TEXT    DEFAULT 'ocr',
                created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trends (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                month           TEXT NOT NULL,
                category        TEXT NOT NULL,
                total_amount    REAL NOT NULL,
                avg_amount      REAL NOT NULL,
                trend_direction TEXT,
                percent_change  REAL,
                updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, month, category) ON CONFLICT REPLACE
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS investment_plans (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           TEXT NOT NULL,
                month             TEXT NOT NULL,
                monthly_income    REAL NOT NULL,
                monthly_expenses  REAL NOT NULL,
                rd_amount         REAL NOT NULL,
                gold_amount       REAL NOT NULL,
                emergency_amount  REAL NOT NULL,
                created_at        TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             TEXT NOT NULL,
                alert_type          TEXT NOT NULL,
                category            TEXT,
                message             TEXT NOT NULL,
                threshold_exceeded  REAL,
                sent_via            TEXT,
                sent_at             TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Store user WhatsApp number
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id       TEXT PRIMARY KEY,
                whatsapp_number TEXT,
                monthly_income  REAL,
                profession      TEXT,
                updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    # ── Bills ────────────────────────────────────────────────────────────────
    def add_bill(self, user_id, category, amount, raw_text="", source="ocr"):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO bills (user_id, date, category, amount, raw_text, source) VALUES (?,?,?,?,?,?)",
            (user_id, datetime.now().strftime("%Y-%m-%d"), category, amount, raw_text, source)
        )
        self.conn.commit()
        return c.lastrowid

    def get_bills_by_month(self, user_id, month):
        c = self.conn.cursor()
        c.execute(
            "SELECT id,date,category,amount,raw_text,source FROM bills "
            "WHERE user_id=? AND strftime('%Y-%m',date)=? ORDER BY date DESC",
            (user_id, month)
        )
        return [{"id":r[0],"date":r[1],"category":r[2],"amount":r[3],
                 "raw_text":r[4],"source":r[5]} for r in c.fetchall()]

    def get_category_total(self, user_id, month, category):
        c = self.conn.cursor()
        c.execute(
            "SELECT SUM(amount) FROM bills WHERE user_id=? AND strftime('%Y-%m',date)=? AND category=?",
            (user_id, month, category)
        )
        r = c.fetchone()[0]
        return r if r else 0.0

    # ── Trends ───────────────────────────────────────────────────────────────
    def save_trend(self, user_id, month, category, total, avg, direction, pct):
        self.conn.execute(
            "INSERT OR REPLACE INTO trends "
            "(user_id,month,category,total_amount,avg_amount,trend_direction,percent_change) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, month, category, total, avg, direction, pct)
        )
        self.conn.commit()

    # ── Plans ────────────────────────────────────────────────────────────────
    def save_plan(self, user_id, income, expenses, rd, gold, emergency):
        self.conn.execute(
            "INSERT INTO investment_plans "
            "(user_id,month,monthly_income,monthly_expenses,rd_amount,gold_amount,emergency_amount) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, datetime.now().strftime("%Y-%m"), income, expenses, rd, gold, emergency)
        )
        self.conn.commit()

    def get_latest_plan(self, user_id):
        c = self.conn.cursor()
        c.execute(
            "SELECT month,monthly_income,monthly_expenses,rd_amount,gold_amount,emergency_amount "
            "FROM investment_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )
        r = c.fetchone()
        if r:
            return {"month":r[0],"income":r[1],"expenses":r[2],
                    "rd_amount":r[3],"gold_amount":r[4],"emergency_amount":r[5]}
        return None

    # ── Alerts ───────────────────────────────────────────────────────────────
    def save_alert(self, user_id, alert_type, category, message, threshold=0.0, sent_via="console"):
        self.conn.execute(
            "INSERT INTO alerts (user_id,alert_type,category,message,threshold_exceeded,sent_via) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, alert_type, category, message, threshold, sent_via)
        )
        self.conn.commit()

    # ── User settings (WhatsApp number) ──────────────────────────────────────
    def save_user_settings(self, user_id, whatsapp_number=None,
                           monthly_income=None, profession=None):
        self.conn.execute("""
            INSERT INTO user_settings (user_id, whatsapp_number, monthly_income, profession, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                whatsapp_number = COALESCE(excluded.whatsapp_number, whatsapp_number),
                monthly_income  = COALESCE(excluded.monthly_income,  monthly_income),
                profession      = COALESCE(excluded.profession,      profession),
                updated_at      = excluded.updated_at
        """, (user_id, whatsapp_number, monthly_income, profession, datetime.now().isoformat()))
        self.conn.commit()

    def get_user_settings(self, user_id):
        c = self.conn.cursor()
        c.execute(
            "SELECT whatsapp_number,monthly_income,profession FROM user_settings WHERE user_id=?",
            (user_id,)
        )
        r = c.fetchone()
        if r:
            return {"whatsapp_number": r[0], "monthly_income": r[1], "profession": r[2]}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1: BILL SCANNER
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "Food & Groceries": [
        "grocery","rice","dal","vegetable","milk","food","restaurant","sabji",
        "atta","oil","sugar","parotta","aloo","payaaz","onion","roti","biryani",
        "pizza","burger","cafe","hotel","dhaba","sweet","paneer","chicken",
        "bread","butter","egg","tea","chai","tiffin","thali"
    ],
    "Utilities": [
        "electricity","water","gas","bill","bijli","paani","current","meter",
        "cylinder","lpg","broadband","wifi","connection"
    ],
    "Transport": [
        "auto","bus","rickshaw","petrol","diesel","train","travel","cab",
        "uber","ola","rapido","metro","ticket","fuel","parking","toll","fare"
    ],
    "Healthcare": [
        "medical","medicine","doctor","hospital","pharmacy","tablet","dawai",
        "clinic","health","chemist","apollo","injection","test","lab","pathology"
    ],
    "Education": [
        "school","fee","book","tuition","college","stationery","pen","copy",
        "uniform","exam","coaching","class","study","notes"
    ],
    "Telecom": [
        "mobile recharge","dth","internet","jio","airtel","vodafone","vi",
        "bsnl","sim","data plan","tata sky","broadband recharge"
    ],
    "Shopping": [
        "shopping","clothes","shirt","pant","saree","shoes","amazon","flipkart",
        "meesho","myntra","garment","dress","kurta","kurti","mall","store"
    ],
    "Entertainment": [
        "movie","cinema","game","netflix","pvr","inox","hotstar","sport","club"
    ],
    "Others": [],
}

class BillScannerAgent:
    def __init__(self, db: AgenticDB):
        self.db = db

    def scan_and_categorize(self, user_id: str, bill_text: str,
                             source: str = "ocr") -> Dict[str, Any]:
        text_lower = bill_text.lower()

        # Smart amount extraction — find Grand Total first
        total = 0.0
        gt = re.search(r'grand\s*total[\s|:]*(\d+(?:\.\d{1,2})?)', text_lower)
        if gt:
            total = float(gt.group(1))
        else:
            amounts = re.findall(r'(?:₹|rs\.?|inr)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)', text_lower)
            valid = [float(a.replace(',','')) for a in amounts if 5 <= float(a.replace(',','')) <= 100000]
            if valid:
                # Filter obvious noise (tax percentages etc.)
                valid = [v for v in valid if v != 2.5 and v != 5.0 and v != 12.0 and v != 18.0]
                total = max(valid) if valid else 0.0

        # Categorize
        scores = {}
        for cat, keywords in CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[cat] = score
        best_cat = max(scores, key=scores.get) if scores else "Others"

        bill_id = self.db.add_bill(user_id, best_cat, total, bill_text, source)
        return {
            "success": True,
            "bill_id": bill_id,
            "amount": round(total, 2),
            "category": best_cat,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

    def add_manual_expense(self, user_id: str, category: str,
                           amount: float, description: str = "") -> Dict[str, Any]:
        """
        Add expense manually — e.g. Shopping ₹2000, Rent ₹5000
        User types category + amount directly without scanning a bill
        """
        if category not in CATEGORIES and category != "Rent":
            category = "Others"

        # Validate amount
        if amount <= 0 or amount > 500000:
            return {"success": False, "error": "Invalid amount"}

        raw_text = f"Manual entry | {category} | Amount: {amount} | {description}"
        bill_id = self.db.add_bill(user_id, category, amount, raw_text, source="manual")

        return {
            "success": True,
            "bill_id": bill_id,
            "amount": amount,
            "category": category,
            "description": description,
            "source": "manual",
            "message": f"✅ ₹{amount:,.0f} added to {category}",
            "timestamp": datetime.now().isoformat(),
        }

    def get_monthly_summary(self, user_id: str, month: str = None) -> Dict[str, Any]:
        if not month:
            month = datetime.now().strftime("%Y-%m")
        bills = self.db.get_bills_by_month(user_id, month)
        cat_totals = {}
        for b in bills:
            cat_totals[b["category"]] = round(cat_totals.get(b["category"], 0) + b["amount"], 2)
        return {
            "month": month,
            "total_bills": len(bills),
            "total_spending": round(sum(cat_totals.values()), 2),
            "categories": cat_totals,
            "bills": bills,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2: TREND ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class TrendAnalyzerAgent:
    def __init__(self, db: AgenticDB):
        self.db = db

    def analyze_trends(self, user_id: str, num_months: int = 3) -> Dict[str, Any]:
        now = datetime.now()
        months_data = []

        for i in range(num_months):
            month_str = (now - timedelta(days=30 * i)).strftime("%Y-%m")
            bills = self.db.get_bills_by_month(user_id, month_str)
            cat_totals = {}
            for b in bills:
                cat_totals[b["category"]] = cat_totals.get(b["category"], 0) + b["amount"]
            months_data.append({
                "month": month_str,
                "totals": cat_totals,
                "total_spending": sum(cat_totals.values()),
            })

        # Per-category trends
        all_cats = set()
        for m in months_data:
            all_cats.update(m["totals"].keys())

        category_trends = {}
        for cat in all_cats:
            amounts = [m["totals"].get(cat, 0) for m in reversed(months_data)]
            if len(amounts) >= 2:
                recent, previous = amounts[-1], amounts[-2]
                pct = ((recent - previous) / previous * 100) if previous > 0 else (100.0 if recent > 0 else 0.0)
                direction = "increasing" if pct > 10 else ("decreasing" if pct < -10 else "stable")
                avg = sum(amounts) / len(amounts)
                category_trends[cat] = {
                    "amounts": [round(a, 2) for a in amounts],
                    "avg": round(avg, 2),
                    "recent": round(recent, 2),
                    "previous": round(previous, 2),
                    "pct_change": round(pct, 2),
                    "direction": direction,
                }
                self.db.save_trend(user_id, now.strftime("%Y-%m"),
                                   cat, recent, avg, direction, pct)

        # Overall trend
        totals = [m["total_spending"] for m in reversed(months_data)]
        if len(totals) >= 2:
            recent_t, prev_t = totals[-1], totals[-2]
            pct_overall = ((recent_t - prev_t) / prev_t * 100) if prev_t > 0 else 0.0
            direction_overall = "increasing" if pct_overall > 5 else ("decreasing" if pct_overall < -5 else "stable")
        else:
            recent_t = totals[0] if totals else 0
            prev_t, pct_overall, direction_overall = 0, 0.0, "stable"

        # Human-readable insights
        insights = []
        for cat, data in category_trends.items():
            if data["direction"] == "increasing" and data["pct_change"] > 20:
                insights.append(f"⚠️ {cat} spending increased by {data['pct_change']:.0f}% — review karo")
            elif data["direction"] == "decreasing" and data["pct_change"] < -20:
                insights.append(f"✅ {cat} spending reduced by {abs(data['pct_change']):.0f}% — great job!")
        if not insights:
            insights.append("📊 Spending is stable this month — keep it up!")

        return {
            "success": True,
            "category_trends": category_trends,
            "overall_trend": {
                "direction": direction_overall,
                "pct_change": round(pct_overall, 2),
                "recent_total": round(recent_t, 2),
                "previous_total": round(prev_t, 2),
            },
            "insights": insights,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3: PLAN ADJUSTER
# ══════════════════════════════════════════════════════════════════════════════

class PlanAdjusterAgent:
    def __init__(self, db: AgenticDB):
        self.db = db

    def adjust_plan(self, user_id: str, monthly_income: float,
                    trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        recent_total = trend_analysis["overall_trend"]["recent_total"]
        pct_change   = trend_analysis["overall_trend"]["pct_change"]
        savings      = max(0, monthly_income - recent_total)

        # Dynamically adjust buckets based on spending trend
        if pct_change > 15:       # spending rising fast
            rd_amount        = round(monthly_income * 0.08, 2)
            gold_amount      = round(monthly_income * 0.04, 2)
            emergency_amount = round(savings * 0.40, 2)
            reason = "⚠️ Kharcha badh raha hai — emergency fund priority"
        elif pct_change < -15:    # spending falling — invest more
            rd_amount        = round(monthly_income * 0.12, 2)
            gold_amount      = round(monthly_income * 0.06, 2)
            emergency_amount = round(savings * 0.25, 2)
            reason = "🌱 Kharcha kam hua — investment badha rahe hain"
        else:                     # stable
            rd_amount        = round(monthly_income * 0.10, 2)
            gold_amount      = round(monthly_income * 0.05, 2)
            emergency_amount = round(savings * 0.30, 2)
            reason = "📊 Stable spending — standard allocation"

        self.db.save_plan(user_id, monthly_income, recent_total,
                          rd_amount, gold_amount, emergency_amount)

        return {
            "success": True,
            "income": monthly_income,
            "expenses": recent_total,
            "savings_possible": round(savings, 2),
            "adjustment_reason": reason,
            "trend_direction": trend_analysis["overall_trend"]["direction"],
            "trend_pct_change": pct_change,
            "buckets": {
                "rd_savings": {
                    "name": "Recurring Deposit (RD)", "emoji": "🏦",
                    "monthly": rd_amount,
                    "growth_1yr": round(rd_amount * 12 * 1.065, 2),
                    "growth_3yr": round(rd_amount * 36 * 1.195, 2),
                },
                "digital_gold": {
                    "name": "Digital Gold", "emoji": "✨",
                    "monthly": gold_amount,
                    "growth_1yr": round(gold_amount * 12 * 1.08, 2),
                    "growth_3yr": round(gold_amount * 36 * 1.26, 2),
                },
                "emergency_cash": {
                    "name": "Emergency Cash", "emoji": "🆘",
                    "monthly": emergency_amount,
                },
            },
            "previous_plan": self.db.get_latest_plan(user_id),
        }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4: ALERT AGENT (WhatsApp via Twilio)
# ══════════════════════════════════════════════════════════════════════════════

def _format_whatsapp_number(number: str) -> str:
    """
    Normalize WhatsApp number to Twilio format: whatsapp:+91XXXXXXXXXX
    Accepts: 9876543210 / +919876543210 / 919876543210 / whatsapp:+919876543210
    """
    if not number:
        return ""
    n = number.strip()
    # Remove 'whatsapp:' prefix if present
    if n.startswith("whatsapp:"):
        n = n[9:]
    # Remove spaces and dashes
    n = re.sub(r'[\s\-]', '', n)
    # Add country code if missing
    if n.startswith("0"):
        n = "+91" + n[1:]
    elif n.startswith("91") and len(n) == 12:
        n = "+" + n
    elif not n.startswith("+"):
        n = "+91" + n  # default India
    return "whatsapp:" + n


class AlertAgent:
    def __init__(self, db: AgenticDB,
                 twilio_sid: str = None, twilio_token: str = None):
        self.db = db
        sid   = twilio_sid   or os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = twilio_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.whatsapp_enabled = bool(sid and token)
        self.twilio_client = None

        if self.whatsapp_enabled:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(sid, token)
                print("[ALERT] ✅ Twilio WhatsApp ready")
            except ImportError:
                print("[ALERT] twilio not installed — pip install twilio")
                self.whatsapp_enabled = False
            except Exception as e:
                print(f"[ALERT] Twilio init error: {e}")
                self.whatsapp_enabled = False

    def check_and_alert(self, user_id: str,
                        trend_analysis: Dict[str, Any],
                        plan: Dict[str, Any],
                        phone_number: str = None) -> Dict[str, Any]:
        """
        phone_number: user's WhatsApp number
        Accepts any format: 9876543210 / +919876543210 / whatsapp:+919876543210
        """
        # Get number from DB if not passed
        if not phone_number:
            settings = self.db.get_user_settings(user_id)
            phone_number = settings.get("whatsapp_number", "")

        formatted_number = _format_whatsapp_number(phone_number) if phone_number else ""
        alerts_sent = []

        income   = plan["income"]
        expenses = plan["expenses"]
        limit    = income * 0.80

        # Budget alert
        if expenses > limit:
            pct_over = ((expenses - limit) / limit) * 100
            msg = (
                f"⚠️ *Budget Alert — NiveshSaathi*\n\n"
                f"Is mahine ka kharcha: *₹{expenses:,.0f}*\n"
                f"Aapki income ka 80%: ₹{limit:,.0f}\n"
                f"Aap {pct_over:.1f}% zyada kharch kar rahe ho!\n\n"
                f"💡 Kharcha kam karne ki koshish karo — "
                f"₹{(expenses-limit):,.0f} bachao is mahine."
            )
            self._send(user_id, "budget_exceeded", "Overall", msg, pct_over, formatted_number)
            alerts_sent.append({"type": "budget_exceeded", "severity": "high"})

        # Category spike alerts
        for cat, data in trend_analysis["category_trends"].items():
            if data["direction"] == "increasing" and data["pct_change"] > 50:
                msg = (
                    f"🚨 *{cat} Alert — NiveshSaathi*\n\n"
                    f"{cat} kharcha {data['pct_change']:.0f}% badh gaya!\n"
                    f"Pichle mahine: ₹{data['previous']:,.0f}\n"
                    f"Is mahine: ₹{data['recent']:,.0f}\n\n"
                    f"💡 Review karo apna {cat} kharcha."
                )
                self._send(user_id, "category_spike", cat, msg, data["pct_change"], formatted_number)
                alerts_sent.append({"type": "category_spike", "category": cat})

        # Positive alert
        if trend_analysis["overall_trend"]["direction"] == "decreasing":
            saved = abs(trend_analysis["overall_trend"]["recent_total"] -
                        trend_analysis["overall_trend"]["previous_total"])
            if saved > 200:
                msg = (
                    f"🎉 *Badhai ho! — NiveshSaathi*\n\n"
                    f"Is mahine aapne ₹{saved:,.0f} zyada bachaye!\n"
                    f"RD mein ₹{plan['buckets']['rd_savings']['monthly']:.0f}/month daalna mat bhoolna. 🌱"
                )
                self._send(user_id, "positive_savings", "Overall", msg, 0, formatted_number)
                alerts_sent.append({"type": "positive_savings"})

        return {
            "success": True,
            "alerts_sent": len(alerts_sent),
            "alerts": alerts_sent,
            "whatsapp_enabled": self.whatsapp_enabled,
            "number_used": formatted_number or "not set",
        }

    def _send(self, user_id, alert_type, category, message, threshold, phone_number):
        sent_via = "console"
        if self.whatsapp_enabled and phone_number and self.twilio_client:
            try:
                self.twilio_client.messages.create(
                    from_="whatsapp:+14155238886",   # Twilio sandbox number
                    to=phone_number,
                    body=message,
                )
                sent_via = "whatsapp"
                print(f"[ALERT] ✅ WhatsApp sent to {phone_number}")
            except Exception as e:
                print(f"[ALERT] WhatsApp failed: {e}")
                sent_via = "console_fallback"

        print(f"\n{'='*55}")
        print(f"[ALERT] {alert_type.upper()} — {category}")
        print(f"{'='*55}")
        print(message)
        print(f"{'='*55}\n")
        self.db.save_alert(user_id, alert_type, category, message, threshold, sent_via)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class AutoInvestmentPlanner:
    """
    Coordinates all 4 agents autonomously.
    WhatsApp number can be:
      - passed per-call:  process_new_bill(..., phone_number="9876543210")
      - saved once:       save_user_whatsapp(user_id, "9876543210")
    """

    def __init__(self, db_path="agentic_data.db",
                 twilio_sid=None, twilio_token=None):
        self.db           = AgenticDB(db_path)
        self.bill_scanner = BillScannerAgent(self.db)
        self.trend_analyzer = TrendAnalyzerAgent(self.db)
        self.plan_adjuster  = PlanAdjusterAgent(self.db)
        self.alert_agent    = AlertAgent(self.db, twilio_sid, twilio_token)
        print("[PLANNER] ✅ Auto Investment Planner ready")

    def save_user_whatsapp(self, user_id: str, phone_number: str) -> Dict:
        """
        Save user's WhatsApp number once — all future alerts go here.
        Accepts: 9876543210 / +919876543210 / whatsapp:+919876543210
        """
        formatted = _format_whatsapp_number(phone_number)
        self.db.save_user_settings(user_id, whatsapp_number=formatted)
        return {
            "success": True,
            "saved_number": formatted,
            "message": f"WhatsApp number saved: {formatted}"
        }

    def add_manual_expense(self, user_id: str, category: str,
                           amount: float, description: str = "",
                           monthly_income: float = 15000,
                           phone_number: str = None) -> Dict:
        """
        User manually adds an expense like Shopping ₹2000 or Rent ₹5000.
        Then re-runs trend analysis and adjusts plan.
        """
        # Add the expense
        scan_result = self.bill_scanner.add_manual_expense(
            user_id, category, amount, description
        )
        # Re-analyze and adjust
        trends = self.trend_analyzer.analyze_trends(user_id, 3)
        plan   = self.plan_adjuster.adjust_plan(user_id, monthly_income, trends)
        alerts = self.alert_agent.check_and_alert(user_id, trends, plan, phone_number)

        return {
            "success": True,
            "expense_added": scan_result,
            "updated_plan": plan,
            "alerts": alerts,
            "message": f"₹{amount:,.0f} added to {category} successfully!"
        }

    def process_new_bill(self, user_id: str, bill_text: str,
                         monthly_income: float,
                         phone_number: str = None) -> Dict:
        """Full autonomous pipeline for a scanned bill."""
        print(f"\n{'='*55}\n[PLANNER] Processing bill for: {user_id}\n{'='*55}")

        scan   = self.bill_scanner.scan_and_categorize(user_id, bill_text)
        print(f"  ✅ Scanned: {scan['category']} ₹{scan['amount']}")

        trends = self.trend_analyzer.analyze_trends(user_id, 3)
        print(f"  ✅ Trends: {trends['overall_trend']['direction']}")

        plan   = self.plan_adjuster.adjust_plan(user_id, monthly_income, trends)
        print(f"  ✅ Plan adjusted: {plan['adjustment_reason']}")

        alerts = self.alert_agent.check_and_alert(user_id, trends, plan, phone_number)
        print(f"  ✅ Alerts: {alerts['alerts_sent']} sent")

        return {
            "success": True,
            "pipeline": {"bill_scan": scan, "trend_analysis": trends,
                         "adjusted_plan": plan, "alerts": alerts},
            "summary": {
                "bill_amount": scan["amount"],
                "bill_category": scan["category"],
                "spending_trend": trends["overall_trend"]["direction"],
                "spending_change_pct": trends["overall_trend"]["pct_change"],
                "new_rd_amount": plan["buckets"]["rd_savings"]["monthly"],
                "new_gold_amount": plan["buckets"]["digital_gold"]["monthly"],
                "alerts_sent": alerts["alerts_sent"],
                "insights": trends["insights"],
            },
        }

    def get_user_dashboard(self, user_id: str) -> Dict:
        month   = datetime.now().strftime("%Y-%m")
        summary = self.bill_scanner.get_monthly_summary(user_id, month)
        trends  = self.trend_analyzer.analyze_trends(user_id, 3)
        plan    = self.db.get_latest_plan(user_id)
        settings = self.db.get_user_settings(user_id)
        return {
            "user_id": user_id,
            "current_month": month,
            "monthly_summary": summary,
            "trends": trends,
            "current_plan": plan,
            "bills_this_month": summary["total_bills"],
            "total_spending": summary["total_spending"],
            "whatsapp_configured": bool(settings.get("whatsapp_number")),
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    planner = AutoInvestmentPlanner(db_path="test_demo.db")
    uid = "demo_user"

    # Save WhatsApp number once
    planner.save_user_whatsapp(uid, "9876543210")

    # Process a scanned bill
    planner.process_new_bill(uid, "Grand Total 450.00 | Food Bill | Rice Dal", 15000)

    # Add manual expenses
    planner.add_manual_expense(uid, "Shopping", 2000, "Clothes from market", 15000)
    planner.add_manual_expense(uid, "Utilities", 680, "Bijli bill", 15000)

    # Dashboard
    import json
    dash = planner.get_user_dashboard(uid)
    print(json.dumps(dash, indent=2, default=str))
