"""
rag_module.py — RAG Financial Advisor for NiveshSaathi
Uses ChromaDB (vector store) + Groq API (free LLM) + HuggingFace embeddings
Answers financial questions in Hindi for low-income users
"""

import os
import threading

_chroma_lock = threading.Lock()
_chroma_collection = None
import json
from pathlib import Path

# ── FINANCIAL KNOWLEDGE BASE ──────────────────────────────────────────────────
# All the financial knowledge in simple Hindi + English
# This is the "R" in RAG — the documents we retrieve from

FINANCIAL_KNOWLEDGE = [

    # ── RD / FD BASICS ────────────────────────────────────────────────────────
    {
        "id": "rd_basics_1",
        "topic": "Recurring Deposit RD",
        "content": """
        Recurring Deposit (RD) kya hai: RD ek bank savings scheme hai jisme aap har mahine ek fixed amount jama karte ho.
        Bank aapko 6.5% se 7.5% tak byaaj deta hai.
        RD bilkul safe hai — bank kisi bhi halat mein aapka paisa wapas karta hai.
        Minimum amount: SBI mein sirf Rs.100/mahine se shuru kar sakte ho.
        Duration: 6 mahine se 10 saal tak.
        RD mature hone par: aapka original paisa + byaaj milta hai.
        Example: Agar aap Rs.500/mahine 1 saal RD mein daalte ho, to 12 mahine baad Rs.6,175 milenge (original Rs.6,000 + Rs.175 byaaj).
        RD kahan khole: SBI, PNB, Post Office — sabse safe options hain.
        RD vs FD: RD mein monthly jama karte ho, FD mein ek baar mein poora paisa dete ho.
        Teacher, daily wage worker, kirana shopkeeper — sabke liye RD best hai.
        """,
        "keywords": ["rd", "recurring deposit", "baar baar", "monthly savings", "bank savings", "byaaj", "interest"]
    },
    {
        "id": "fd_basics_1",
        "topic": "Fixed Deposit FD",
        "content": """
        Fixed Deposit (FD) kya hai: FD mein aap ek baar mein puri raqam bank ko dete ho.
        Bank aapko 6% se 8% tak byaaj deta hai — RD se thoda zyada.
        FD bilkul safe hai — DICGC insurance se Rs.5 lakh tak guarantee hai.
        Minimum amount: Rs.1,000 se shuru.
        Duration: 7 din se 10 saal.
        Premature withdrawal: zarurat padne par paise nikal sakte ho par thodi penalty lagti hai.
        Senior citizens ko 0.5% extra byaaj milta hai.
        Tax: 40,000 se zyada byaaj par TDS katega.
        Best FD rates 2024: Small Finance Banks mein 8-9% milta hai.
        FD sirf usi paison ke liye jo 1-2 saal tak zarurat na ho.
        """,
        "keywords": ["fd", "fixed deposit", "ek baar", "lump sum", "bank mein daalo", "guaranteed return"]
    },

    # ── DIGITAL GOLD ──────────────────────────────────────────────────────────
    {
        "id": "digital_gold_1",
        "topic": "Digital Gold",
        "content": """
        Digital Gold kya hai: Aap 1 rupee se bhi sona khareed sakte ho digitally.
        Yahi asli sona hai — sirf locker ki zarurat nahi.
        Kahan kharidein: Paytm, PhonePe, Google Pay, Groww — sab par available.
        Price: Market rate pe milta hai — physical gold jaisa hi.
        Purity: 24 karat, 99.9% pure guaranteed.
        Storage: Company secure vault mein rakhti hai.
        Sell karna: Jab chahiye tab bech sakte ho, paisa turant account mein aata hai.
        Festival pe physical gold mein convert bhi kar sakte ho.
        Return: Gold historically 8-10% annual return deta hai.
        Risk: Gold price upar neeche hota hai — par long term mein safe hai.
        Minimum investment: Rs.1 — koi bhi shuru kar sakta hai.
        Teachers aur daily wage workers ke liye: Har mahine Rs.200-500 ka gold kharidna shuru karo.
        """,
        "keywords": ["gold", "sona", "digital gold", "paytm gold", "phonepe gold", "har mahine sona"]
    },

    # ── EMERGENCY FUND ────────────────────────────────────────────────────────
    {
        "id": "emergency_fund_1",
        "topic": "Emergency Fund",
        "content": """
        Emergency Fund kya hai: Aapke paas 3-6 mahine ke kharche ke barabar cash ready rehna chahiye.
        Kyun zaroori: Hospital, job jaana, ghar ki tameer, accident — koi bhi time aa sakta hai.
        Emergency fund savings account mein rakhein — FD ya investment mein nahi.
        Kitna chahiye: Monthly kharche ka 3 guna. Example: Rs.10,000/mahine kharche hain to Rs.30,000 emergency fund.
        Kahan rakhein: SBI ya Post Office savings account — easily nikal sake.
        Emergency fund ko kabhi invest mat karo — yeh aapka safety net hai.
        Ghar ka kiraya, bijli, khana — in sab ke liye 3 mahine ka paisa hona chahiye.
        Agar emergency fund nahi hai to pehle yahi banao — baad mein invest karo.
        """,
        "keywords": ["emergency", "achanak", "hospital", "zarurat", "cash", "safety net", "reserve"]
    },

    # ── INVESTMENT BASICS ─────────────────────────────────────────────────────
    {
        "id": "investment_basics_1",
        "topic": "Investment Basics for Beginners",
        "content": """
        Investment kya hai: Apna paisa aisi jagah lagana jo zyada paisa bana sake.
        Kyun invest karna chahiye: Sirf bank mein rakhne se inflation ki wajah se paison ki value kam hoti hai.
        Inflation matlab: Jo cheez aaj Rs.100 mein milti hai, 10 saal baad Rs.170 ki ho jaati hai.
        Isliye invest karo warna paisa ghatta hai.
        3 Bucket Rule: 
        Bucket 1 (Safety): 10% RD mein — guaranteed return.
        Bucket 2 (Growth): 5% Digital Gold mein — inflation se bachao.
        Bucket 3 (Emergency): Baaki bachat savings account mein.
        Kab shuru karein: Aaj se — chahe Rs.100 hi ho.
        Kitna invest karein: Jitna invest karo — woh amount 3 saal mein double hone ki chance hai.
        Common mistake: Log sochte hain bahut paisa chahiye invest karne ke liye — galat hai, Rs.100 se shuru karo.
        """,
        "keywords": ["invest kaise", "investment", "paisa lagana", "shuru kaise karein", "beginner", "pehli baar"]
    },

    # ── NIFTY / STOCKS ────────────────────────────────────────────────────────
    {
        "id": "stocks_basics_1",
        "topic": "Stocks and Mutual Funds Basics",
        "content": """
        Stock kya hai: Kisi company ka ek chhota sa hissa kharidna.
        Example: SBI ka ek share kharidne par aap SBI ke chhote se malik ban jaate ho.
        Index Fund kya hai: Ek sath 50 companies mein invest karna — Nifty 50 index fund.
        Index Fund kyun safe hai: Agar ek company doobi to baaki 49 bachayengi.
        Return: Nifty 50 ne historically 12% annual return diya hai.
        Mutual Fund: Professional log aapka paisa invest karte hain — SIP Rs.500/mahine se shuru.
        SIP kya hai: Systematic Investment Plan — har mahine fixed amount mutual fund mein jaata hai.
        Risk: Short term mein upar neeche — 5+ saal mein safe hai.
        Kahan shuru karein: Groww, Zerodha Coin, Paytm Money — sab free hain.
        Teacher ke liye suggestion: Rs.500/mahine Nifty 50 Index Fund SIP — 10 saal mein Rs.1.1 lakh bane Rs.1.2 lakh.
        Stocks SIRF emergency fund aur FD ke baad — agar extra paisa ho tab.
        """,
        "keywords": ["stock", "share", "mutual fund", "sip", "nifty", "index fund", "market mein lagana"]
    },

    # ── SAVINGS TIPS ──────────────────────────────────────────────────────────
    {
        "id": "savings_tips_1",
        "topic": "Savings Tips for Low Income",
        "content": """
        Kam income mein savings kaise karein — practical tips:
        Tip 1: Pay Yourself First — salary aate hi pehle RD mein daalo, baad mein kharch karo.
        Tip 2: 50-30-20 Rule: 50% zaroorat, 30% chahat, 20% savings.
        Tip 3: Kirana store ki weekly list banao — Rs.100-200 bachenge.
        Tip 4: LED bulbs lagao — bijli bill Rs.150-200 kam hoga.
        Tip 5: Jan Aushadhi store se generic medicines — 50-60% sasta.
        Tip 6: Monthly bus pass — daily auto se sasta.
        Tip 7: Mobile recharge plan 3 mahine ka lena — per month sasta padta hai.
        Tip 8: Ghar ka khana tiffin mein le jaao — Rs.500-1000/mahine bachega.
        Agar Rs.15,000 income hai: Rs.1,500 RD (10%) + Rs.750 gold (5%) + Rs.1,000 emergency = Rs.3,250 savings possible.
        """,
        "keywords": ["savings tips", "bachat kaise karein", "kam paison mein", "budget", "kharcha kam karo"]
    },

    # ── POST OFFICE SCHEMES ───────────────────────────────────────────────────
    {
        "id": "post_office_1",
        "topic": "Post Office Savings Schemes",
        "content": """
        Post Office schemes — government guaranteed, sabse safe:
        1. Post Office RD: Rs.100/mahine minimum, 6.7% byaaj, 5 saal.
        2. Post Office FD: 6.9% - 7.5% byaaj, 1-5 saal.
        3. PPF (Public Provident Fund): 7.1% byaaj, tax free return, 15 saal.
        4. NSC (National Savings Certificate): 7.7% byaaj, 5 saal, tax benefit.
        5. Sukanya Samriddhi: Beti ke liye — 8.2% byaaj, tax free.
        6. Kisan Vikas Patra: Paisa double — 7.5% byaaj.
        Post Office schemes ki khas baat: Government guarantee — kabhi doob nahi sakti.
        Gaon aur chhote sheher mein post office sabse aasaan option hai.
        PPF sabse accha long term option hai — 15 saal mein paisa lagbhag double hota hai.
        """,
        "keywords": ["post office", "ppf", "nsc", "sukanya", "sarkari yojana", "government scheme", "dak ghar"]
    },

    # ── UPI / DIGITAL PAYMENTS ────────────────────────────────────────────────
    {
        "id": "upi_digital_1",
        "topic": "UPI and Digital Payments",
        "content": """
        UPI kya hai: Mobile se seedha bank account se payment karna.
        Apps: PhonePe, Google Pay, Paytm, BHIM — sab free hain.
        Investment ke liye UPI: Directly bank account se RD, gold, ya mutual fund mein paisa ja sakta hai.
        UPI se digital gold kharidna: PhonePe ya Google Pay mein Gold section jaao.
        UPI se RD: SBI YONO app mein seedha RD khol sakte ho aur UPI se payment kar sakte ho.
        Safe hai ya nahi: Haan, NPCI (government) regulate karta hai — bilkul safe.
        Transaction limit: Rs.1 lakh per transaction by default.
        Koi charge nahi: UPI transactions free hain.
        """,
        "keywords": ["upi", "phonepe", "google pay", "paytm", "digital payment", "online payment"]
    },

    # ── INCOME TAX BASICS ─────────────────────────────────────────────────────
    {
        "id": "tax_basics_1",
        "topic": "Income Tax Basics",
        "content": """
        Income tax — aapko kitna dena hoga:
        Rs.3 lakh tak: Koi tax nahi (annual income).
        Rs.3-6 lakh: 5% tax (par rebate milti hai, effectively zero).
        Rs.6-9 lakh: 10% tax.
        Rs.9-12 lakh: 15% tax.
        Tax bachane ke tarike:
        Section 80C: Rs.1.5 lakh tak investment — PPF, ELSS, LIC — tax free.
        Section 80D: Health insurance premium — tax benefit.
        Standard Deduction: Rs.75,000 automatically milti hai salaried logo ko.
        Teacher ki salary Rs.15,000/mahine (Rs.1.8 lakh/saal) — koi tax nahi.
        PAN card zaroor banwao — banking aur investment ke liye zaroori.
        """,
        "keywords": ["tax", "income tax", "80c", "ppf tax", "tax bachana", "pan card"]
    },

    # ── BLOSTEM / FINTECH ─────────────────────────────────────────────────────
    {
        "id": "blostem_fintech_1",
        "topic": "Fintech and Digital Banking",
        "content": """
        Fintech kya hai: Technology se financial services — bank nahi jaana padta.
        Digital banking advantages: 24x7 available, ghar baithe sab kaam.
        Jan Dhan account: Zero balance account — koi bhi khol sakta hai.
        PMJJBY: Rs.330/saal mein Rs.2 lakh life insurance.
        PMSBY: Rs.20/saal mein Rs.2 lakh accident insurance — bahut sasta.
        Atal Pension Yojana: Rs.210/mahine par Rs.5,000/mahine pension milegi retirement mein.
        PM Mudra Loan: Chhote business ke liye Rs.10 lakh tak loan — bina guarantee ke.
        Loan types: Home loan, education loan, personal loan — bank se le sakte ho.
        Credit score: CIBIL score — 750+ hona chahiye loan ke liye.
        """,
        "keywords": ["jan dhan", "pmjjby", "atal pension", "mudra loan", "digital banking", "fintech"]
    },
]


# ── CHROMA DB SETUP ────────────────────────────────────────────────────────────
def setup_chromadb(persist_dir: str = "./rag_db"):
    """
    Initialize ChromaDB with financial knowledge.
    Thread-safe singleton — safe to call from multiple threads.
    Pin chromadb==0.4.24 in requirements.txt to avoid Rust backend errors.
    """
    global _chroma_collection
    with _chroma_lock:
        if _chroma_collection is not None:
            return _chroma_collection

        import chromadb

        print("[RAG] Setting up ChromaDB...")

        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Create persistent client
        client = chromadb.PersistentClient(path=persist_dir)

        # Build embedding function
        emb_fn = _get_embedding_function()

        # Create or get collection
        collection = client.get_or_create_collection(
            name="financial_knowledge",
            embedding_function=emb_fn,
            metadata={"description": "NiveshSaathi financial knowledge base"}
        )

        # Check if already populated
        if collection.count() >= len(FINANCIAL_KNOWLEDGE):
            print(f"[RAG] ChromaDB already has {collection.count()} documents ✅")
            _chroma_collection = collection
            return collection

        # Add all documents
        print(f"[RAG] Adding {len(FINANCIAL_KNOWLEDGE)} documents...")
        collection.add(
            ids=[doc["id"] for doc in FINANCIAL_KNOWLEDGE],
            documents=[doc["content"] for doc in FINANCIAL_KNOWLEDGE],
            metadatas=[{"topic": doc["topic"], "keywords": ",".join(doc["keywords"])} for doc in FINANCIAL_KNOWLEDGE]
        )
        print(f"[RAG] ChromaDB setup complete ✅ — {collection.count()} documents indexed")
        _chroma_collection = collection
        return collection


def _get_embedding_function():
    """
    Get embedding function — handles different chromadb versions.
    Uses multilingual model that supports Hindi.
    """
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"

    # Try chromadb 0.4.x style
    try:
        from chromadb.utils import embedding_functions
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        print(f"[RAG] Using embedding model: {model_name} (chromadb 0.4.x style)")
        return emb_fn
    except Exception:
        pass

    # Try chromadb 0.5.x style
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
        print(f"[RAG] Using embedding model: {model_name} (chromadb 0.5.x style)")
        return emb_fn
    except Exception:
        pass

    # Fallback: manual sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        class ManualEmbeddingFn:
            def __call__(self, input):
                return model.encode(input).tolist()

        print(f"[RAG] Using manual embedding function with {model_name}")
        return ManualEmbeddingFn()
    except Exception as e:
        raise RuntimeError(f"Could not load any embedding function: {e}")


# ── RETRIEVAL ──────────────────────────────────────────────────────────────────
def retrieve_context(query: str, collection, n_results: int = 3) -> str:
    """Retrieve relevant financial documents for a query"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    # Combine top results
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        topic = results["metadatas"][0][i].get("topic", "")
        context_parts.append(f"[{topic}]\n{doc.strip()}")

    return "\n\n---\n\n".join(context_parts)


# ── LLM — GROQ API (FREE) ─────────────────────────────────────────────────────
def ask_groq(question: str, context: str, profession: str = "user") -> str:
    """
    Use Groq API (free) with Llama 3 to generate Hindi answers.
    Get free API key at: console.groq.com (no credit card needed)
    """
    import requests

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        return _fallback_answer(question, context)

    system_prompt = f"""Aap NiveshSaathi ke AI financial advisor hain.
Aapka kaam hai {profession} ko simple Hindi mein financial guidance dena.

Rules:
- HAMESHA Hindi mein jawab do (Hinglish chalega)
- Simple bhasha use karo — jaise family member ko samjhate ho
- Specific numbers aur examples do (Rs. mein)
- Chhota aur clear jawab do — 3-5 sentences maximum
- Encourage karo — positively bolna
- Agar koi advanced cheez ho to simple analogy use karo

Context (yahi information use karo jawab ke liye):
{context}
"""

    user_message = f"Mera sawaal: {question}"

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",   # Free Groq model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            timeout=15
        )
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"[RAG] Groq error: {data}")
            return _fallback_answer(question, context)
    except Exception as e:
        print(f"[RAG] Groq API error: {e}")
        return _fallback_answer(question, context)


def _fallback_answer(question: str, context: str) -> str:
    """Rule-based fallback when Groq API not available"""
    q = question.lower()

    # Simple keyword-based responses
    if any(w in q for w in ["rd", "recurring", "deposit"]):
        return "RD (Recurring Deposit) mein aap bank mein har mahine ek fixed amount jama karte ho. Bank aapko 6.5-7% byaaj deta hai. SBI mein sirf Rs.100/mahine se shuru kar sakte ho — bilkul safe hai! 🏦"

    elif any(w in q for w in ["gold", "sona", "digital gold"]):
        return "Digital Gold mein aap Rs.1 se bhi sona khareed sakte ho PhonePe ya Google Pay se. Yahi asli 24 karat sona hai — sirf locker ki zarurat nahi. Festival pe physical gold mein convert bhi kar sakte ho! ✨"

    elif any(w in q for w in ["emergency", "achanak", "hospital"]):
        return "Emergency Fund bahut zaroori hai! Apne 3 mahine ke kharche ke barabar cash savings account mein rakhein. Example: Agar Rs.10,000/mahine kharche hain to Rs.30,000 emergency fund banao. Yeh aapka safety net hai! 🆘"

    elif any(w in q for w in ["invest", "paise lagana", "stock", "mutual fund"]):
        return "Investment ke liye 3 bucket rule follow karo: 10% RD mein (safe), 5% Digital Gold mein (growth), baaki emergency fund mein. Rs.100 se bhi shuru kar sakte ho — aaj hi shuru karo! 🌱"

    elif any(w in q for w in ["tax", "income tax", "80c"]):
        return "Agar aapki salary Rs.15,000/mahine (Rs.1.8 lakh/saal) hai to koi income tax nahi lagega! PPF aur ELSS mein invest karne se 80C ke under Rs.1.5 lakh tak tax bhi bachta hai. 📊"

    elif any(w in q for w in ["post office", "ppf", "nsc"]):
        return "Post Office schemes sabse safe hain — government guarantee hoti hai! PPF mein 7.1% byaaj milta hai aur yeh tax-free bhi hai. 15 saal mein paisa lagbhag double hota hai. Gaon mein bhi asaani se available hai! 🏤"

    else:
        # Extract first relevant sentence from context
        if context:
            lines = [l.strip() for l in context.split('\n') if len(l.strip()) > 30]
            if lines:
                return lines[0] + " Zyada jankari ke liye apne bank ya post office mein jaayein. 💡"

        return "Yeh ek achha sawaal hai! Investment ke baare mein: pehle emergency fund banao (3 mahine ka kharcha), phir RD shuru karo (10% income), phir digital gold (5% income). Chhote kadam se bada fark padta hai! 🌱"


# ── MAIN RAG FUNCTION ─────────────────────────────────────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        _collection = setup_chromadb()
    return _collection


def ask_financial_question(
    question: str,
    profession: str = "user",
    income: float = 15000,
    expenses: float = 11000
) -> dict:
    """
    Main RAG function: Question → Retrieve → Generate Answer
    Returns answer + source topics used
    """
    try:
        collection = get_collection()

        # Retrieve relevant context
        context = retrieve_context(question, collection, n_results=3)

        # Add user context to make answers personalized
        user_context = f"\nUser info: {profession}, monthly income Rs.{income:.0f}, expenses Rs.{expenses:.0f}, savings Rs.{income-expenses:.0f}/month"
        full_context = context + user_context

        # Generate answer with Groq LLM
        answer = ask_groq(question, full_context, profession)

        # Get source topics
        results = collection.query(query_texts=[question], n_results=3)
        sources = [m.get("topic", "") for m in results["metadatas"][0]] if results["metadatas"] else []

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "method": "groq_rag" if os.environ.get("GROQ_API_KEY") else "fallback_rag"
        }

    except Exception as e:
        print(f"[RAG] Error: {e}")
        return {
            "success": False,
            "question": question,
            "answer": "Maafi chahta hoon, abhi jawab nahi de pa raha. Kripya dobara try karein. 🙏",
            "sources": [],
            "method": "error"
        }


# ── TEST ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RAG module...\n")

    test_questions = [
        "RD kya hota hai?",
        "Mujhe digital gold mein invest karna chahiye?",
        "Emergency fund kitna hona chahiye?",
        "Post office schemes safe hain kya?",
        "Rs.15000 salary mein kaise bachat karein?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = ask_financial_question(q, profession="teacher", income=15000, expenses=11000)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)