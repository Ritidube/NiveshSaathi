---
title: NiveshSaathi
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# 🌱 NiveshSaathi — पैसा बढ़ाओ | AI Investment Advisor for Bharat

> *"Har Indian deserves financial guidance — not just the rich ones."*

---

## 💡 The Idea — Why I Built This

India has over 500 million people earning below ₹20,000 a month — daily wage workers, small shopkeepers, teachers, auto drivers. These people work hard, but nobody ever taught them how to save or invest. A financial advisor? That's only for people with lakhs. A bank app? Too complicated, all in English.

I built **NiveshSaathi** (meaning "Investment Friend" in Hindi) because I wanted to change that. The idea is simple — what if a daily wage worker could just click a photo of their grocery bill, and the app automatically figures out where their money is going, tells them how much they can save, and sends them a WhatsApp message when they're overspending?

No jargon. No complicated forms. Just simple Hindi guidance, like a friend who happens to know finance.

---

## 🎯 What This App Does

**1. Scan Your Bills (OCR)**
You click a photo of any bill — grocery, medical, electricity. The app reads it using EasyOCR (supports Hindi + English), extracts the amounts, and automatically categorizes spending into buckets like Food, Healthcare, Transport, Utilities etc.

**2. Track Spending Trends (Agentic AI)**
Every bill you scan gets stored. Over time the app builds a picture of your spending habits. It compares this month vs last month and detects when something is going up too fast — like if your medical bills suddenly spike.

**3. Investment Recommendations (3-Bucket System)**
Based on your income and expenses, the app suggests splitting savings into 3 buckets:
- 🏦 **RD (Recurring Deposit)** — Safe, guaranteed returns at your bank (6.5-7.5%)
- ✨ **Digital Gold** — Buy gold digitally in small amounts, no locker needed
- 🆘 **Emergency Fund** — Keep liquid cash for sudden needs

**4. WhatsApp Alerts (Twilio)**
When you overspend your budget or a category spikes suddenly, you get a WhatsApp message in Hindi — not an email nobody reads, but a WhatsApp message that actually reaches you.

**5. Hindi Q&A with AI (RAG System)**
Ask anything in Hindi — "RD kya hota hai?", "Post Office mein paisa kaise lagayen?", "Tax kaise bachate hain?" — and get a real answer in simple Hindi, powered by Groq's free Llama 3.

**6. Growth Visualizer**
Your savings journey is shown as a growing plant — seedling when you just started, full tree when you've been saving consistently. Makes saving feel motivating.

**7. Live Market Data**
Live prices for Nifty50, Gold (GOLDBEES), and SBI Bank — so users can see what's happening in the market.

---

## 🏗️ Project Structure

```
NiveshSaathi/
│
├── app.py                  ← Main FastAPI server. All API routes live here.
│
├── agentic_api.py          ← FastAPI router for agentic AI features.
│                             Handles /api/agentic/* endpoints.
│
├── agentic_planner.py      ← The brain. Contains 4 autonomous agents:
│                             1. BillScannerAgent — categorizes bills
│                             2. TrendAnalyzerAgent — month-over-month comparison
│                             3. PlanAdjusterAgent — recalculates investment buckets
│                             4. AlertAgent — sends WhatsApp via Twilio
│
├── rag_module.py           ← RAG system. Stores financial knowledge in ChromaDB.
│                             Uses Groq free Llama 3 to answer Hindi questions.
│
├── gan_module.py           ← Growth visualization. Uses pre-generated PNGs.
│                             16 images (4 professions x 4 growth stages).
│
├── requirements.txt        ← All Python dependencies with pinned versions.
│
├── Dockerfile              ← Container setup for HuggingFace Spaces.
│
├── .env.example            ← Template showing which secrets you need.
│
├── .gitignore              ← Keeps .env, venv, and DB out of git.
│
├── keep_alive.py           ← Daily ping to keep Twilio sandbox active.
│
├── static/
│   ├── index.html          ← Complete frontend (single HTML file, vanilla JS).
│   ├── sw.js               ← Service worker for offline capability.
│   └── gan_cache/          ← 16 pre-generated PNG images (committed to git).
│
├── rag_db/                 ← ChromaDB vector DB (auto-created, NOT in git).
│
└── agentic_data.db         ← SQLite DB for bills/trends/alerts (NOT in git).
```

---

## 🧰 Tech Stack — What I Used and Why

### Backend
| Package | Version | Why |
|---|---|---|
| **FastAPI** | 0.109.0 | Modern async Python API framework. Auto-generates docs. Much faster than Flask. |
| **Uvicorn** | 0.27.0 | ASGI server that runs FastAPI. Handles concurrent requests. |
| **python-dotenv** | 1.0.1 | Loads `.env` file automatically so API keys never get hardcoded in code. |

### OCR
| Package | Version | Why |
|---|---|---|
| **EasyOCR** | 1.7.1 | Supports Hindi + English. Works offline. Much easier than Tesseract. |
| **OpenCV** | 4.9.0.80 | Image preprocessing. Pinned to 4.9.x — newer versions require NumPy 2.x which breaks ChromaDB. |
| **NumPy** | 1.26.4 | Pinned strictly — this exact version works with both EasyOCR and ChromaDB 0.4.24. |
| **Pillow** | 10.2.0 | Image loading and manipulation. |

### RAG (Hindi Q&A)
| Package | Version | Why |
|---|---|---|
| **ChromaDB** | 0.4.24 | Local vector database. Pinned to 0.4.24 — newer versions have a Rust backend that breaks on NumPy 1.x. |
| **sentence-transformers** | 2.3.1 | Creates Hindi-compatible embeddings using multilingual MiniLM model. Free, runs locally. |
| **tokenizers** | 0.20.3 | Required by sentence-transformers. Pinned to match ChromaDB 0.4.24 compatibility. |
| **Groq API** | via requests | Free LLM API using Llama 3. 30 req/min free tier. No GPU needed. |

### Alerts
| Package | Version | Why |
|---|---|---|
| **Twilio** | 8.10.0 | WhatsApp messaging API. Free sandbox for development and testing. |

### Data & Market
| Package | Version | Why |
|---|---|---|
| **yfinance** | 0.2.36 | Free Yahoo Finance data. Live Nifty50, Gold, SBI prices. No API key needed. |
| **scikit-learn** | 1.3.2 | Trend analysis and spending pattern calculations. |

### Frontend
Pure **Vanilla HTML + CSS + JavaScript** — single `index.html` file. No React, no Node.js, no build step. Works on cheap Android phones, works offline via service worker.

---

## 🤖 How the RAG System Works

RAG = Retrieval-Augmented Generation. Here is the exact flow:

```
User asks: "Post Office mein RD kaise kholein?"
         ↓
Step 1 — EMBED
Convert question to a 384-dimension vector using
paraphrase-multilingual-MiniLM-L12-v2 (supports Hindi perfectly)
         ↓
Step 2 — RETRIEVE
Search ChromaDB for top 3 most similar documents from our
knowledge base of 11 financial documents covering RD, FD,
Digital Gold, Emergency Fund, Tax, Post Office savings etc.
         ↓
Step 3 — AUGMENT
Build a prompt: "Answer this using these 3 documents as context.
Answer in simple Hindi. User is a {profession}."
         ↓
Step 4 — GENERATE
Send to Groq API (free Llama 3) → get Hindi answer
         ↓
User gets a proper answer with specific numbers and examples
```

The `rag_db/` folder is NOT committed to git — it auto-creates fresh on first startup (about 30 seconds). The embedding model (~90MB) downloads automatically from HuggingFace on first run.

---

## 🌳 How Growth Visuals Work

The app shows savings progress as a growing plant with 4 stages:
- 🌱 Seedling — 1 to 3 months saving
- 🌿 Sapling — 4 to 6 months
- 🌳 Young Tree — 7 to 12 months
- 🌲 Full Tree — more than 12 months

Combined with 4 professions (default, teacher, shopkeeper, daily_wage) = **16 PNG images** pre-generated and stored in `static/gan_cache/`.

`gan_module.py` just picks the right image. No GPU, no AI at runtime — just serving a cached PNG. This keeps the app fast on free servers.

The module is named "GAN" because the original plan was Stable Diffusion-generated images, but that needs ~2GB of model weights which isn't practical for free deployment. Cached PNGs work perfectly.

---

## 📱 WhatsApp Alerts — Twilio Setup

> ⚠️ **Free Sandbox Limitation — Please Read**

NiveshSaathi uses Twilio's free WhatsApp sandbox. The sandbox has one important restriction — **it can only send messages to phone numbers that have manually joined the sandbox first.** This is fine for development and demos. For a real production app with real users, you would need Twilio's paid WhatsApp Business API.

### Step by Step Twilio Setup

**Step 1 — Create Twilio account**
Go to [twilio.com/try-twilio](https://twilio.com/try-twilio) → Sign up free → Verify your phone number

**Step 2 — Get your credentials**
- Twilio Console Dashboard → copy **Account SID** (starts with `AC...`)
- Dashboard → click the eye icon → copy **Auth Token**
- Add both to your `.env` file

**Step 3 — Set up WhatsApp Sandbox**
- Twilio Console → **Messaging** → **Try it out** → **Send a WhatsApp message**
- Note your sandbox keyword (something like `join silver-tiger`)

**Step 4 — Join the sandbox from your phone**
- Save **+1 415 523 8886** as a contact in WhatsApp
- Send your join message (e.g. `join silver-tiger`) to that number
- You will get a confirmation reply ✅

**Step 5 — Keep it active**
The sandbox expires after **72 hours of no messages**. Fix options:
- Trigger a bill scan every day, OR
- Run `python keep_alive.py` daily to send an automatic ping

```powershell
# Schedule daily at 9AM (run PowerShell as Administrator)
$action = New-ScheduledTaskAction -Execute "D:\NivshShathi\venv\Scripts\python.exe" -Argument "D:\NivshShathi\keep_alive.py" -WorkingDirectory "D:\NivshShathi"
$trigger = New-ScheduledTaskTrigger -Daily -At "9:00AM"
Register-ScheduledTask -TaskName "NiveshSaathi_KeepAlive" -Action $action -Trigger $trigger -RunLevel Highest
```

---

## 🔑 API Keys You Need

| Key | Where to Get | Cost |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | **Free** — 30 req/min, no card needed |
| `TWILIO_ACCOUNT_SID` | [twilio.com](https://twilio.com/try-twilio) | **Free** sandbox |
| `TWILIO_AUTH_TOKEN` | Same Twilio dashboard | **Free** |

---

## 💻 Local Setup — Complete Guide

### Requirements
- Python 3.11
- Windows 10/11 (or Linux/Mac)
- 4GB RAM minimum
- Internet for first run (downloads ~600MB of models)

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/NiveshSaathi.git
cd NiveshSaathi
```

### Step 2 — Create virtual environment
```powershell
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```powershell
pip install -r requirements.txt
```
Takes 5-10 minutes. Downloads EasyOCR, ChromaDB, sentence-transformers, Twilio etc.

### Step 4 — Set up your .env file
```powershell
copy .env.example .env
```
Open `.env` and fill in your real keys:
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 5 — Run the app
```powershell
python app.py
```

Expected output:
```
[GAN] Device: cpu | GPU: False
[APP] GAN module loaded ✅
[APP] RAG module loaded ✅
INFO: Uvicorn running on http://0.0.0.0:7860
[RAG] Setting up ChromaDB...
[RAG] Adding 11 documents...
[RAG] ChromaDB setup complete ✅ — 11 documents indexed
INFO: Application startup complete.
```

### Step 6 — Open the app
Go to **http://localhost:7860** ✅

### Notes for first run
- `rag_db/` folder creates automatically (~30 seconds)
- EasyOCR downloads Hindi language model on first bill scan (~500MB, one time only)
- After first run, startup takes under 5 seconds

---

## 🤗 HuggingFace Spaces Deployment

HuggingFace gives you a **free public URL** with Docker. Works on college WiFi — it's just a git push.

### Step 1 — Push code to GitHub first

```powershell
cd D:\NivshShathi

# First time git setup
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

git init
git add .

# IMPORTANT: verify .env does NOT appear in this list
git status

git commit -m "NiveshSaathi initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NiveshSaathi.git
git push -u origin main
```

### Step 2 — Create a HuggingFace Space
- Go to [huggingface.co](https://huggingface.co) → Sign up free
- Click your photo → **New Space**
- **Space name:** NiveshSaathi
- **SDK:** Docker ← must be Docker, not Gradio
- **Visibility:** Public
- Click **Create Space**

### Step 3 — Add your secret keys
- Space page → **Settings** tab → **Variables and secrets**
- Click **New secret** for each:
```
TWILIO_ACCOUNT_SID  →  your value
TWILIO_AUTH_TOKEN   →  your value
GROQ_API_KEY        →  your value
```

### Step 4 — Get HuggingFace Access Token
- HuggingFace → your photo → **Settings** → **Access Tokens**
- **New token** → name: `deploy` → role: **Write** → Generate
- Copy the token (starts with `hf_...`) — this is your git password for HF

### Step 5 — Push to HuggingFace
```powershell
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/NiveshSaathi
git push space main
```
When asked:
- Username: your HuggingFace username
- Password: paste the `hf_...` token

### Step 6 — Watch the build
- Go to your Space → click **Logs** tab
- First build takes 10-15 minutes
- When you see `Application startup complete` → live ✅

Your URL: `https://YOUR_HF_USERNAME-NiveshSaathi.hf.space`

### Future updates
```powershell
git add .
git commit -m "your change description"
git push origin main   # update GitHub
git push space main    # update HuggingFace
```

---

## 🐳 Docker — Local Setup

### Step 1 — Install Docker Desktop
Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) → Install → Restart PC → Wait for Docker Desktop to show green in taskbar

### Step 2 — Build
```powershell
cd D:\NivshShathi
docker build -t niveshsaathi .
```
First build: 10-15 mins. Rebuilds: 1-2 mins (cached layers).

### Step 3 — Run
```powershell
docker run -d `
  -p 7860:7860 `
  -e TWILIO_ACCOUNT_SID=your_sid_here `
  -e TWILIO_AUTH_TOKEN=your_token_here `
  -e GROQ_API_KEY=your_groq_key_here `
  -v D:\NivshShathi\rag_db:/app/rag_db `
  -v D:\NivshShathi\agentic_data.db:/app/agentic_data.db `
  --name niveshsaathi `
  --restart unless-stopped `
  niveshsaathi
```

Visit **http://localhost:7860** ✅

### Useful commands
```powershell
docker logs -f niveshsaathi      # live logs
docker stop niveshsaathi         # stop
docker start niveshsaathi        # start
docker restart niveshsaathi      # restart
docker ps                        # see running containers
```

### Rebuild after code changes
```powershell
docker stop niveshsaathi && docker rm niveshsaathi
docker build -t niveshsaathi .
docker run -d ... (same run command)
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/health` | Health check |
| POST | `/api/scan-bill` | Upload bill image → categories + amounts |
| POST | `/api/recommendations` | 3-bucket investment plan |
| GET | `/api/market-data` | Live Nifty50, Gold, SBI prices |
| POST | `/api/generate-visual` | Growth tree image |
| GET | `/api/rd-calculator` | RD maturity calculator |
| POST | `/api/ask` | Hindi Q&A via RAG + Groq |
| GET | `/api/rag-status` | RAG system status |
| POST | `/api/agentic/save-whatsapp` | Save user WhatsApp number |
| POST | `/api/agentic/process-bill` | Full pipeline: scan → trends → alert |
| POST | `/api/agentic/add-manual-expense` | Add expense manually |
| GET | `/api/agentic/dashboard/{user_id}` | User financial dashboard |
| GET | `/api/agentic/status` | Agentic system status |

---

## 🐛 Common Issues & Fixes

**ChromaDB `KeyError: '_type'`**
Your `rag_db/` was made by a different ChromaDB version. Delete the `rag_db/` folder and restart — it recreates automatically.

**`np.float_` removed error**
NumPy 2.x is installed. Fix: `pip install "numpy==1.26.4" --force-reinstall`

**WhatsApp messages not arriving**
Twilio sandbox session expired (72 hour limit). Send your join message again to `+1 415 523 8886` on WhatsApp.

**"Maafi chahta hoon" response from AI**
Either `GROQ_API_KEY` is missing in `.env`, or ChromaDB hasn't finished initializing. Check `.env` and wait 30 seconds after startup.

**EasyOCR slow on first bill scan**
Normal — downloading Hindi language model (~500MB) on first use. Subsequent scans are fast.

**`collections.topic` SQLite error**
Version mismatch between ChromaDB that created the db and the one running now. Delete `rag_db/` folder.

---

## 🔮 Roadmap

- [ ] UPI integration for actual investment execution
- [ ] Multi-language support (Tamil, Telugu, Bengali)
- [ ] WhatsApp Business API (remove sandbox limitation)
- [ ] Family account support
- [ ] Loan EMI tracker and debt payoff planner
- [ ] Voice input in Hindi

---

## 👨‍💻 Built for Bharat

This project was built because millions of Indians have no access to financial guidance. NiveshSaathi tries to bridge that gap using AI, in the language people actually speak, delivered on the platform they actually use — WhatsApp.

If you find this useful, give it a ⭐ on GitHub!
