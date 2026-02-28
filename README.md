# BluCare+

AI-powered clinical intelligence platform with conversational diagnosis, hybrid RAG retrieval, severity classification, and safety-aware medical guidance.

---

##  Run Backend

```bash
cd /workspaces/ragblucare
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

##  Run Frontend (Landing)

```bash
cd frontend
npm install
npm run dev
```

---

##  Core Capabilities

- Multi-phase conversational diagnostic engine  
- Hybrid symptom extraction (rule-based + LLM refinement)  
- FAISS-powered RAG over structured medical knowledge base  
- Weighted disease scoring (semantic + overlap + red-flag + prevalence)  
- Hybrid severity classification with emergency escalation  
- Strict OTC-only medication policy with hard-blocked Rx drugs  
- Region-aware home remedies and dietary suggestions  
- Intelligent lab test recommendations  
- Firebase session persistence with in-memory fallback  
- PDF consultation report generation  
- Nearby hospital and ambulance integration  

---

## Tech Stack

### Backend
- FastAPI  
- Uvicorn  
- SentenceTransformers (E5 embeddings)  
- FAISS  
- Groq API (primary LLM)  
- OpenRouter API (fallback LLM)  
- Firebase Firestore  
- ReportLab  

### Frontend
- React + Vite  
- GSAP animations  
- Vanilla JS Chat SPA  
- Server-Sent Events (SSE) streaming  

---

BluCare+ is designed as a clinical reasoning engine focused on safety, explainability, and structured medical intelligence.
