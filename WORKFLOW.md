# Medical VQA System - Clean Workflow Architecture

## 🌳 System Tree (One Job Per Agent)

```
Medical VQA Pipeline
│
├── Translation Agent       (detect language → translate if needed)
├── Memory Manager         (RAM cache of conversations)
├── Router Agent           (LLM decides what to do)
├── Image Agent            (VQA models + 2-layer OOD)
├── PubMed Agent          (search medical literature)
├── Response Generator     (combine results → answer)
└── Session Manager        (save to disk)
```

---

## 🔄 Main Workflow (7 Steps)

```
┌─────────────────────────────────────────────────┐
│  USER INPUT: question + optional image          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  1. TRANSLATION      │
        │  • Detect language   │
        │  • Translate to EN   │
        │  • Skip if English   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  2. MEMORY           │
        │  (LangChain-style)   │
        │                      │
        │  ┌────────────────┐  │
        │  │ Memory Manager │  │
        │  │ (Custom impl)  │  │
        │  │                │  │
        │  │ RAM Cache:     │  │
        │  │ {              │  │
        │  │   session_1: [ │  │
        │  │     "User: Hi",│  │
        │  │     "AI: Hello"│  │
        │  │   ],           │  │
        │  │   session_2: […│  │
        │  │ }              │  │
        │  └────────────────┘  │
        │                      │
        │  • Get conversation  │
        │    history           │
        │  • Store in RAM      │
        │  • Pass to Router    │
        └──────────┬───────────┘
                   │
                   ▼
    ╔══════════════════════════════════════════════════════╗
    ║  3. ROUTER (Gemma-3-4B) ← THE BRAIN                 ║
    ║                                                      ║
    ║  Analyzes:                                           ║
    ║  • User message                                      ║
    ║  • has_image (True/False)                            ║
    ║  • Conversation history (last 5 turns from memory)   ║
    ║                                                      ║
    ║  Decides 3 things:                                   ║
    ║  ┌────────────────────────────────────────────────┐ ║
    ║  │ 1. needs_vqa? (True/False)                     │ ║
    ║  │    • If has_image=True AND question about      │ ║
    ║  │      image → TRUE                               │ ║
    ║  │    • If no image → FALSE                        │ ║
    ║  │    • If casual chat with image → FALSE          │ ║
    ║  └────────────────────────────────────────────────┘ ║
    ║  ┌────────────────────────────────────────────────┐ ║
    ║  │ 2. needs_pubmed? (True/False)                  │ ║
    ║  │    • If medical question → TRUE                │ ║
    ║  │    • If casual chat (thanks, hello) → FALSE    │ ║
    ║  │    • If modify request → FALSE                 │ ║
    ║  │    • If user info question → FALSE             │ ║
    ║  └────────────────────────────────────────────────┘ ║
    ║  ┌────────────────────────────────────────────────┐ ║
    ║  │ 3. response_mode (string)                      │ ║
    ║  │    • "medical_answer" - needs explanation      │ ║
    ║  │    • "casual_chat" - greeting/thanks           │ ║
    ║  │    • "modify_previous" - edit last response    │ ║
    ║  └────────────────────────────────────────────────┘ ║
    ╚══════════════════════════════════════════════════════╝
                           │
            ┌──────────────┴──────────────┐
            │                             │
        has_image?                    no_image?
            │                             │
            ▼                             ▼
    ┌───────────────────┐        ┌──────────────────┐
    │ 4a. IMAGE AGENT   │        │ needs_pubmed?    │
    │                   │        │                  │
    │ • Layer 1 OOD     │        │ TRUE → Go to 4b  │
    │ • VQA Model       │        │ FALSE → Skip 4b  │
    │ • Layer 2 OOD     │        │                  │
    │                   │        │ Goes directly to │
    │ Output:           │        │ Step 5           │
    │ "pneumonia in     │        └──────┬───────────┘
    │  right lung..."   │               │
    └─────────┬─────────┘               │
              │                         │
              │ VQA output passed       │
              │ directly to PubMed!     │
              ▼                         │
    ┌───────────────────┐               │
    │ 4b. PUBMED AGENT  │◄──────────────┘
    │ (if needs_pubmed) │
    │                   │
    │ Search query:     │
    │ • From VQA output │
    │   (if exists)     │
    │ • From user Q     │
    │   (if no VQA)     │
    │                   │
    │ Example:          │
    │ • VQA: "pneumonia"│
    │   → Search:       │
    │   "pneumonia lung"│
    │ • No VQA: "diabetes"│
    │   → Search:       │
    │   "diabetes def"  │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ 5. RESPONSE GEN   │
    │ (Gemma-3-4B)      │
    │                   │
    │ Combines:         │
    │ • VQA (if any)    │
    │ • PubMed (if any) │
    │ • Memory context  │
    └───────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  6. TRANSLATION      │
       │  Translate back to   │
       │  user's language     │
       │  (skip if English)   │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  7. SAVE             │
       │  • Memory (RAM)      │
       │  • Session (Disk)    │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  RETURN TO USER      │
       └──────────────────────┘
```

---

## 🎯 Router Decision Tree (The Brain)

```
                    ROUTER AGENT
                    (analyzes message + image + history)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    has_image?       medical_q?         casual?
         │                 │                 │
    ┌────┴────┐       ┌────┴────┐           │
    │         │       │         │           │
   YES       NO      YES       NO           YES
    │         │       │         │           │
    ▼         │       ▼         │           ▼
needs_vqa     │   needs_       │       response_mode
  = TRUE      │   pubmed       │       = "casual_chat"
              │   = TRUE       │
              │                │
              └────────┬───────┘
                       │
                       ▼
              Execute based on flags
```

---

## 🛡️ Image Agent (2-Layer OOD Protection)

```
Image Upload
    │
    ▼
┌─────────────────────────────┐
│ LAYER 1: Classifier OOD     │
│ (Statistical checks)         │
│                              │
│ Checks:                      │
│ • MSP < 80%?      → REJECT   │
│ • Entropy > 0.55? → REJECT   │
│ • Energy > -2.0?  → REJECT   │
└─────────────┬───────────────┘
              │ PASS
              ▼
┌─────────────────────────────┐
│ Run VQA Model               │
│ (Qwen2-VL or Qwen3-VL)      │
│                              │
│ Input: image + question      │
│ Output: text answer          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ LAYER 2: Semantic OOD       │
│ (Text analysis)              │
│                              │
│ Check VQA answer for:        │
│ • "person", "face" → REJECT  │
│ • "car", "building" → REJECT │
│ • "landscape" → REJECT       │
│ • Has medical terms? → PASS  │
└─────────────┬───────────────┘
              │ PASS
              ▼
         VQA Answer ✓
```

---

## 📋 Example Flow (Medical Image Upload)

```
Input: [Chest X-ray] + "What do you see?"

1. Translation
   Detect: en (English)
   Skip translation ✓

2. Memory
   New session → Create

3. Router
   has_image: TRUE
   Mode: medical_answer
   needs_vqa: TRUE  ← Analyze image first!
   needs_pubmed: TRUE
   search: (will use VQA answer)

4a. Image Agent (RUNS FIRST!)
   Layer 1 OOD: MSP=92% → PASS ✓
   VQA Model: "pneumonia in right lung"
   Layer 2 OOD: Has "pneumonia" → PASS ✓
   
   Output: "pneumonia in right lung"

4b. PubMed Agent (RUNS SECOND, uses VQA answer!)
   Search query: "pneumonia in right lung"  ← Uses VQA output!
   Found: 5 articles
   Top: "Pneumonia diagnosis", "Right lung infection"...

5. Response Generator
   VQA answer: "pneumonia in right lung"
   PubMed: 5 articles about pneumonia
   
   Combined answer:
   "The X-ray shows pneumonia in the right lung [1].
    This is typically caused by bacterial infection [2]..."

6. Translation
   Skip (English) ✓

7. Save
   User: "[Image Uploaded]"
   AI: "The X-ray shows pneumonia..."

Return: Complete medical answer with VQA + literature
```

---

## 📋 Example Flow (Text-Only Medical Question)

```
Input: "What is diabetes?"

1. Translation
   Detect: en (English)
   Skip translation ✓

2. Memory
   New session → Create

3. Router
   has_image: FALSE
   Mode: medical_answer
   needs_vqa: FALSE  ← No image!
   needs_pubmed: TRUE
   search: "diabetes definition"

4a. Image Agent
   SKIPPED (no image) ✗

4b. PubMed Agent
   Search query: "diabetes definition"  ← Uses user question!
   Found: 5 articles
   Top: "Diabetes mellitus overview"...

5. Response Generator
   VQA answer: None
   PubMed: 5 articles about diabetes
   
   Answer:
   "Diabetes is a chronic disease that affects how your
    body processes blood sugar [1]..."

6. Translation
   Skip (English) ✓

7. Save & Return
```

---


### **How Memory Flows Through System**

```
┌─────────────────────────────────────────────────────┐
│  Step 2: Memory Manager                             │
│                                                      │
│  1. Check RAM cache:                                │
│     active_sessions = {                             │
│       session_1: InMemoryConversation([...]),       │
│       session_2: InMemoryConversation([...])        │
│     }                                                │
│                                                      │
│  2. If session exists in RAM:                       │
│     → Use cached conversation ✓                     │
│                                                      │
│  3. If NOT in RAM:                                  │
│     → Load from disk (session_manager.json)         │
│     → Restore to RAM                                │
│     → Create InMemoryConversation object            │
│                                                      │
│  4. Extract last 5 turns:                           │
│     [                                                │
│       "User: What is diabetes?",                    │
│       "AI: Diabetes is...",                         │
│       "User: What causes it?",                      │
│       "AI: It's caused by...",                      │
│       "User: Tell me more"  ← Current message       │
│     ]                                                │
│                                                      │
│  5. Pass to Router:                                 │
│     Router uses this context to understand          │
│     what "it" refers to (diabetes)                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Step 3: Router (uses memory context)               │
│                                                      │
│  Router receives:                                   │
│  • Current message: "Tell me more"                  │
│  • Memory context: [last 5 turns above]             │
│  • has_image: False                                 │
│                                                      │
│  Router thinks:                                     │
│  "User said 'Tell me more' - more about what?"      │
│  "Looking at history... they asked about diabetes"  │
│  "So this is a follow-up medical question"          │
│                                                      │
│  Decision:                                          │
│  • needs_vqa = FALSE (no image)                     │
│  • needs_pubmed = TRUE (medical follow-up)          │
│  • search_query = "diabetes causes" (from context)  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Step 7: Save back to memory                        │
│                                                      │
│  After generating response:                         │
│                                                      │
│  1. Add to RAM:                                     │
│     memory.add_user_message("Tell me more")         │
│     memory.add_ai_message("Diabetes is caused...")  │
│                                                      │
│  2. Save to disk:                                   │
│     session_manager.save(session_id, {...})         │
│                                                      │
│  Now conversation history has 6 turns!              │
└─────────────────────────────────────────────────────┘
```

### **Why This Matters**

```
WITHOUT Memory:
User: "What is diabetes?"
AI: "Diabetes is a chronic disease..."

User: "What causes it?"
AI: "What causes what?" ❌ No context!

---

WITH Memory:
User: "What is diabetes?"
AI: "Diabetes is a chronic disease..."
  [Saved to memory]

User: "What causes it?"
Router sees history: "it" = diabetes
AI: "Diabetes is caused by..." ✓ Understands context!
```

### **Memory Storage (Two Layers)**

```
Layer 1: RAM (Fast, Temporary)
┌────────────────────────────┐
│ active_sessions = {        │
│   1: InMemoryConversation, │  ← Fast access
│   2: InMemoryConversation, │
│   3: InMemoryConversation  │
│ }                          │
└────────────────────────────┘

Layer 2: Disk (Persistent)
┌────────────────────────────┐
│ sessions/                  │
│ ├── user1/                 │
│ │   ├── 1/                 │  ← Survives restart
│ │   │   └── session.json   │
│ │   └── 2/                 │
│ └── user2/                 │
└────────────────────────────┘

When API restarts:
- RAM cleared (all InMemoryConversation objects lost)
- Disk persists (session.json files remain)
- On first message: restore from disk to RAM
```

---

## 📋 Example Flow (Casual Chat - No PubMed)

```
Input: "Thanks for your help!"

1. Translation
   Detect: en (English)
   Skip translation ✓

2. Memory
   Load existing session

3. Router
   has_image: FALSE
   Mode: casual_chat  ← Just chatting!
   needs_vqa: FALSE
   needs_pubmed: FALSE  ← No medical info needed!
   search: null

4a. Image Agent
   SKIPPED (no image) ✗

4b. PubMed Agent
   SKIPPED (needs_pubmed=FALSE) ✗

5. Response Generator
   VQA answer: None
   PubMed: None
   Mode: casual_chat
   
   Answer:
   "You're welcome! Happy to help. 😊"

6. Translation
   Skip (English) ✓

7. Save & Return
```

---

## 📋 Router Decision Examples

| User Input | has_image | needs_vqa | needs_pubmed | Why? |
|-----------|-----------|-----------|--------------|------|
| [X-ray] + "What is this?" | TRUE | TRUE | TRUE | Medical image question |
| [X-ray] (no text) | TRUE | TRUE | TRUE | Image needs analysis |
| "What is diabetes?" | FALSE | FALSE | TRUE | Medical text question |
| "Thanks!" | FALSE | FALSE | FALSE | Casual chat |
| "Hello" | FALSE | FALSE | FALSE | Greeting |
| "Remove references" | FALSE | FALSE | FALSE | Modify previous |
| "Tell me more" | FALSE | FALSE | TRUE | Follow-up medical Q |

---

## 📋 Example Flow (Khmer Medical Question)

```
Input: "តើជំងឺទឹកនោមផ្អែមជាអ្វី?" (What is diabetes in Khmer)

1. Translation
   Detect: km (Khmer)
   Translate to: "What is diabetes?"

2. Memory
   New session → Create

3. Router
   Mode: medical_answer
   needs_vqa: FALSE (no image)
   needs_pubmed: TRUE
   search: "diabetes definition"

4a. Image: SKIP (no image)

4b. PubMed
   Search: "diabetes definition"
   Found: 5 articles
   Top: [54%, 51%, 50%]

5. Response Generator
   Combines PubMed articles
   Generates answer with citations

6. Translation
   Translate to: km (Khmer)
   Output: "ជំងឺទឹកនោមផ្អែម គឺជាជំងឺរ៉ាំរ៉ៃ..."

7. Save
   RAM: User + AI messages (English)
   Disk: User "[Question]" + AI (Khmer)

Return: Khmer response with references
```

---

## 🎨 Key Points

### ✅ Clean Architecture
- **1 file = 1 job**
- Easy to debug
- Easy to modify

### ✅ Smart Routing
- LLM decides everything
- No hardcoded rules
- Context-aware

### ✅ Translation Optimization
- Skip English (70% of users)
- Deep-translator (free)
- Fast (0ms for English, 500ms for others)

### ✅ 2-Layer OOD
- Layer 1: Statistics (fast, catches obvious)
- Layer 2: Semantics (smart, catches tricky)
- Both must pass

### ✅ Models Used
| Model | Job | Size |
|-------|-----|------|
| Gemma-3-4B | Router + Response | 4B |
| Qwen2-VL-7B | PathVQA images | 7B |
| Qwen3-VL-2B | VQA-RAD images | 2B |
| deep-translator | Translation | N/A (API) |

---

## 🚨 Current Issues (To Fix)

1. **OOD not rejecting faces**
   - Problem: Face images passing through
   - Location: image_agent.py Layer 2

2. **Translation issues**
   - Problem: Short text detection ("good job" → Somali)
   - Location: translation_agent.py

---

## 📊 File Structure

```
agents/
├── router_agent.py          (50 lines)  ← Routing decisions
├── response_generator.py    (80 lines)  ← Generate answers
├── memory_manager.py        (60 lines)  ← RAM conversations
├── image_agent.py           (400 lines) ← VQA + OOD
├── pubmed_agent.py          (200 lines) ← Search papers
├── translation_agent.py     (150 lines) ← Translate
└── session_manager.py       (150 lines) ← Disk storage

main.py                      (300 lines) ← Coordinates all
api_refactored.py            (400 lines) ← FastAPI endpoints
auth.py                      (150 lines) ← JWT auth
```

**Total: ~2000 lines** (was 4000+ before refactoring!)

---

This is your complete system in a nutshell! 🎯
