https://huggingface.co/datasets/flaviagiammarino/path-vqa
https://huggingface.co/datasets/flaviagiammarino/vqa-rad

┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Any Language)                 │
│                  question + optional image                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Translation Agent                        │
│  • Detect language (langdetect + Khmer check)               │
│  • Translate to English if needed                           │
│  • Returns: original_language + english_question            │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Route by Input  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                                 │
            ▼                                 ▼
    ┌──────────────┐                  ┌──────────────┐
    │ IMAGE INPUT  │                  │  TEXT ONLY   │
    └──────┬───────┘                  └──────┬───────┘
           │                                 │
           ▼                                 ▼
    ┌─────────────────────────┐      ┌─────────────────────┐
    │    Image Agent          │      │  Text Only Agent    │
    │ ┌─────────────────────┐ │      │  • Classify:        │
    │ │  Modality Classifier│ │      │    - casual         │
    │ │  (ViT 2-class)      │ │      │    - medical        │
    │ │  • PathVQA (0)      │ │      └──────┬──────────────┘
    │ │  • VQA-RAD (1)      │ │             │
    │ │  • OOD Detection:   │ │    ┌────────┴──────────┐
    │ │    - MSP < 0.80     │ │    │                   │
    │ │    - Entropy > 0.55 │ │    ▼                   ▼
    │ │    - Energy > -2.0  │ │  casual            medical
    │ └─────────┬───────────┘ │  response         (continue)
    │           │             │                        │
    │  ┌────────┴────────┐   │                        │
    │  │ If OOD:         │   │                        │
    │  │ Return rejection│   │                        │
    │  │ Skip PubMed +   │   │                        │
    │  │ Reasoning       │   │                        │
    │  └────────┬────────┘   │                        │
    │           │            │                        │
    │  ┌────────┴────────┐  │                        │
    │  │ If In-Domain:   │  │                        │
    │  │ Route to Model  │  │                        │
    │  │ • PathVQA →     │  │                        │
    │  │   Qwen2-VL-7B   │  │                        │
    │  │ • VQA-RAD →     │  │                        │
    │  │   Qwen3-VL-2B   │  │                        │
    │  └────────┬────────┘  │                        │
    └───────────┼───────────┘                        │
                │                                     │
                └─────────────┬───────────────────────┘
                              │
                     (if not OOD, not casual)
                              ▼
                  ┌───────────────────────┐
                  │    PubMed Agent       │
                  │  • Extract keywords   │
                  │  • NCBI E-utilities    │
                  │  • Top 3 articles     │
                  │  • Return:            │
                  │    - title            │
                  │    - abstract (500ch) │
                  │    - PMID + URL       │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Reasoning Agent     │
                  │  (Gemini Gemma-3-4B)  │
                  │  Synthesizes:         │
                  │  • Question           │
                  │  • VQA answer         │
                  │  • PubMed context     │
                  │  Output format:       │
                  │  - Answer (1 sent)    │
                  │  - Explanation + [n]  │
                  │  - Clinical context   │
                  │  - References (auto)  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Translation Agent    │
                  │  (Back Translation)   │
                  │  English → Original   │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Session Manager     │
                  │  Save to:             │
                  │  sessions/username/   │
                  │    session_id/        │
                  │  • session_data.json  │
                  │  • input_image.*      │
                  └───────────────────────┘



CONVERSATIONAL MEMORY SYSTEM

┌──────────────────────────────────────────────────────┐
│                   User Request                        │
│              (username + message + session_id?)       │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Session Check  │
                 └────────┬───────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
      session_id = null           session_id exists
       (New Chat)                  (Continue Chat)
            │                           │
            ▼                           ▼
    ┌───────────────┐          ┌────────────────┐
    │ Create New    │          │ Load Session   │
    │ session_id    │          │ Metadata       │
    └───────┬───────┘          └────────┬───────┘
            │                           │
            │                           ▼
            │                  ┌────────────────┐
            │                  │ Memory Check   │
            │                  └────────┬───────┘
            │                           │
            │              ┌────────────┴───────────┐
            │              │                        │
            │         In RAM Cache           Not in RAM
            │              │                        │
            │              ▼                        ▼
            │      ┌──────────────┐      ┌──────────────────┐
            │      │ Use Cached   │      │ Load from JSON   │
            │      │ Memory       │      │ Restore to RAM   │
            │      └──────┬───────┘      └─────────┬────────┘
            │             │                        │
            └─────────────┴────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Get Conversation      │
              │ Context (history)     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Run VQA Pipeline     │ ◄─── Diagram 1
              │  (with context)       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Save New Turn         │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ LangChain Memory │    │ Session Manager  │
    │ (RAM)            │    │ (JSON)           │
    │                  │    │                  │
    │ Active chats:    │    │ Persistent:      │
    │ {1: memory,      │    │ /username/       │
    │  2: memory,      │    │   1/data.json    │
    │  5: memory}      │    │   2/data.json    │
    │                  │    │                  │
    │ Cleared on       │    │ Survives app     │
    │ app restart      │    │ restart          │
    └──────────────────┘    └──────────────────┘