# Catalyst: AI-Powered Talent Scouting & Engagement Agent 🚀

An end-to-end multi-agent AI recruiter built for the Deccan AI Experts Hackathon. This system automates job description parsing, RAG-based candidate discovery, simulated conversational engagement, and outputs a mathematically aggregated ranked shortlist.

## 🏗 Architecture Diagram

```mermaid
graph TD
    A[Recruiter Inputs JD] --> B(Phase 1: RAG Discovery Engine)
    
    subgraph Phase 1: Vector Matching
    B --> C[(ChromaDB: Local Vector Store)]
    C --> D[Retrieve Top K Candidates]
    D --> E{Groq Llama 3.1: JD vs Profile}
    E --> F[Generate Match Score 0-100 & Explainability]
    end
    
    F --> G(Phase 2: Conversational Agent)
    
    subgraph Phase 2: Engagement Simulator
    G --> H[Simulated Chat Interface]
    H --> I{Groq Llama 3.1: Recruiter Persona}
    I <--> J[Candidate / User Responses]
    J --> K[End Chat]
    K --> L{Groq Llama 3.1: Sentiment & Intent Analysis}
    L --> M[Generate Interest Score 0-100]
    end
    
    M --> N(Phase 3: Final Dashboard)
    N --> O[Calculate Final Combined Score]
    O --> P[Ranked Output UI]