# Civil Law RAG Agent (Kodeks Cywilny Assistant)

An agentic RAG system that answers questions about the Polish Civil Code (*Kodeks Cywilny*) with cited statute fragments, and generates new practice case studies (*kazusy*) in the style of real law exam materials — with automatic validation of the generated output.

Built as a personal project to learn practical RAG/agent design patterns beyond a single-retriever demo: two independently-tuned retrievers, metadata filtering, a LangGraph tool-calling loop, and a structured-output generation tool with schema validation.

## What it does

- **Answers legal questions** by retrieving the most relevant articles from the Civil Code and citing them directly in the response (in Polish).
- **Generates new case studies** on a requested topic (e.g. *"zasiedzenie"*, *"rękojmia"*) at a chosen difficulty and length, styled after a corpus of real exam and competition case studies, with or without a model solution.
- **Validates its own output** — checks that generated cases contain the required sections, avoid forbidden topics, and match the requested number of questions before returning them.
- **Decides for itself** which tool to call and how many times, via a LangGraph agent loop rather than a fixed pipeline.

## Architecture

```
Kodeks_cywilny.pdf ───┐
                       ├─▶ split & embed (OpenAI text-embedding-3-small) ─▶ Chroma vector store
kazusy_rozwiazania.pdf ┘        each chunk tagged: source_type = "act" | "case"
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │              two independent retrievers        │
                        │   retriever_act  (MMR, filter: act)             │
                        │   retriever_case (MMR, filter: case)            │
                        └───────────────────────┬───────────────────────┘
                                                │
User question ──────────────────────▶  LangGraph agent (GPT-4o)
                                        tools available:
                                         • retriever_act_tool
                                         • retriever_case_tool
                                         • generate_kazus_from_pdf (Pydantic-validated)
                                                │
                                                ▼
                                   Cited answer, or a generated +
                                   self-validated case study
```

The agent graph is a simple two-node loop (`llm` ⇄ `retriever_agent`): the model calls tools as needed, results are fed back in, and the loop ends once the model responds without further tool calls.

## Tech stack

- **Python**
- **LangChain** / **LangGraph** — agent orchestration and tool-calling loop
- **OpenAI API** — `gpt-4o` for generation, `text-embedding-3-small` for embeddings
- **ChromaDB** — persisted vector store with metadata-filtered retrieval (MMR search)
- **Pydantic** — structured, validated input schema for the case-generation tool
- **PyPDF** — PDF ingestion

## Project structure

```
law_agents_pl/
├── My_RAG_excel.py         # main agent: dual retrievers + case generation tool + LangGraph loop
├── Kodeks_cywilny.pdf      # source: Polish Civil Code
├── kazusy_rozwiazania.pdf  # merged corpus of real exam/competition case studies
├── kazusy_baza/            # individual source case-study PDFs + merge script
│   └── pdf_merger.py       # combines the individual PDFs above into kazusy_rozwiazania.pdf
├── prompty.txt             # reference copy of the agent's system prompt
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <this-repo>
cd law_agents_pl
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

Run the agent:

```bash
python My_RAG_excel.py
```

On first run this builds a local Chroma vector store from the two source PDFs (this can take a minute); subsequent runs reuse it. You'll get an interactive prompt — ask a question in Polish and the agent will respond with cited statute fragments, or ask it to generate a case study on a given topic.

## Example

```
What is your question: Co grozi za niewykonanie zobowiązania?

=== ODPOWIEDŹ ===
Zgodnie z art. 471 Kodeksu Cywilnego: "Dłużnik obowiązany jest do naprawienia
szkody wynikłej z niewykonania lub nienależytego wykonania zobowiązania,
chyba że niewykonanie lub nienależyte wykonanie jest następstwem
okoliczności, za które dłużnik odpowiedzialności nie ponosi." [...]
```

## Known limitations / next steps

- No automated evaluation yet — a natural next step would be an LLM-as-judge or BERTScore comparison between generated case studies and real ones.
- No API layer — the agent currently runs as a CLI loop; wrapping it in FastAPI would make it usable as a service.
- The vector store persist path is currently a hardcoded local path from development; should be made relative before reuse elsewhere.

## Disclaimer

This is a personal/educational project exploring RAG and legal-document Q&A. It is not legal advice and has not been validated by a legal professional.
