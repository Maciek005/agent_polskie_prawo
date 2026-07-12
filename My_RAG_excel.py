from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool 

load_dotenv()

llm = ChatOpenAI(
    model='gpt-4o', temperature=0
)


embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

# pdf paths
pdf_path = 'Kodeks_cywilny.pdf'
kazusy_pdf = 'kazusy_rozwiazania.pdf'


# checking if path works
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f'PDF file not found: {pdf_path}')

if not os.path.exists(kazusy_pdf):
    raise FileNotFoundError(f'PDF file not found: {kazusy_pdf}')


# pdf loading
pdf_loader = PyPDFLoader(pdf_path)
pdf2_loader = PyPDFLoader(kazusy_pdf)

try:
    pages = pdf_loader.load()
    print(f'PDF has been loaded and has {len(pages)} pages')
except Exception as e:
    print(f'Error loading PDF: {e}')
    raise 

try:
    pages2 = pdf2_loader.load()
    print(f'PDF has been loaded and has {len(pages2)} pages')
except Exception as e:
    print(f'Error loading PDF: {e}')
    raise 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)
pages2_split = text_splitter.split_documents(pages2)


persist_directory = r"C:/Fluxen/learning/LangGraph course/AI_agent_1"
collection_name = 'prawo_cywilne'


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


pages_split = [d for d in pages_split if (d.page_content or "").strip()]
pages2_split = [d for d in pages2_split if (d.page_content or "").strip()]


if not pages_split:
    raise ValueError(
        "Po wczytaniu/splitowaniu nie ma żadnego niepustego tekstu. "
        "PDF może być skanem (obraz), zaszyfrowany, albo ekstrakcja tekstu nie działa."
    )

if not pages2_split:
    raise ValueError(
        "Po wczytaniu/splitowaniu nie ma żadnego niepustego tekstu. "
        "PDF może być skanem (obraz), zaszyfrowany, albo ekstrakcja tekstu nie działa."
    )

for d in pages_split:
    d.metadata = {**(d.metadata or {}), "source_type": "act", "source_file": pdf_path}

for d in pages2_split:
    d.metadata = {**(d.metadata or {}), "source_type": "case", "source_file": kazusy_pdf}

all_docs = pages_split + pages2_split  # jedna lista Documentów

try:
    # creating vector database
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f'Created ChromaDB vector store!')

except Exception as e:
    print(f'Error setting up ChromaDB: {str(e)}')
    raise

# print("PAGES:", len(pages))
# print("Non-empty pages:", sum(1 for d in pages if d.page_content and d.page_content.strip()))
# if pages:
#     print("Sample page[0] chars:", len(pages[0].page_content or ""))
#     print("Sample page[0] preview:", (pages[0].page_content or "")[:200])

# print("SPLIT:", len(pages_split))
# print("Non-empty chunks:", sum(1 for d in pages_split if d.page_content and d.page_content.strip()))
# if pages_split:
#     print("Sample chunk[0] chars:", len(pages_split[0].page_content or ""))
#     print("Sample chunk[0] preview:", (pages_split[0].page_content or "")[:200])



# retriever = vectorstore.as_retriever(
#     search_type = 'similarity',
#     search_kwargs={'k':5} # K - amount of chunks to return
# )

retriever_act = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "filter": {"source_type": "act"}}
)

retriever_case = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "filter": {"source_type": "case"}}
)


# @tool
# def retriever_tool(query: str) -> str:
#     '''
#     This tool searches and returns the information from the kodeks cywilny dokument.
#     '''

#     docs = retriever.invoke(query)

#     if not docs:
#         return 'Nie znalazłem informacji na ten temat.'
    
#     if not query or not query.strip():
#         return "Podaj niepuste zapytanie."
    
#     results = []
#     for i,doc in enumerate(docs):
#         results.append(f'Document {i+1}: \n{doc.page_content}')

#     return '\n\n'.join(results)


def _format_docs(docs):
    out = []
    for i, doc in enumerate(docs):
        text = (doc.page_content or "").strip()
        if not text:
            continue
        out.append(f"Document {i+1}:\n{text}")
    return "\n\n".join(out) if out else "Znaleziono dokumenty, ale ich treść jest pusta."

@tool
def retriever_act_tool(query: str) -> str:
    """Szuka w Kodeksie Cywilnym (KC)."""
    if not query or not query.strip():
        return "Podaj niepuste zapytanie."
    docs = retriever_act.invoke(query)
    if not docs:
        return "Nie znalazłem informacji w Kodeksie Cywilnym."
    return _format_docs(docs)

@tool
def retriever_case_tool(query: str) -> str:
    """Szuka w bazie kazusów (przykłady/rozwiązania)."""
    if not query or not query.strip():
        return "Podaj niepuste zapytanie."
    docs = retriever_case.invoke(query)
    if not docs:
        return "Nie znalazłem podobnych kazusów."
    return _format_docs(docs)

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import re
from langchain_core.tools import tool

class GenerateCaseInput(BaseModel):
    temat: str = Field(..., description="Temat kazusu (np. 'odpowiedzialnosc kontraktowa', 'zasiedzenie', 'rękojmia')")
    poziom: Literal["latwy", "sredni", "trudny"] = "sredni"
    liczba_pytan: int = Field(3, ge=1, le=10)
    dlugosc: Literal["krotki", "standard", "dlugi"] = "standard"
    wymagane_sekcje: List[str] = Field(default_factory=lambda: ["STAN FAKTYCZNY", "PYTANIA"])
    zakazane: List[str] = Field(default_factory=list, description="Hasła/tematy, których nie wolno użyć (np. poza sylabusem)")
    tryb: Literal["bez_rozwiazania", "z_rozwiazaniem"] = "bez_rozwiazania"

def _simple_validate(text: str, required_sections: List[str], forbidden: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    req = {sec: (sec.lower() in lower) for sec in required_sections}
    forb = {f: (f.lower() in lower) for f in forbidden}
    ok = all(req.values()) and not any(forb.values())
    return {"ok": ok, "required": req, "forbidden_hits": forb}

@tool("generate_kazus_from_pdf", args_schema=GenerateCaseInput)
def generate_kazus_from_pdf(temat: str,
                            poziom: str = "sredni",
                            liczba_pytan: int = 3,
                            dlugosc: str = "standard",
                            wymagane_sekcje: Optional[List[str]] = None,
                            zakazane: Optional[List[str]] = None,
                            tryb: str = "bez_rozwiazania") -> Dict[str, Any]:
    """
    Generuje nowy kazus na wzór kazusów z PDF (styl/układ) + podpiera się przepisami KC.
    """
    wymagane_sekcje = wymagane_sekcje or ["STAN FAKTYCZNY", "PYTANIA"]
    zakazane = zakazane or []

    # 1) Pobierz wzorce kazusów (styl, układ, typ pytań)
    case_docs = retriever_case.invoke(
        f"Znajdz 2-4 najbardziej podobne kazusy do tematu: {temat}. "
        f"Interesuje mnie układ, styl, typ pytan i sposób formułowania."
    )
    case_context = _format_docs(case_docs)

    # 2) Pobierz przepisy KC pod temat
    act_docs = retriever_act.invoke(
        f"Przepisy Kodeksu cywilnego dotyczące tematu: {temat}. "
        f"Zwroc najistotniejsze artykuly i fragmenty."
    )
    act_context = _format_docs(act_docs)

    # 3) Ścisły prompt: format identyczny jak w PDF + brak lania wody
    sys = (
        "Jestes ekspertem od polskiego prawa cywilnego. "
        "Twoim zadaniem jest stworzyc NOWY kazus w stylu i ukladzie IDENTYCZNYM jak w dostarczonych wzorcach. "
        "Nie cytuj doslownie calych kazusow z bazy - tworz nowy stan faktyczny. "
        "Trzymaj sie tematu i realiow typowych dla kazusow z PDF. "
        "Jeśli brakuje danych, NIE pytaj użytkownika — wygeneruj najlepszą wersję na podstawie kontekstu."
    )

    # 4) Wymuszony format wyjścia (łatwo walidować)
    format_instr = f"""
Wygeneruj kazus o temacie: {temat}
Poziom: {poziom}
Dlugosc: {dlugosc}
Liczba pytan: {liczba_pytan}

WYMAGANY FORMAT (NIE ZMIENIAJ NAGLOWKOW):
TYTUL:
STAN FAKTYCZNY:
PYTANIA:
- 1) ...
- 2) ...
...
PODSTAWA PRAWNA (KC):
- art. ... (krótko, tylko to co potrzebne do pytan)

DODATKOWE ZASADY:
- Stan faktyczny: konkretny, realistyczny, bez ogolnikow.
- Pytania: musza wymuszac zastosowanie KC (np. roszczenia, skutki niewykonania, terminy, oświadczenia woli).
- NIE używaj tematów/pojęć z listy zakazanej: {zakazane}
- Jeśli tryb = z_rozwiazaniem: dodaj na koncu jeszcze:
ROZWIAZANIE (SKROT):
- punktowo, max 10-15 linijek, bez lania wody.
"""

    prompt = f"{sys}\n\n=== WZORCE KAZUSOW (PDF) ===\n{case_context}\n\n=== PRZEPISY KC (PDF) ===\n{act_context}\n\n=== INSTRUKCJA ===\n{format_instr}"

    # 5) Generacja przez ten sam llm (z tool-bindingiem już zrobionym u Ciebie)
    # Tu wywołujemy "gołego" modelu, bez kolejnych tooli (bo kontekst już mamy).
    # Jeśli chcesz, mogę dopasować to do Twojej architektury LangGraph (np. osobny node).
    response = ChatOpenAI(model="gpt-4o", temperature=0).invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
    text = response.content

    # 6) Walidacja
    validation = _simple_validate(text, wymagane_sekcje, zakazane)

    # 7) (Opcjonalnie) twarda kontrola liczby pytań
    found_questions = len(re.findall(r"^\s*-\s*\d+\)", text, flags=re.MULTILINE))
    validation["found_questions"] = found_questions
    validation["questions_ok"] = (found_questions == liczba_pytan)

    return {
        "kazus": text,
        "validation": validation,
        "used_context": {
            "case_chunks": [d.metadata for d in case_docs],
            "act_chunks": [d.metadata for d in act_docs],
        }
    }


tools = [retriever_act_tool, retriever_case_tool, generate_kazus_from_pdf]

llm = llm.bind_tools(tools)
tools_dict = {t.name: t for t in tools}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState):
    '''Check if the last message contains tool calls.'''
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
Jesteś wyspecjalizowanym w Polskim prawie cywilnym prawnikiem, który asystuje ludziom oraz prawnikom przy pytaniach dotyczących Kodeksu Cywilnego na bazie pliku pdf.
Użyj dostępnego retriever tool aby odpowiadać na pytania dotyczące kodeksu cywilnego. Możesz go użyć wielokrotnie jeśli potrzebujesz.
Jeśli potrzebujesz dopytać użykownika o dodatkowe informacje - zrób to.
Cytuj dokładnie fragmenty pliku pdf, które używasz, odpowiadaj w języku Polskim.
Na proste pytania odpowiadaj zawsze na dokładnie podstawie artykułu, którego dotyczy sprawa
Do rozwiązywania kazusów używaj narzędzia generate_kazus_from_pdf i na podstawie dostępnych kazusów pdf zwracaj rozwiązania.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}



# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== AGENT ===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ODPOWIEDŹ ===")
        print(result['messages'][-1].content)


running_agent()