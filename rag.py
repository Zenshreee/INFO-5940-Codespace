import os
import uuid
from io import BytesIO
from typing import List, Dict

from pypdf import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

CHAT_MODEL = "openai.gpt-4o-mini"
EMBED_MODEL = "openai.text-embedding-3-large"
CHROMA_DIR = "chroma_storage"

def read_file(name: str, data: bytes):
    name = name.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text)
    return ""


def split_text(text: str, filename: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"filename": filename}) for c in chunks]

def get_store():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name="info5940-a1",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def add_files(files):
    store = get_store()
    added = []
    for f in files:
        text = read_file(f.name, f.read()).strip()
        if not text:
            continue

        id = str(uuid.uuid4())
        docs = split_text(text, f.name)

        for d in docs:
            d.metadata["doc_id"] = id

        store.add_documents(docs)
        added.append({"filename": f.name, "doc_id": id})

    store.persist()
    return added


def retrieve(question, k=5):
    store = get_store()
    return store.similarity_search(question, k=k)

def build_prompt(question: str, docs: List[Document]) -> str:
    parts = [f"{d.metadata.get('filename')}\n{d.page_content}" for d in docs]
    context = "\n\n---\n\n".join(parts)
    return (
        "You are a helpful assistant for question answering.\n"
        "Use ONLY the provided context to answer concisely.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        "IMPORTANT: When you use information from the context, add a bracketed citation immediately after the clause or sentence that uses it, using the FILENAME exactly as shown in the context (for example: [test.pdf], [lecture1.pdf]).\n"
        "Do NOT invent facts or cite filenames for information not present in the context.\n"
        "Keep answers short and directly supported by the cited context.\n\n"
        f"Question: {question}\n\n"
        f"Context (each chunk shows its source filename):\n{context}\n\n"
        "At the end of your answer, include a single-line 'CITED' summary listing the filenames you used (for example: CITED: [test.pdf], [lecture1.pdf])."
    )


def answer(question: str, history: List[Dict[str, str]]) -> str:
    docs = retrieve(question, k=5)
    system_prompt = build_prompt(question, docs)
    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(SystemMessage(content=f"(assistant) {m['content']}"))
    messages.append(HumanMessage(content=question))
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    response = llm.invoke(messages).content.strip()
    return response