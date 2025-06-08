from src.VectorDb import *
from src.LLM import *
from src.RAGChain import *

from langserve import add_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def MakeChain():
    #Load Db
    db = VectorDb()
    retriever = db.GetRetriever()
    #LLM
    llmObj =  LLM()
    llm = llmObj.getLLMInstance()
    #Chain
    rag = RAGChain(llm=llm, retriever=retriever)
    return rag.MakeChain()


answer_chain = MakeChain()

app = FastAPI(
    title="RAG Server",
    version="1.0",
    description="A simple RAG api server acting as an Insurance Assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    answer_chain,
    path="/rag",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
