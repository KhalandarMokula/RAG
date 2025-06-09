**RAG Pipeline:**

![image](https://github.com/user-attachments/assets/77cab612-79fa-4c54-9cfe-acb17c3fa766)


**Steps to Generate & Store Data in VectorDB:**
1. Run Insurance_RAG.ipynb (PDFParsing/Insurance_RAG.ipynb) Notebook - [Extracts text, tables ->  perfoms chunking -> Generates Documents -> transforms to embeddings ->  Stores in VectorDB -> Dumps the Documents into .json

**Steps to Run the ChatBot application:**
1. Run main.py (python main.py)

[This Builds the VectorDB from the Dumped Json (VectorDB.py) -> Initializes LLM (LLM.py) (uses groq api) -> Create PromptTemplate + RAGChain (RAGChain.py) -> Define fastapi App & Route -> runs the app with langserve]  

**Embedding Model Used:**
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**LLM Used:** 
llama-3.1-8b-instant served by Groq

