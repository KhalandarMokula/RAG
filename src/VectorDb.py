import json
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDb:

    def __init__(self):
        self.LoadEmbeddings()
        self.MakeDb()
        
    def LoadEmbeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="embeddingModel/all-MiniLM-L6-v2")

    def MakeDb(self):
        # Load exported docs
        with open("Db/docs.json") as f:
            raw_docs = json.load(f)

        docs = [Document(page_content=d, metadata={}) for d in raw_docs]  # add back metadata if needed

        db = Chroma.from_documents(docs, self.embeddings, persist_directory="../Db/chroma")
        db.persist()

    def GetRetriever(self):
        vectorstore = Chroma(persist_directory='../Db/chroma', embedding_function=self.embeddings)
        print(len(vectorstore.get()['documents']))
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return retriever