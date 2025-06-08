from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

class RAGChain:

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def MakePromptTemplate(self):
        ANSWER_PROMPT_TEMPLATE = """
        You are a helpful customer support assistant. Use the following context to answer the question as accurately as possible.

        If you cannot find a clear answer, say: "I'm sorry, I don't have that information in my documentation."

        Be concise and do not make up information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        self.ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)

    def MakeChain(self):
        self.MakePromptTemplate()
        answer_chain = (
            RunnablePassthrough.assign(
                    context=lambda x: self.retriever.get_relevant_documents(x["question"])
            )
            | self.ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )

        return answer_chain
