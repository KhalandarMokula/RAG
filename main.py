# main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
load_dotenv()
DB_PATH = "chroma_db"

# --- Pydantic Models for API ---
class ChatRequest(BaseModel):
    message: str
    session_id: str

# --- Global Components (Initialized once) ---
# Initialize the LLM for generation
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# Initialize the Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="embedding_model/all-MiniLM-L6-v2")

# Load the existing vector store
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# In-memory store for chat histories. 
# In a production app, you'd replace this with Redis, a DB, etc.
chat_histories: Dict[str, List[Any]] = {}


# --- Prompt Templates ---
# 1. Prompt for condensing a question based on chat history
CONDENSE_QUESTION_PROMPT_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

# 2. Prompt for answering the question using retrieved context
ANSWER_PROMPT_TEMPLATE = """
You are a helpful and friendly customer support assistant. Answer the user's question based ONLY on the following context.
If the answer is not in the context, respond with "I'm sorry, I don't have that information in my documentation. Please contact a human support agent for further assistance."
Do not make up information. Be concise and clear.

Context:
{context}

Question:
{question}

Answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)


# --- LangChain Runnable Chains ---

def _format_chat_history(chat_history: List[Any]) -> str:
    """Formats chat history into a human-readable string."""
    buffer = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            buffer.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            buffer.append(f"AI: {message.content}")
    return "\n".join(buffer)

# Chain to condense the question
condense_question_chain = (
    CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser()
)

# Main chain to answer the question
answer_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retriever.get_relevant_documents(x["question"])
    )
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)

# The final conversational chain
def conversational_rag_chain(question: str, chat_history: List[Any]):
    """
    The main RAG chain. If there's a chat history, it first condenses the question.
    Otherwise, it uses the original question.
    """
    if chat_history:
        standalone_question_chain = RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        ) | condense_question_chain
        
        standalone_question = standalone_question_chain.invoke({
            "chat_history": chat_history,
            "input": question
        })
        return answer_chain.stream({"question": standalone_question})
    else:
        return answer_chain.stream({"question": question})

# --- FastAPI Application ---
app = FastAPI(
    title="RAG Chatbot Backend",
    description="A backend for a customer support RAG chatbot with memory.",
    version="1.0.0",
)

@app.post("/chat")
async def chat(request_body: ChatRequest):
    """
    Handles a chat request, manages history, and streams the response.
    """
    session_id = request_body.session_id
    user_message = request_body.message

    # Retrieve or create chat history for the session
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    current_chat_history = chat_histories[session_id]

    async def stream_generator():
        # Use the conversational RAG chain to get a streaming response
        streaming_response = conversational_rag_chain(user_message, current_chat_history)
        
        full_bot_response = ""
        for chunk in streaming_response:
            full_bot_response += chunk
            yield f"data: {chunk}\n\n"
        
        # Update the chat history after the stream is complete
        current_chat_history.append(HumanMessage(content=user_message))
        current_chat_history.append(AIMessage(content=full_bot_response))
        
        # Signal the end of the stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    Retrieves the chat history for a given session.
    """
    if session_id in chat_histories:
        return {"history": chat_histories[session_id]}
    return {"history": []}