from langchain_groq import ChatGroq

class LLM:

    def __init__(self):
        self.api_key = ""
    
    def getLLMInstance(self):
        llm = ChatGroq(temperature=0, groq_api_key= self.api_key, model_name="llama-3.1-8b-instant")
        return llm