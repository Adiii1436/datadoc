from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import GPT4All

def load_image_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision",convert_system_message_to_human=True)
    return llm

def load_text_llm(offline=False):
    if not offline:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",convert_system_message_to_human=True)
    else:
        llm = GPT4All(model="models/mistral-7b-openorca.gguf2.Q4_0.gguf", n_threads=8)
    
    return llm