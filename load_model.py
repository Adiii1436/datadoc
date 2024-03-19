from langchain_google_genai import ChatGoogleGenerativeAI

def load_image_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision",convert_system_message_to_human=True)
    return llm

def load_text_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
    return llm