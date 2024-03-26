from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from embed_data import chroma_db_store
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from load_model import load_image_llm,load_text_llm

def get_result(vectordb,query,model,image_path=None,chat_history=[],explain_to_kid=False,is_offline=False):
    
    if(model=="gemini-pro-vision"):
        
        image_llm = load_image_llm()
        message = HumanMessage(
        content=[
                {
                    "type": "text",
                    "text": query,
                },
                {"type": "image_url", "image_url": image_path},
            ]
        )
        content = image_llm.invoke([message]).content
        # save_string_to_txt(content)
        return content,chat_history
    else:
        text_llm = load_text_llm(is_offline)

        retriever = vectordb.as_retriever()

        # prompt = hub.pull("rlm/rag-prompt")

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_q_chain = contextualize_q_prompt | text_llm | StrOutputParser()

        if explain_to_kid:
            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            You're an experienced teacher who loves simplifying complex topics for young children to understand. \
            Your task is to explain a complex topic as if you are talking to a 5-year-old. \
            Make sure to use playful and engaging language to keep the child's attention and break down any difficult ideas into simple, manageable parts.For example, if you were explaining photosynthesis, you could say something like: "Plants eat sunlight by dancing in the sun all day long. Imagine the sun as their yummy snack! They also drink from the ground through their roots like using a straw. With these snacks and drinks, they make their own food just like magic!" \
            If you don't know the answer, just say that you don't know. \

            {context}"""
        else:
            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            Try to analyze the given context and answer the question to the best of your ability. \
            If you don't know the answer, just say that you don't know. \

            {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | text_llm
        )

        res = rag_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=query), res])
        
        if not is_offline:
            return res.content, chat_history
        else:
            return res, chat_history
