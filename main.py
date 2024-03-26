import os
from embed_data import chroma_db_store
from get_result import get_result

chat_history = []
 
def get_llm_response(query,model,image_path=None,api_key=None,explain_to_kid=False,offline=False):
    global chat_history
    
    if not offline:
        os.environ["GOOGLE_API_KEY"] = api_key

    vectordb = chroma_db_store()
    result, chat_history = get_result(vectordb,query,model,image_path,chat_history,explain_to_kid,offline)
    return result
