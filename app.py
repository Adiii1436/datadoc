import os
import streamlit as st
from main import get_llm_response

def save_uploaded_file(uploaded_file):
    dir_name = "uploaded_images"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        with open(os.path.join(dir_name, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return os.path.join(dir_name, uploaded_file.name)
    except Exception as e:
        print(e)
        return None

st.set_page_config(page_title="Datadoc", page_icon=":robot:", layout="wide", initial_sidebar_state="expanded")

col1, col2 = st.columns([6,1])
with col1:
    st.title("Datadoc: Your AI DOC Assistant")

with col2:
    vert_space = '<div style="padding: 20px;"></div>'
    st.markdown(vert_space, unsafe_allow_html=True)
    offline_mode = st.toggle("Offline Mode", key='offline_toggle')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

st.sidebar.title("Options")

if(offline_mode):
    model_name = st.sidebar.radio("Choose a model:", ("GPT4All offline",))
else:
    model_name = st.sidebar.radio("Choose a model:", ("Gemini Pro", "Gemini Pro Vision"))

if not offline_mode:
    api_key = st.text_input("Enter your API key:", type="password")

clear_button = st.sidebar.button("Clear Conversation", key="clear")

if not offline_mode and model_name == "Gemini Pro":
    model = "gemini-pro"
elif not offline_mode and model_name == "Gemini Pro Vision":
    model = "gemini-pro-vision"
else:
    model = "GPT4All"

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []

def generate_response(prompt, model, image_path=None, explain_to_kid=False):
    if len(st.session_state['past']) == len(st.session_state['generated']):
        st.session_state['past'].append(prompt)
    else:
        st.session_state['past'][-1] = prompt

    if not offline_mode:
        response = get_llm_response(prompt, model, image_path, api_key, explain_to_kid)
    else:
        response = get_llm_response(prompt, model, image_path, explain_to_kid, offline=True)

    if len(st.session_state['generated']) < len(st.session_state['past']):
        st.session_state['generated'].append(response)
    else:
        st.session_state['generated'][-1] = response

    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100, max_chars=500)
        
        uploaded_file = ""
        if model_name == "Gemini Pro Vision":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        
        col1, col2 = st.columns([6, 1])

        with col1:
           submit_button = st.form_submit_button(label='Send')

        with col2:
            explain_kid = st.toggle("Child Mode", key='explain_toggle')
        
    if not offline_mode and submit_button and not api_key:
        st.warning("Please enter your API key.")
    elif not offline_mode and submit_button and not uploaded_file and model_name == "Gemini Pro Vision":
        st.warning("Please upload an image to use the Image Model.")
    elif uploaded_file:
        image_path = save_uploaded_file(uploaded_file)
        if image_path:
            output = generate_response(user_input, model, image_path, explain_kid)
    elif submit_button and user_input:
        output = generate_response(user_input, model,explain_to_kid=explain_kid)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            st.markdown(f"**You:** {st.session_state['past'][i]}")
            with st.container(border=True):
                st.markdown(f"{st.session_state['generated'][i]}")    