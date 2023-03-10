import streamlit as st
from streamlit_chat import message

from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.chains import ConversationChain
from langchain.llms import OpenAI,Cohere
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
from langchain.chains import ChatVectorDBChain
import pickle
from langchain import OpenAI, VectorDBQA
from langchain.prompts.prompt import PromptTemplate


# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
    chunk_overlap  = 0)
#documents = text_splitter.split_text(raw_documents)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """You are an AI assistant whose  name is  Reservoir Buddy and
you will  answer questions from the relevant  vectorstore embeddings of Reservoir Management.  
Provide a conversational answer from the context and basic Reservoir Engineering knowledge and in the end of every answer  Reservoir suggest the user to read books by Tarak Ahmed
and L.P Dake for more expertise.
If you are asked about anything else than oil and gas , just say that you  are not allowed to talk about it, don't try to make up an answer. 
{context}
Question: {question}
Helpful Answer:"""
prompt_template1 = """You are an AI assistant whose  name is  Reservoir Buddy and
you will  answer questions from the relevant  vectorstore embeddings of Reservoir .  
Provide a conversational answer from the context and basic Reservoir Engineering knowledge and in the end of every answer  Reservoir suggest the user to read books by Tarak Ahmed
and L.P Dake for more expertise.
If you are asked about anything else than oil and gas , just say that you  are not allowed to talk about it, don't try to make up an answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    #template=prompt_template, input_variables=["context", "question"]
    template=prompt_template1, input_variables=["context", "question"]
)

# Load Data to vectorstore
#embeddings = CohereEmbeddings(cohere_api_key= "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg")
#embeddings = OpenAIEmbeddings()
#docsearch = FAISS.from_texts(documents, embeddings)
def load_vectorstore():
    '''load embeddings and vectorstore'''
           
    embeddings = CohereEmbeddings(cohere_api_key= "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg")
       
    return FAISS.load_local('res_hw_embeddings', embeddings)
    #return FAISS.load_local('resr_manang_embeddings', embeddings)

   

#default embeddings
docsearch = load_vectorstore()




qa=VectorDBQA.from_chain_type(llm=Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",temperature=0),
                              chain_type="map_reduce",prompt=QA_prompt, vectorstore=docsearch, return_source_documents=False)

#qa=ChatVectorDBChain.from_llm(llm=Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",temperature=0.7),
#                              chain_type="map_reduce",qa_prompt=QA_PROMPT,vectorstore=docsearch,return_source_documents=False,verbose=True,streaming=True
        #condense_question_prompt=CONDENSE_QUESTION_PROMPT
#                             )
#chain = load_chain(vectorstore,QA_PROMPT,CONDENSE_QUESTION_PROMPT)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chatbot", page_icon=":shark:")
st.header(" Reservoir Buddy🤖 Your Assistant")
expander = st.expander("Know about Me ")

expander.write("""
     I am an AI assistant for Oil and Gas Engineers based on LLMs(Large Language Models).Presently I know about the Reservoir Mangament Basics. Consider the generated response as starting point to assist in our work. 
     
 """)
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


placeholder = st.empty()
def get_text():
    
    input_text = placeholder.text_input("Enter what you want to know 👇", "", key="input")
    return input_text


user_input = get_text()
if st.button("Submit Your Query"):
    # check 
    docs = docsearch.similarity_search(user_input)
    # if checkbox is checked, print docs

    print(len(docs))
#if user_input:
    chat_history = []
    #output = qa({"question": user_input, "chat_history": chat_history})
    output = qa.run(user_input)
    
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(output["source_documents"][0])
    #st.session_state.generated.append([output["answer"],output["source_documents"]])
    #st.session_state.generated.append(output["answer"])
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True,avatar_style="thumbs",seed='Aneka',key=str(i) + "_user")

        message(st.session_state["generated"][i],avatar_style="fun-emoji", key=str(i))
        #message(st.session_state["generated"][i+1], key=str(i+1))
