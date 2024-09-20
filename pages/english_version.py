import os
import pandas as pd
from io import StringIO
import streamlit as st
from elasticsearch import Elasticsearch
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig, Image, Part

# This code shows VertexAI GenAI integration with Elastic Vector Search features
# to connect publicly trained LLMs with private data
# Gemini Pro model is used

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code

st.set_page_config(page_title="English version", layout="wide") 

# Required Environment Variables
# gcp_project_id - Google Cloud project ID
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

projid = os.environ['gcp_project_id']
cid = os.environ['cloud_id']
cp = os.environ['cloud_pass']
cu = os.environ['cloud_user']

generation_config = GenerationConfig(
    temperature=0.4, # 0 - 1. The higher the temp the more creative and less on point answers become
    max_output_tokens=2048, #modify this number (1 - 1024) for short/longer answers
    top_p=0.8,
    top_k=40,
    candidate_count=1,
)

#            safety_settings={
#                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#            }

vertexai.init(project=projid, location="us-central1")

model = GenerativeModel("gemini-pro")
visionModel = GenerativeModel("gemini-1.0-pro-vision-001")
#Gemini can hold chat history in the chat variable. Pass this variable every time along with the prompt one. 
#chat = model.start_chat()

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es

# Search ElasticSearch index and return details on relevant products
def search_products(query_text):

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    '''
    query = {
        "bool": {
            "must": [{
                "match": {
                    "product-name": {
                        "query": query_text,
                        "boost": 0.2
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "details_embedding"
                }
            }]
        }
    }
    '''

    knn = [
    {
        "field": "details_embedding",
        "k": 10,
        "num_candidates": 50,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": ".multilingual-e5-small_linux-x86_64",
                "model_text": query_text
            }
        },
        "boost": 0.2
    },
    {
        "field": "title_embedding",
        "k": 10,
        "num_candidates": 50,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": ".multilingual-e5-small_linux-x86_64",
                "model_text": query_text
            }
        },
        "boost": 0.8
    }
    ]

    fields = ["product-link", "product-name", "product-characteristics", "product-price", "product-image-src", "product-link-href", "category-link"]
    index = 'leroy-merlin-fr-catalog-vector'
    resp = es.search(index=index,
                     #query=query,
                     knn=knn,
                     fields=fields,
                     size=10,
                     source=False)

    doc_list = resp['hits']['hits']
    body = resp['hits']['hits']
    url = ''
    for doc in doc_list:
        #body = body + doc['fields']['description'][0]
        url = url + "\n\n" +  doc['fields']['product-link-href'][0]

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from Gemini based on the given prompt - NOT USED
def vertexAI(chat: ChatSession, prompt: str) -> str:
    response = chat.send_message(prompt)
    return response.text

def generateResponse(prompt):
    response = model.generate_content(prompt, 
                                      generation_config=generation_config
    )
    return response.text

def generateVisionResponse(prompt, image):
    image_bytes_data = image.getvalue()
    convertedImage = Part.from_image(Image.from_bytes(image_bytes_data))
    response = visionModel.generate_content(
        [
        convertedImage,
        prompt
        ], generation_config = generation_config
    )
    return response.text


#image = Image.open('homecraft_logo.jpg')

with st.container():
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.image("./resources/logo.png", caption=None, width=350)
    with c2:
        pass
    with c3:
        pass
    
    st.title("Your e-commerce search bar")

with st.container():
    left_column, right_column = st.columns([2,1])

# Main chat form
with left_column:
    with st.form("chat_form"):
        user_query = st.text_input("You: ")
        submit_button = st.form_submit_button("Submit")
#image load component
with right_column:
    uploaded_file = st.file_uploader("Add an image to your prompt (.png or .jpg)", ['png','jpg'])

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Leroy Merlin dataset."
if submit_button:
    queryForElastic = ''
    answerVision = ''
    if uploaded_file is not None:
        visionQuery = 'What is the product in the picture? Describe it with max 20 words'
        answerVision = generateVisionResponse(visionQuery,uploaded_file)
        #To Elastic, for semantic search, we send both the question and the first answer from the vision model
        queryForElastic = answerVision 
        st.write(f"**Vision assistant answer:**  \n\n{answerVision.strip()}")
    
    es = es_connect(cid, cu, cp)
    #topicCheck = generateResponse(f"What's the product the user is talking about in the question? Answer in french. Question: {user_query}")
    resp_products, url_products = search_products(user_query if queryForElastic == '' else queryForElastic)
    #resp_docs, url_docs = search_docs(user_query if queryForElastic == '' else queryForElastic)
    #resp_order_items = search_orders(1) # 1 is the hardcoded userid, to simplify this scenario. You should take user_id by user session
    #prompt = f"You are an e-commerce AI assistant. Answer this question: {query} using this context: \n{resp_products} \n {resp_docs} \n {resp_order_items}."
    #prompt = f"You are an e-commerce AI assistant. Answer this question: {query}.\n If product information is requested use the information provided in this JSON: {resp_products} listing the identified products in bullet points with this format: Product name, product key features, price, web url. \n For other questions use the documentation provided in these docs: {resp_docs} and your own knowledge. \n If the question contains requests for user past orders consider the following order list: {resp_order_items}"
    ##prompt = [f"You are an e-commerce AI assistant.", f"You answer question around product catalog, general company information and user past orders", f"Answer this question: {query}.", f"Context: Picture content = {answerVision}; Product catalog = {resp_products}; General information = {resp_docs}; Past orders = {resp_order_items} "]
    #prompt = f"You are an e-commerce customer AI assistant. Answer this question: {query}.\n with your own knowledge and using the information provided in the context. Context: JSON product catalog: {resp_products} \n, these docs: \n {resp_docs} \n and this user order history: \n {resp_order_items}"
    #answer = vertexAI(chat, prompt)
    #prompt = [f"You are an e-commerce AI assistant.", f"You answer question around product catalog, general company information and user past orders", f"You answer questions in french language", f"Question: {query};", f"Context: Picture content = {answerVision}; Product catalog = {resp_products};" ]
    prompt = [f"You are an e-commerce AI assistant. You answer question around product catalog, general company information and user past orders. You answer questions in english and using this context: Picture content = {answerVision}, Product catalog = {resp_products}." f"Question: {user_query}."]
    #prompt = [f"Vous êtes un assistant IA e-commerce. Vous répondez aux questions concernant le catalogue de produits, les informations générales sur l'entreprise et les commandes passées des utilisateurs. Vous répondez aux questions en français et en utilisant ce contexte : Contenu de l'image = {answerVision}, Catalogue de produits = {resp_products}." f"Demande: {user_query}"]
    answer = generateResponse(prompt)

    if answer.strip() == '':
        st.write(f"**Elastic-powered Assistant:** \n\n{answer.strip()}")
    else:
        st.write(f"**Elastic-powered Assistant:** \n\n{answer.strip()}\n\n")
        #st.write(f"Order items: {resp_order_items}")

