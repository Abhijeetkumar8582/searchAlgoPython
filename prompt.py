import langchain
import os
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pymongo
import jwt


load_dotenv('file.env')
jwt_token = "24b19dee4a73614386df760ad1d4b574eeb493ff86491bf52e508fe6097d0073"
secret_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IkFiaGlqZWV0QHllbGxvdy5jb20iLCJpYXQiOjE2OTI3NjU4NDYsImV4cCI6MTY5MjgwMTg0Nn0.CEmVg_t2aH0f2jOHzwVPF6qS7DMYLrVfcZWMGWGe28Q"

try:
    decoded_token = jwt.decode(secret_key, jwt_token, algorithms=["HS256"])
    print(decoded_token, 'decoded')
except jwt.ExpiredSignatureError:
    print("Token has expired.",decoded_token)
except jwt.InvalidTokenError:
    print("Invalid token.",decoded_token)
except jwt.ExpiredSignatureError:
    print("Token has expired.", 401)
except jwt.InvalidTokenError:
    print("Invalid token.", 401)

mongo_uri = "mongodb+srv://User_registration:Jan1457%40mongodb@cluster0.xs2ztmm.mongodb.net/userdatabase?retryWrites=true&w=majority"
database_name = "userdatabase"
collection_name = "searchAlgo"
print(decoded_token['email'])
# Establish a connection to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]
email_id = decoded_token['email']
query = {"email": email_id}
result = collection.find(query)
for document in result:
    documentUrl = document["documentUrl"]

print(documentUrl)
loader = PyPDFLoader(documentUrl)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

prompt_template = """Your goal is to answer user question from data you are getting and if you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in english:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


db = FAISS.from_documents(docs, embeddings)


query = "tell me about your skills"
openai.api_key = os.getenv('OPENAI_API_KEY')
completion = openai.Completion.create(model="text-davinci-003", max_tokens=250, temperature=0.6, top_p=0.9, prompt=f"Generate only one of insightful and detailed questions based on the following topic: {query}. Don't change the meaning of the question .")

optimize_query = completion.choices[0].text


docs = db.similarity_search(query)
combined_content = ""

for document in docs:
    combined_content += document.page_content + " "  # Add a space between chunks

# Print the combined content as a single paragraph
print(combined_content)