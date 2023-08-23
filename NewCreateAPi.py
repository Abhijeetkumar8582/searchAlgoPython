import langchain as lc
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import requests
from langchain.document_loaders.csv_loader import CSVLoader
import jwt
import pymongo



app = Flask(__name__)


load_dotenv('file.env')

# app.run(port=8080)
@app.route('/', methods=['GET'])
def home():
    x_api_key = request.headers.get('X-Api-Key')
    print(f"Received API key: {x_api_key}")  
    return f"Hello, Flask! x_api_key: {x_api_key}"


@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        jwt_token = os.getenv('JWT_TOKEN')
        secret_key = request.headers.get('X-Api-Key')
        print(secret_key)
        try:
            decoded_token = jwt.decode(secret_key, jwt_token, algorithms=["HS256"])
            print(decoded_token)
        except jwt.ExpiredSignatureError:
            return "Token has expired.", 401
        except jwt.InvalidTokenError:
            return "Invalid token.", 401

        # Fetch document details using the x_api_key
        mongo_uri = os.getenv('MONGO_URI')
        database_name = os.getenv('DATABASE_NAME')
        collection_name = os.getenv('COLLECTION_NAME')
        print(os.getenv('COLLECTION_NAME'))
        # Establish a connection to MongoDB
        client = pymongo.MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]
        email_id = decoded_token['email']
        query = {"email": email_id}
        result = collection.find(query)
        for document in result:
           document_url = document["documentUrl"]

        loader = PyPDFLoader(document_url)
        
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        texts = text_splitter.split_documents(pages)
        
        prompt_template = """Your goal is to answer user question from data you are getting and if you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in english:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY'))

        db = Chroma.from_documents(texts, embeddings)

        x_api_key = request.headers.get('X-Api-Key')
        data = request.json
        user_input = data['user_input']

        # Generate a question based on user input
        openai.api_key = os.getenv('OPENAI_API_KEY')
        completion = openai.Completion.create(
            model="text-davinci-003",
            max_tokens=150,
            temperature=0.1,
            top_p=1,
            prompt=f"Generate only one of insightful and detailed questions based on the following topic: {user_input}.Don't change the meaning of the question what user asked just optimize that."
        )
        optimize_query = completion.choices[0].text
        llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),
                     temperature=0.1, top_p=1, model='text-embedding-ada-002')
    
        conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(),
            verbose=True
        )
        chain = RetrievalQA.from_chain_type(
            llm=OpenAI(), 
            chain_type="stuff", 
            retriever=db.as_retriever(), 
            chain_type_kwargs=chain_type_kwargs
            )
        
        # conversation.predict(optimize_query)
        result = chain.run(optimize_query)
        

        return jsonify({'question': optimize_query, 'answer': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
