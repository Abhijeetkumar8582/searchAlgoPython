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

app = Flask(__name__)

load_dotenv('file.env')
document_response = requests.get(
    'http://13.233.174.182:3000/user/getDetail',
    headers={
        'X-Api-Key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFiaGlqZWVzdEB5ZWwuYWkiLCJpYXQiOjE2OTI2MjUwNzgsImV4cCI6MTY5MjY2MTA3OH0.C1fsfYAc96SedbsiiD3AS2wJyb6zkdZEZYhB0M2UvZI'
    }
)
document_data = document_response.json()

document_url = document_data['Document']
file_extension = document_url.split('.')[-1].lower()

if file_extension == 'pdf':
    loader = PyPDFLoader(document_url)

elif file_extension == 'csv':
    response = requests.get(document_url)
    if response.status_code == 200:
        local_file_path = 'downloaded.csv'

        with open(local_file_path, 'wb') as f:
            f.write(response.content)

        loader = CSVLoader(local_file_path)
    else:
        print('Failed to download the CSV file')
else:
    raise ValueError("Unsupported file type")


pages = loader.load_and_split()
print(os.getenv('OPENAI_API_KEY'))
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

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


db = Chroma.from_documents(texts, OpenAIEmbeddings())

# app.run(host="127.0.0.2", port=8080, debug=True)÷

@app.route('/')
def home():
    x_api_key = request.headers.get('X-Api-Key')
    print(f"Received API key: {x_api_key}")  # Add this line for debugging
    return f"Hello, Flask! x_api_key: {x_api_key}"


@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        x_api_key = request.headers.get('X-Api-Key')
        data = request.json
        user_input = data['user_input']
        print(os.getenv('OPENAI_API_KEY'))

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

        chain = RetrievalQA.from_chain_type(llm=OpenAI(
        ), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)

        # Retrieve answer using the question
        result = chain.run(optimize_query)
        vectorizer = CountVectorizer().fit_transform([optimize_query, result])
        cosine_sim = cosine_similarity(vectorizer)
        similarity_score = cosine_sim[0, 1]

        return jsonify({'question': optimize_query, 'answer': result})

    except Exception as e:
        return jsonify({'error': str(e)})


def generate_answer(question):
    # Replace this with your logic to retrieve an answer based on the generated question
    # For demonstration purposes, a placeholder answer is returned here
    return "This is a placeholder answer."


if __name__ == '__main__':
    app.run(debug=True)
