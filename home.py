import langchain as lc
import os
import pinecone
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
# import nltk
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain



load_dotenv('file.env')



loader = PyPDFLoader("AbhijeetKumar.pdf")
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


db = Chroma.from_documents(texts, OpenAIEmbeddings())
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.9,max_tokens=350)

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='northamerica-northeast1-gcp' 
)
index_name = 'resume'
index = pinecone.Index(index_name)

docsearch = Pinecone.from_texts([t.page_content for t in texts],embeddings,index_name=index_name)
query = "tell me about your project"


chain = load_qa_chain(llm=llm,chain_type='stuff')
docs = docsearch.similarity_search(query)

# print(docs)

result = chain.run(input_documents=docs,question = query)
print(result)

vectorizer = CountVectorizer().fit_transform([query, result])
cosine_sim = cosine_similarity(vectorizer)
similarity_score = cosine_sim[0, 1]

print(similarity_score)



# Now you can work with the 'data' obtained from the website

