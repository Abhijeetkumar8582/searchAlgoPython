import langchain as lc
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.document_loaders.csv_loader import CSVLoader
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


load_dotenv('file.env')
print(os.getenv('OPENAI_API_KEY'))

loader = CSVLoader(file_path='email-password-recovery-code.csv')
# loader = PyPDFLoader("https://searchalgoaws.s3.amazonaws.com/abhijeest@yel.ai-73e8cffd-43ca-4c0a-86a9-f4efbbd5044f-𝐒3𝐛𝐮𝐜𝐤𝐞𝐭𝐬𝐮𝐬𝐢𝐧𝐠𝐧𝐨𝐝𝐞.pdf")
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
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


db = Chroma.from_documents(texts, OpenAIEmbeddings())


query = "tell me aws bucket"
openai.api_key = os.getenv('OPENAI_API_KEY')
completion = openai.Completion.create(model="text-davinci-003", max_tokens=250, temperature=0.6, top_p=0.9, prompt=f"Generate only one of insightful and detailed questions based on the following topic: {query}. Don't change the meaning of the question .")

optimize_query = completion.choices[0].text

llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),
             temperature=0.1, top_p=1, model='text-embedding-ada-002')

chain = RetrievalQA.from_chain_type(llm=OpenAI(
), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)


conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)


# conversation.run(optimize_query)

result = chain.run(optimize_query)
# conversation.run(result)


vectorizer = CountVectorizer().fit_transform([optimize_query, result])
cosine_sim = cosine_similarity(vectorizer)
similarity_score = cosine_sim[0, 1]

# # # Print the result
print(optimize_query)
print(result)
print(similarity_score)
