import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import tempfile
import whisper
from pytube import YouTube
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv() #loading .env variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#youtube video url we are going to transcribe
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=2DvrRadXwWY"

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser  = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question} 
"""

prompt = ChatPromptTemplate.from_template(template)

if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    whisper_mod = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmp:
        file = audio.download(output_path=tmp)
        transcription = whisper_mod.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as file:
            file.write(transcription)

#Spliting the transcription 
loader = TextLoader("transcription.txt")
txt_docs = loader.load()
txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = txt_splitter.split_documents(txt_docs)

#Finding relevant parts of transcription through embeddings & vector store
embeddings = OpenAIEmbeddings()
index_name = "youtube-index" 
pinecone = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
