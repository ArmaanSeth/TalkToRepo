from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import stat
from dotenv import load_dotenv
import git
import os


class TalkToRepo:
    def __init__(self) -> None:
        load_dotenv()
        self.chat_history=""
        self.allowed_extensions = [".ipynb", ".py", ".md"]
        
        self.query_transform_prompt = ChatPromptTemplate.from_template("""chat-history: {chat_history}\n Given the above conversation, generate a search query to search for relevant information based on the chat history and the last question from a vectorstore. Just return the search query and nothing else""")
        SYSTEM_TEMPLATE = """
        You are a software engineer expert at reading github repositories. You are asked to answer questions about the code in the repository. You are given a context and a question. You have to answer the question based on the context. You have to elaborate the answer related to the question from the context in english, unless stated otherwise. You have to answer the question in a detailed manner.
        <context>
        {context}
        </context>
        question:{question}
        Elaborate the answer related to the question from the context in english, unless stated otherwise.
        Dont state about the context in the answer.
        Answer:
        """
        self.question_answering_prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    
    def build_engine(self, repo:str):
        print("Building Engine....")
        self.repo = repo
        self.repo_name = repo.split("/")[-1].split(".")[0]
        if not os.path.exists(self.repo_name):
            git.Repo.clone_from(self.repo, self.repo_name)
        
        print("Calling prepare_vectorstore....")
        self.prepare_vectorstore()
        
    def prepare_vectorstore(self):
        print("Preparing Vectorstore....")
        docs=[]
        for root, dirs, files in os.walk(self.repo_name):
            for file in files:
                if file.endswith(tuple(self.allowed_extensions)):
                    loader=TextLoader(os.path.join(root, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())

        text_splitter=CharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        chunks=text_splitter.split_documents(docs)
        self.vectorstore=FAISS.from_documents(documents=chunks,embedding=self.embeddings)
        print("Vectorstore prepared successfully....")
        print("Calling delete_repo....")
        self.delete_repo()
        print("Calling prepare_chain....")
        self.prepare_chain()

    def delete_repo(self):
        print("Deleting Repository....")
        for root, dirs, files in os.walk(self.repo_name, topdown=False):
            for name in files:
                filename = os.path.join(root, name)
                os.chmod(filename, stat.S_IWUSR)
                os.remove(filename)
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.repo_name)      
        
    def prepare_chain(self):
        print("Preparing Chain....")
        retriever=self.vectorstore.as_retriever()
        llm=self.model
        query_transforming_retriever_chain = (self.query_transform_prompt | llm | StrOutputParser() | retriever).with_config(run_name="chat_retriever_chain")
        document_chain =create_stuff_documents_chain(llm, self.question_answering_prompt)
        conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
            ).assign(
                answer=document_chain,
            )
        self.chain=conversational_retrieval_chain
        print("Chain Created successfully....")

    def get_response(self, query:str)->dict:
        self.chat_history+=f"""\nHuman: {query}"""
        res=self.chain.invoke({"chat_history":self.chat_history, 'question':query})['answer']
        self.chat_history+=f"""\nAssistant: {res}"""
        return res