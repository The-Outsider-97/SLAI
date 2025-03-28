from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class KnowledgeAgent:
    def __init__(self, documents):
        self.vectorstore = FAISS.from_texts(documents, OpenAIEmbeddings())
        self.qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=self.vectorstore.as_retriever())

    def query(self, question: str) -> str:
        return self.qa_chain.run(question)

    def add_knowledge(self, new_texts: list):
        self.vectorstore.add_texts(new_texts)
