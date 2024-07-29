import os
import time
import dspy
from dsp.utils import deduplicate
from dspy.retrieve.faiss_rm import FaissRM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# os.environ["AZURE_OPENAI_API_KEY"] = ""

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""


    context = dspy.InputField(desc="may contain relevant content")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """give me a answer for user question based on context"""


    context = dspy.InputField(desc="may contain relevant content")
    question = dspy.InputField()
    answer = dspy.OutputField()



class DocQA(dspy.Module):
    def __init__(self, file_path,passages_per_hop=3, max_hops=2):
        super().__init__()
        self.cache = "cache.json"
        self.llm = dspy.AzureOpenAI(api_base="https://azureadople.openai.azure.com/",
                                   api_version="2023-09-15-preview",
                                   model="GPT-3")

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

        self.knowledge_base = self.create_knowledge_base(file_path)

    def load_documents(self, file_path):
        print("file_path", file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

        docs = text_splitter.split_documents(documents)
        document_chunks = [page_content.page_content for page_content in docs]
        print("input context Ready")
        return document_chunks

    def create_knowledge_base(self, file_path):
        print("file_path", file_path)
        document = self.load_documents(file_path)
        split_documents = self.split_documents(document)
        knowledge_base = FaissRM(split_documents)
        return knowledge_base

    def run(self,question):
        dspy.settings.configure(lm=self.llm, rm=self.knowledge_base)


        passages = self.retrieve(question).passages
        context = deduplicate(passages)

        pred = self.generate_answer(context=context, question=question)
        return pred.answer
