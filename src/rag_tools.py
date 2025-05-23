import os
from typing import List, Dict
import langchain
import langchain_community
import langchain_text_splitters
import langchain_openai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_documents(file_path: str):
    """Load and chunk PDF documents with metadata tracking"""
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        add_start_index=True  # Critical for source tracking
    )
    return loader.load_and_split(text_splitter)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def initialize_vector_store(documents):
    """Create FAISS index with document embeddings"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.from_documents(documents, embeddings)


from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

def create_verification_chain(vector_store):
    """Create QA chain with source tracking capabilities"""
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 5}
        ),
        return_source_documents=True
    )

class PDFAssertionValidator:
    def __init__(self, pdf_directory: str):
        self.documents = []
        self.vector_store = None
        self.qa_chain = None

        # Load and process documents
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                self.documents += load_pdf_documents(file_path)

        # Initialize vector store
        self.vector_store = initialize_vector_store(self.documents)

        # Create verification chain
        self.qa_chain = create_verification_chain(self.vector_store)

    def validate_assertion(self, assertion: str) -> Dict:
        """Validate an assertion against stored documents"""
        result = self.qa_chain.invoke({"question": assertion})
        return {
            "assertion": assertion,
            "validation": result["answer"],
            "confidence": self._calculate_confidence(result),
            "sources": self._process_sources(result["source_documents"])
        }

    def _calculate_confidence(self, result) -> float:
        """Calculate validation confidence score"""
        return min(1.0, len(result["source_documents"]) / 5)

    def _process_sources(self, documents) -> List[Dict]:
        """Extract source metadata"""
        return [{
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "text": doc.page_content[:500] + "..."
        } for doc in documents]
