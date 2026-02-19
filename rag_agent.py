import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env file")
    exit(1)

def get_file_path():
    while True:
        file_path = input("Enter the full path to your book (PDF or TXT): ").strip()
        # Remove quotes if the user added them
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        
        if os.path.exists(file_path):
            return file_path
        print(f"File not found at: {file_path}")
        print("Please try again.")

def load_document(file_path):
    print(f"Loading document: {file_path}...")
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        print("Unsupported file format. Please use PDF or TXT.")
        return None
    
    docs = loader.load()
    print(f"Loaded {len(docs)} pages/documents.")
    return docs

def create_vector_store(docs):
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    
    print("Creating embeddings (this may take a moment)...")
    # Using a high-quality open source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building vector store...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def run_rag_agent():
    print("--- RAG Agent Setup ---")
    file_path = get_file_path()
    
    docs = load_document(file_path)
    if not docs:
        return

    vectorstore = create_vector_store(docs)
    retriever = vectorstore.as_retriever()

    # Initialize LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        api_key=api_key
    )
    # Create Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- RAG Agent Ready! ---")
    print("You can now ask questions about your book.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour Question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        try:
            print("Thinking...")
            response = rag_chain.invoke({"input": question})
            print(f"\nAnswer: {response['answer']}")
            
            # Optional: Show sources
            # print("\nSources:")
            # for doc in response["context"]:
            #     print(f"- Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content[:50]}...")
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        run_rag_agent()
    except KeyboardInterrupt:
        print("\nExiting...")
