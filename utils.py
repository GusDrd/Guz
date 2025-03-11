from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from fastapi import HTTPException, Header
from langchain_core.tools import tool
from dotenv import load_dotenv
from fastapi import Request
import datetime
import jwt
import os


# ----- Load environment variables -----
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)


# -------------------------------------
# ========== DOCUMENT STORES ==========
# -------------------------------------

# -- PDF --
pdf_loader = PyPDFLoader(
    file_path = "./documents/Resume.pdf"
)
pdf_docs = pdf_loader.load()

# -- JSON --
json_loader = JSONLoader(
    file_path="./documents/QA.json",
    jq_schema=".question_answers[]",
    text_content=False
)
json_docs = json_loader.load()


# Create document splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
pdf_splits = text_splitter.split_documents(pdf_docs)
json_splits = text_splitter.split_documents(json_docs)

all_splits = pdf_splits + json_splits


# Embeddings & Vector Stores
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)


# RETRIEVER Agent tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# -------------------------------------------------
# ========== CONVERSATION (DE)SERIALIZER ==========
# -------------------------------------------------
def serialize_message(message):
    """Convert message objects to JSON serializable format."""
    return {
        "type": message.__class__.__name__,  # Store the message type (e.g., SystemMessage, HumanMessage)
        "content": message.content,  # Store the text content
        "additional_kwargs": message.additional_kwargs,  # Store extra metadata
        "response_metadata": message.response_metadata,  # Store metadata if needed
    }

def deserialize_message(message_data):
    """Convert JSON back to LangChain message objects."""
    msg_type = message_data["type"]
    if msg_type == "SystemMessage":
        return SystemMessage(content=message_data["content"], 
                             additional_kwargs=message_data["additional_kwargs"], 
                             response_metadata=message_data["response_metadata"])
    elif msg_type == "HumanMessage":
        return HumanMessage(content=message_data["content"], 
                            additional_kwargs=message_data["additional_kwargs"], 
                            response_metadata=message_data["response_metadata"])
    elif msg_type == "AIMessage":
        return AIMessage(content=message_data["content"], 
                         additional_kwargs=message_data["additional_kwargs"], 
                         response_metadata=message_data["response_metadata"])
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
    

# --------------------------------------
# ========== TOKEN MANAGEMENT ==========
# --------------------------------------
token_key = os.getenv('TOKEN_KEY')

def generate_jwt():
    """Generate a JWT token with a 10-minute expiration."""
    expiration_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)
    payload = {"exp": expiration_time}
    token = jwt.encode(payload, token_key, algorithm="HS256")
    return token, expiration_time

def validate_jwt(token: str = Header(None)):
    """Middleware to validate JWT token from headers."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        jwt.decode(token, token_key, algorithms=["HS256"])
        return True  # Token is valid
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ------------------------------------------
# ========== REQUEST IP RETRIEVAL ==========
# ------------------------------------------
def ip_key_func(request: Request):
    """Extracts the correct IP address from headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]  # First IP in the list is the real client IP
    return request.client.host  # Fallback if no proxy is used