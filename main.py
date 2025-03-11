from utils import serialize_message, deserialize_message, generate_jwt, validate_jwt, ip_key_func
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph
from RAG import query_or_respond, tools, generate
from langgraph.prebuilt import tools_condition
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from starlette.requests import Request
from langgraph.graph import END
from pydantic import BaseModel
import redis
import json


# ----- Redis Setup -----
redis_client = redis.StrictRedis(host='redis', port=6379, db=0, decode_responses=True)

# ----- FastAPI Setup -----
app = FastAPI()
limiter = Limiter(key_func=ip_key_func, storage_uri="redis://redis:6379/0")
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ----- Graph Structure -----
graph_builder = StateGraph(MessagesState)

# Existing nodes
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# Set entry point
graph_builder.set_entry_point("query_or_respond")

# Conditional edges for deciding whether to trigger retrieval
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

# If context is retrieved, proceed to generate a response
graph_builder.add_edge("tools", "generate")

# Edge to handle end of the flow
graph_builder.add_edge("generate", END)

# Main LLM graph endpoint
graph = graph_builder.compile()


# ----- System Prompt -----
system_prompt = SystemMessage(content="""
Your name is Guz, Gus’s personal AI assistant.  
You respond in a conversational tone and answer questions about Gus based on available knowledge.
- If you don’t have specific information, offer general knowledge and encourage further interaction.
- For topics you're unsure about, respond confidently with what you know and acknowledge any gaps in knowledge.
- Be open and friendly, but **don’t drift into topics that aren’t directly related to Gus**.
- Structure your answers properly using markdown to make them visually pleasing to read.
- If the provided context is specific, also be specific, but don't exactly copy the context.
- Be smooth in how you use the context, lightly extrapolate information if needed but **don’t invent anything**.

If the conversation drifts from Gus, politely steer it back. You can redirect by saying something like:
- "Let’s focus back on Gus."
- "I can tell you more about Gus’s interests!"
- "Would you like to hear more about Gus's latest project?"

But most importantly: **DO NOT drift away from Gus!**
If the conversation starts talking about anything else than Gus, stop and get the conversation back to Gus. Do not engage in unrelated topics under any circumstances.
""")


# ----- REDIS Load/Save/Delete Conversations -----
def get_user_session(user_id: str):
    """Retrieve user session from Redis or create a new one if not found"""
    session_data = redis_client.get(f"user_session:{user_id}")
    if session_data:
        session_data = json.loads(session_data)  # Deserialize the data
        return {"messages": [deserialize_message(msg) for msg in session_data["messages"]]}
    else:
        return {"messages": [system_prompt]} # Return new messages history

def save_user_session(user_id: str, session_data: dict):
    """Save user session to Redis"""
    serialized_data = {"messages": [serialize_message(msg) for msg in session_data["messages"]]}
    redis_client.set(f"user_session:{user_id}", json.dumps(serialized_data))  # Serialize data

def delete_user_session(user_id: str):
    """Delete a user's session from Redis"""
    redis_client.delete(f"user_session:{user_id}")


# ----- FastAPI Endpoints -----
origins = ["https://www.augustindirand.com", "https://augustindirand.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Request model
class ChatRequest(BaseModel):
    user_id: str
    query: str

# API Endpoints
@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, chat_request: ChatRequest, valid: bool = Depends(validate_jwt)):
    """Main endpoint with the chatbot"""
    # Retrieve conversation from redis
    try:
        user_config = get_user_session(chat_request.user_id)
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis service unavailable")
    
    # Append new query, and send to chatbot for response
    prompt = user_config["messages"] + [HumanMessage(chat_request.query)]
    response = graph.invoke({"messages": prompt})["messages"][-1].content
    
    # Append chatbot response and save new message history
    new_history = prompt + [AIMessage(response)]
    save_user_session(chat_request.user_id, {"messages": new_history})

    return {"response": response}

@app.delete("/chat/{user_id}")
def clear_session(user_id: str):
    """Clear session conversation history"""
    try:
        delete_user_session(user_id)
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis service unavailable") 
    return {"message": "User session cleared."}

@app.post("/session")
@limiter.limit("8/minute")
def get_token(request: Request):
    """Generate and return a JWT"""
    token, expiration_time = generate_jwt()
    return {"token": token, "expires_at": expiration_time}
