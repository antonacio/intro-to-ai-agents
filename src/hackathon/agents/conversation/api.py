"""FastAPI backend for the legal conversation agent."""

import os
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from hackathon.agents.conversation import MultiAgentLegalGraph


# Request/Response models
class ConversationStartRequest(BaseModel):
    """Request to start a new conversation."""
    thread_id: Optional[str] = Field(default=None, description="Optional thread ID. If not provided, one will be generated.")


class ConversationStartResponse(BaseModel):
    """Response when starting a new conversation."""
    thread_id: str
    message: str
    status: str = "active"


class ConversationMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    thread_id: str = Field(..., description="The conversation thread ID")
    message: str = Field(..., description="The user's message")


class ConversationMessageResponse(BaseModel):
    """Response after processing a message."""
    thread_id: str
    message: str
    conversation_complete: bool = False
    legal_area: Optional[str] = None
    status: str = "active"
    draft_ready: bool = False


class ConversationStatusRequest(BaseModel):
    """Request to check conversation status."""
    thread_id: str


class ConversationStatusResponse(BaseModel):
    """Response with conversation status."""
    thread_id: str
    status: str
    conversation_complete: bool
    legal_area: Optional[str] = None
    message_count: int
    draft_ready: bool = False


# Create FastAPI app
app = FastAPI(
    title="Legal Conversation API",
    description="API for Iris, the AI front-of-house legal assistant",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow instance (in production, use proper dependency injection)
llm = None
workflow = None

# In-memory storage for conversation states (in production, use Redis or database)
conversation_states: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the workflow on startup."""
    global llm, workflow
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize LLM and workflow
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    workflow = MultiAgentLegalGraph(llm=llm, use_memory=True)
    
    print("✅ Legal Conversation API initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "legal-conversation-api"}


@app.post("/conversation/start", response_model=ConversationStartResponse)
async def start_conversation(request: ConversationStartRequest):
    """
    Start a new conversation with Iris.
    
    Returns Iris's introduction message and a thread ID for the conversation.
    """
    # Generate thread ID if not provided
    thread_id = request.thread_id or f"conv_{uuid.uuid4()}"
    
    try:
        # Run workflow with empty input to get Iris's introduction
        result = workflow.run({"messages": []}, thread_id=thread_id)
        
        # Extract Iris's introduction
        messages = result.get("messages", [])
        if messages and hasattr(messages[-1], 'content'):
            iris_message = messages[-1].content
        else:
            iris_message = "Hello! I'm Iris, the AI front-of-house for our law firm. How can I help you today?"
        
        # Store conversation state
        conversation_states[thread_id] = {
            "status": "active",
            "message_count": len(messages),
            "last_state": result
        }
        
        return ConversationStartResponse(
            thread_id=thread_id,
            message=iris_message,
            status="active"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")


@app.post("/conversation/message", response_model=ConversationMessageResponse)
async def send_message(request: ConversationMessageRequest):
    """
    Send a message in an existing conversation.
    
    Returns Iris's response and conversation status.
    """
    thread_id = request.thread_id
    
    # Check if conversation exists
    if thread_id not in conversation_states:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found")
    
    # Check if conversation is already complete
    if conversation_states[thread_id].get("status") == "completed":
        raise HTTPException(status_code=400, detail="Conversation has already ended")
    
    try:
        # Prepare input with user message
        input_data = {"messages": [HumanMessage(content=request.message)]}
        
        # Run workflow
        result = workflow.run(input_data, thread_id=thread_id)
        
        # Extract Iris's response
        messages = result.get("messages", [])
        iris_message = ""
        
        # Find the last AI message
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'content'):
                iris_message = msg.content
                break
        
        # Check if conversation is complete
        conversation_complete = result.get("conversation_complete", False)
        drafting_complete = result.get("drafting_complete", False)
        
        # Update conversation state
        conversation_states[thread_id].update({
            "status": "completed" if conversation_complete else "active",
            "message_count": len(messages),
            "last_state": result,
            "legal_area": result.get("legal_area"),
            "draft_ready": drafting_complete
        })
        
        return ConversationMessageResponse(
            thread_id=thread_id,
            message=iris_message,
            conversation_complete=conversation_complete,
            legal_area=result.get("legal_area"),
            status=conversation_states[thread_id]["status"],
            draft_ready=drafting_complete
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/conversation/status", response_model=ConversationStatusResponse)
async def get_conversation_status(request: ConversationStatusRequest):
    """
    Get the current status of a conversation.
    """
    thread_id = request.thread_id
    
    if thread_id not in conversation_states:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found")
    
    state = conversation_states[thread_id]
    last_state = state.get("last_state", {})
    
    return ConversationStatusResponse(
        thread_id=thread_id,
        status=state.get("status", "unknown"),
        conversation_complete=last_state.get("conversation_complete", False),
        legal_area=last_state.get("legal_area"),
        message_count=state.get("message_count", 0),
        draft_ready=state.get("draft_ready", False)
    )


@app.get("/conversation/{thread_id}/history")
async def get_conversation_history(thread_id: str):
    """
    Get the full conversation history for a thread.
    """
    if thread_id not in conversation_states:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found")
    
    state = conversation_states[thread_id]
    last_state = state.get("last_state", {})
    messages = last_state.get("messages", [])
    
    # Format messages for response
    history = []
    for msg in messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            history.append({
                "role": msg.type,
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            })
    
    return {
        "thread_id": thread_id,
        "history": history,
        "message_count": len(history),
        "status": state.get("status", "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)