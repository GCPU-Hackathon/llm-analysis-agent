from fastapi import FastAPI, Depends
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from schemas.models import (
    StartConversationRequest, 
    ContinueConversationRequest, 
    StartConversationResponse,
    ConversationResponse,
    ConversationHistory
)
from controllers import conversation_controller
from database import get_db

# Load environment variables
load_dotenv()

app = FastAPI(title="LLM Analysis Agent API", version="1.0.0")

# Validate required environment variables
required_vars = {
    "RAG_CORPUS": os.getenv("RAG_CORPUS"),
    "MODEL_ID": os.getenv("MODEL_ID"),
    "GOOGLE_CLOUD_API_KEY": os.getenv("GOOGLE_CLOUD_API_KEY")
}

missing_vars = [var for var, val in required_vars.items() if not val]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

@app.post("/conversation/start", response_model=StartConversationResponse)
async def start_conversation_route(req: StartConversationRequest, db: Session = Depends(get_db)):
    """Start a new conversation with RAG"""
    return await conversation_controller.start_conversation(req, db)

@app.post("/conversation/{conversation_id}/continue", response_model=ConversationResponse)
async def continue_conversation_route(conversation_id: str, req: ContinueConversationRequest, db: Session = Depends(get_db)):
    """Continue an existing conversation"""
    return await conversation_controller.continue_conversation(conversation_id, req, db)

@app.get("/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_route(conversation_id: str, db: Session = Depends(get_db)):
    """Get conversation history"""
    return await conversation_controller.get_conversation(conversation_id, db)

@app.get("/health")
async def health_check_route():
    """Health check endpoint that verifies configuration and API access"""
    return await conversation_controller.health_check()
