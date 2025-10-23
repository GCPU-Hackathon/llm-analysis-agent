from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from google import genai
from google.genai import types
import base64
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

conversations = {}

class Message(BaseModel):
    role: str
    content: str

class StartConversationRequest(BaseModel):
    question: str
    system_prompt: Optional[str] = None

class ContinueConversationRequest(BaseModel):
    question: str

class ConversationResponse(BaseModel):
    conversation_id: str
    response: str
    messages: List[Message]

# Configuration from environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RAG_CORPUS = os.getenv("RAG_CORPUS")
MODEL_ID = os.getenv("MODEL_ID")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# Validate required environment variables
required_vars = {
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
    "RAG_CORPUS": RAG_CORPUS,
    "MODEL_ID": MODEL_ID,
    "GOOGLE_CLOUD_API_KEY": GOOGLE_CLOUD_API_KEY
}

missing_vars = [var for var, val in required_vars.items() if not val]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def generate_response(messages: List[Dict]) -> str:
    """Generate response using Google genai with RAG"""
    
    # Initialize the client
    client = genai.Client(
        vertexai=True,
        api_key=GOOGLE_CLOUD_API_KEY,
    )

    # Convert messages to genai format
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))

    # Configure RAG tools
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus=RAG_CORPUS
                        )
                    ],
                )
            )
        )
    ]

    # Generation configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        tools=tools,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )

    # Generate response
    full_response = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL_ID,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            full_response += chunk.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    return full_response

@app.post("/conversation/start", response_model=ConversationResponse)
async def start_conversation(req: StartConversationRequest):
    """Start a new conversation with RAG"""
    
    conversation_id = str(uuid.uuid4())
    
    messages = []
    
    # Add system prompt if provided
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    
    # Add user question
    messages.append({"role": "user", "content": req.question})
    
    # Generate response
    response_text = generate_response(messages)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response_text})
    
    # Store conversation
    conversations[conversation_id] = {
        "messages": messages,
        "system_prompt": req.system_prompt
    }
    
    return ConversationResponse(
        conversation_id=conversation_id,
        response=response_text,
        messages=[Message(**msg) for msg in messages]
    )

@app.post("/conversation/{conversation_id}/continue", response_model=ConversationResponse)
async def continue_conversation(conversation_id: str, req: ContinueConversationRequest):
    """Continue an existing conversation"""
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get existing messages
    messages = conversations[conversation_id]["messages"].copy()
    
    # Add new user message
    messages.append({"role": "user", "content": req.question})
    
    # Generate response with full history
    response_text = generate_response(messages)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response_text})
    
    # Update storage
    conversations[conversation_id]["messages"] = messages
    
    return ConversationResponse(
        conversation_id=conversation_id,
        response=response_text,
        messages=[Message(**msg) for msg in messages]
    )

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]["messages"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies configuration and API access"""
    try:
        # Check if all environment variables are loaded
        config_status = {
            "PROJECT_ID": bool(PROJECT_ID),
            "LOCATION": bool(LOCATION),
            "RAG_CORPUS": bool(RAG_CORPUS),
            "MODEL_ID": bool(MODEL_ID),
            "GOOGLE_CLOUD_API_KEY": bool(GOOGLE_CLOUD_API_KEY) and len(GOOGLE_CLOUD_API_KEY) > 10
        }
        
        # Try to initialize client to verify API key
        try:
            client = genai.Client(
                vertexai=True,
                api_key=GOOGLE_CLOUD_API_KEY,
            )
            api_status = "api_key_valid"
        except Exception as e:
            api_status = f"api_key_invalid: {str(e)}"
        
        return {
            "status": "healthy",
            "config": config_status,
            "api_status": api_status,
            "model": MODEL_ID,
            "rag_corpus": RAG_CORPUS
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Analysis Agent API",
        "endpoints": {
            "start_conversation": "POST /conversation/start",
            "continue_conversation": "POST /conversation/{conversation_id}/continue",
            "get_conversation": "GET /conversation/{conversation_id}",
            "health_check": "GET /health"
        },
        "model": MODEL_ID,
        "rag_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)