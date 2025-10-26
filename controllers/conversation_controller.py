from typing import List, Dict
from fastapi import HTTPException
from google import genai
from google.genai import types
import uuid
import os
import json
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

from schemas.models import (
    StartConversationRequest, 
    ContinueConversationRequest, 
    StartConversationResponse,
    ConversationResponse,
    ConversationHistory,
    Message
)

# Configuration from environment variables
RAG_CORPUS = os.getenv("RAG_CORPUS")
MODEL_ID = os.getenv("MODEL_ID")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# In-memory storage (replace with database later)
conversations = {}

def read_system_prompt() -> str:
    """Read system prompt from text file"""
    with open("/app/prompts/system_prompt.txt", "r", encoding="utf-8") as file:
        return file.read().strip()

def read_report_task() -> str:
    """Read report task instructions from text file"""
    with open("/app/prompts/report_task.txt", "r", encoding="utf-8") as file:
        return file.read().strip()

def read_study_metrics(study_code: str) -> Dict:
    """Read study metrics from JSON file"""
    study_dir = Path("storage") / "studies" / study_code
    metrics_file = study_dir / "metrics.json"
    
    with open(metrics_file, "r", encoding="utf-8") as file:
        return json.load(file)

def save_report_md(study_code: str, report_content: str) -> str:
    """Save the generated report as markdown file in the study directory"""
    study_dir = Path("storage") / "studies" / study_code
    study_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = study_dir / "report.md"
    
    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_content)
    
    return str(report_file)

def convert_md_to_pdf(study_code: str, markdown_content: str) -> str:
    """Convert markdown content to PDF and save it in the study directory"""
    study_dir = Path("storage") / "studies" / study_code
    study_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_file = study_dir / "report.pdf"
    
    # Create PDF with table of contents from headings up to level 3
    pdf = MarkdownPdf(toc_level=3, optimize=True)
    
    # Add the markdown content as a section with custom CSS for better formatting
    css = """
    body { font-family: Arial, sans-serif; line-height: 1.6; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
    h3 { color: #7f8c8d; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    table, th, td { border: 1px solid #bdc3c7; }
    th { background-color: #ecf0f1; font-weight: bold; padding: 12px; text-align: left; }
    td { padding: 10px; }
    strong { color: #2c3e50; }
    ul { margin: 10px 0; }
    li { margin: 5px 0; }
    """
    
    pdf.add_section(Section(markdown_content), user_css=css)
    
    # Set PDF metadata
    pdf.meta["title"] = f"Brain Tumor Analysis Report - {study_code}"
    pdf.meta["author"] = "LLM Analysis Agent"
    pdf.meta["subject"] = "Medical Brain Tumor Segmentation Analysis"
    pdf.meta["creator"] = "LLM Analysis Agent API"
    
    # Save the PDF
    pdf.save(str(pdf_file))
    
    return str(pdf_file)

def generate_response(messages: List[Dict]) -> str:
    """Generate response using Google genai with RAG"""
    
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

async def start_conversation(req: StartConversationRequest) -> StartConversationResponse:
    """Start a new conversation with RAG"""
    
    conversation_id = str(uuid.uuid4())
    
    messages = []
    
    system_prompt = req.system_prompt if req.system_prompt else read_system_prompt()
    messages.append({"role": "system", "content": system_prompt})
    
    study_metrics = read_study_metrics(req.study_code)
    report_task = read_report_task()
    
    # Combine report task and study metrics
    enhanced_prompt = f"""
    {report_task}

    Study Metrics Data:
    {json.dumps(study_metrics, indent=2)}
    """
    
    # Add enhanced user message
    messages.append({"role": "user", "content": enhanced_prompt})
    
    # Generate response
    response_text = generate_response(messages)
    
    # Extract JSON from markdown code blocks and then extract report_md
    if response_text.strip().startswith("```json"):
        start_marker = "```json"
        end_marker = "```"
        start_idx = response_text.find(start_marker) + len(start_marker)
        end_idx = response_text.rfind(end_marker)
        json_content = response_text[start_idx:end_idx].strip()
    else:
        json_content = response_text.strip()
    
    # Parse JSON and extract report_md
    response_data = json.loads(json_content)
    
    # Extract the report_md content from the patient data
    if isinstance(response_data, list) and len(response_data) > 0:
        patient_data = response_data[0]
        if "report_md" in patient_data:
            report_md_content = patient_data["report_md"]
            report_file_path = save_report_md(req.study_code, report_md_content)
            pdf_file_path = convert_md_to_pdf(req.study_code, report_md_content)
        else:
            report_file_path = save_report_md(req.study_code, response_text)
    else:
        report_file_path = save_report_md(req.study_code, response_text)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response_text})
    
    # Store conversation
    conversations[conversation_id] = {
        "messages": messages,
        "system_prompt": req.system_prompt,
        "study_id": req.study_id
    }
    
    return StartConversationResponse(
        conversation_id=conversation_id,
        study_id=req.study_id
    )

async def continue_conversation(conversation_id: str, req: ContinueConversationRequest) -> ConversationResponse:
    """Continue an existing conversation"""
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Verify study_id matches
    if conversations[conversation_id]["study_id"] != req.study_id:
        raise HTTPException(status_code=403, detail="Study ID mismatch")
    
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
        messages=[Message(**msg) for msg in messages],
        study_id=req.study_id
    )

async def get_conversation(conversation_id: str) -> ConversationHistory:
    """Get conversation history"""
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv_data = conversations[conversation_id]
    
    return ConversationHistory(
        conversation_id=conversation_id,
        study_id=conv_data["study_id"],
        messages=[Message(**msg) for msg in conv_data["messages"]]
    )

async def health_check() -> Dict:
    """Health check endpoint that verifies configuration and API access"""
    try:
        config_status = {
            "RAG_CORPUS": bool(RAG_CORPUS),
            "MODEL_ID": bool(MODEL_ID),
            "GOOGLE_CLOUD_API_KEY": bool(GOOGLE_CLOUD_API_KEY) and len(GOOGLE_CLOUD_API_KEY) > 10
        }

        try:
            client = genai.Client(
                vertexai=True,
                api_key=GOOGLE_CLOUD_API_KEY
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
