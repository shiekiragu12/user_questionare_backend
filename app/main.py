from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Dynamic LLM API",
    description="Real-time AI responses for the user questionare",
    version="1.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  
        "http://127.0.0.1:3000",
        "*"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    model: str = "gpt-3.5-turbo"  
    temperature: float = 0.7  

@app.post("/api/query")
async def process_query(query: Query):
    """Handle dynamic LLM queries with proper validation"""
    if not query.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )

    try:
        # Dynamic LLM call
        response = openai.ChatCompletion.create(
            model=query.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query.question}
            ],
            temperature=query.temperature,
        )
        
        return {
            "answer": response.choices[0].message.content,
            "model": query.model,
            "tokens_used": response.usage.total_tokens,
            "timestamp": datetime.now().isoformat()  
        }
        
    except openai.error.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI service error: {str(e)}"
        )