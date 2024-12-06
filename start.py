import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
import time
from collections import defaultdict
import re
import os
import uvicorn
from datetime import datetime
import threading
import queue
import logging
import multiprocessing
from accelerate import cpu_offload
from dotenv import load_dotenv, find_dotenv
load_dotenv(
    find_dotenv(),
    override=True
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get CPU count for optimal threading
CPU_COUNT = multiprocessing.cpu_count()
BATCH_SIZE = 1
NUM_WORKERS = max(1, CPU_COUNT - 1)  # Leave one CPU core free

# Create queues for handling requests
request_queue = queue.Queue()
result_queue = queue.Queue()

# Initialize FastAPI app
app = FastAPI(
    title="Stable Diffusion CPU API",
    description="Optimized CPU-based Image generation API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.getenv("API_KEY", "your-secret-key-here")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: Optional[str] = Field(default="", max_length=500)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        v = re.sub(r'\s+', ' ', v).strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        return v

class GenerationResponse(BaseModel):
    status: str
    message: Optional[str] = None
    image: Optional[str] = None
    generated_at: datetime

def initialize_pipeline():
    # Load in 8-bit mode for memory efficiency
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Use float32 for CPU
        safety_checker=None,
        low_cpu_mem_usage=True,
    )
    
    # CPU optimizations
    pipeline.enable_sequential_cpu_offload()
    pipeline.enable_attention_slicing(1)
    pipeline.enable_vae_slicing()
    
    return pipeline

class WorkerPool:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.workers = []
        self.pipeline = initialize_pipeline()

    def start(self):
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self):
        while True:
            try:
                request_data = request_queue.get()
                if request_data is None:
                    break

                prompt, negative_prompt, guidance_scale, num_inference_steps = request_data
                
                # Generate image
                with torch.no_grad():
                    image = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=1,
                    ).images[0]

                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=90, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result_queue.put({
                    "status": "success",
                    "image": img_str,
                    "generated_at": datetime.utcnow()
                })

            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")
                result_queue.put({
                    "status": "error",
                    "message": str(e),
                    "generated_at": datetime.utcnow()
                })
            finally:
                request_queue.task_done()

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

@app.post("/generate/", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Add request to queue
        request_queue.put((
            request.prompt,
            request.negative_prompt,
            request.guidance_scale,
            request.num_inference_steps,
        ))
        
        # Wait for result
        result = result_queue.get(timeout=300)  # 5 minutes timeout
        return result
        
    except queue.Empty:
        raise HTTPException(
            status_code=504,
            detail="Request timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "cpu_count": CPU_COUNT,
        "active_workers": NUM_WORKERS,
        "timestamp": datetime.utcnow()
    }

if __name__ == "__main__":
    # Initialize worker pool
    worker_pool = WorkerPool(NUM_WORKERS)
    worker_pool.start()
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Use only 1 worker as we're using threading
    )