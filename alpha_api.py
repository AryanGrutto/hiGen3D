# Alpha API Server for Hi3DGen
# Provides REST API endpoints for 3D mesh generation

import os
os.environ['SPCONV_ALGO'] = 'native'

import uuid
import base64
import threading
from typing import Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

from hi3dgen.pipelines import Hi3DGenPipeline


# =============================================================================
# Configuration
# =============================================================================

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# =============================================================================
# Concurrency Configuration
# =============================================================================
# Adjust MAX_CONCURRENT_JOBS based on your GPU VRAM:
#   - 24 GB VRAM: 1-2 concurrent jobs
#   - 48 GB VRAM: 2-3 concurrent jobs  
#   - 80 GB VRAM: 4-5 concurrent jobs
# Set to 1 for sequential processing (safest)

MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))

# Thread pool for background jobs
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Semaphore to limit concurrent GPU access
gpu_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

# Lock for thread-safe pipeline initialization only
pipeline_init_lock = threading.Lock()


# =============================================================================
# Job Storage
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.mesh_path: Optional[str] = None
        self.error: Optional[str] = None


# In-memory job storage (use Redis/DB for production)
jobs: dict[str, Job] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class SendRequest(BaseModel):
    image_base64: str  # Base64 encoded image
    seed: int = -1
    ss_guidance_strength: float = 3.0
    ss_sampling_steps: int = 50
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 6


class SendResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusRequest(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    mesh_base64: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Alpha 3D Generation API",
    description="API for generating 3D meshes from images using Hi3DGen",
    version="1.0.0"
)


# =============================================================================
# Pipeline Initialization (lazy loading)
# =============================================================================

hi3dgen_pipeline = None
normal_predictor = None


def get_pipeline():
    """Lazy load the pipeline on first use (thread-safe initialization)."""
    global hi3dgen_pipeline, normal_predictor
    
    # Double-checked locking for thread-safe lazy initialization
    if hi3dgen_pipeline is None or normal_predictor is None:
        with pipeline_init_lock:
            if hi3dgen_pipeline is None:
                print("Loading Hi3DGen pipeline...")
                hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(
                    os.path.join(WEIGHTS_DIR, "trellis-normal-v0-1")
                )
                hi3dgen_pipeline.cuda()
                print("Hi3DGen pipeline loaded.")
            
            if normal_predictor is None:
                print("Loading normal predictor...")
                try:
                    normal_predictor = torch.hub.load(
                        os.path.join(torch.hub.get_dir(), 'hugoycj_StableNormal_main'),
                        "StableNormal_turbo",
                        yoso_version='yoso-normal-v1-8-1',
                        source='local',
                        local_cache_dir='./weights',
                        pretrained=True
                    )
                except:
                    normal_predictor = torch.hub.load(
                        "hugoycj/StableNormal",
                        "StableNormal_turbo",
                        trust_repo=True,
                        yoso_version='yoso-normal-v1-8-1',
                        local_cache_dir='./weights'
                    )
                print("Normal predictor loaded.")
    
    return hi3dgen_pipeline, normal_predictor


# =============================================================================
# Generation Logic
# =============================================================================

def generate_mesh(job_id: str, image: Image.Image, params: dict):
    """Background task to generate a 3D mesh."""
    job = jobs.get(job_id)
    if not job:
        return
    
    try:
        job.status = JobStatus.GENERATING
        
        # Acquire semaphore slot for GPU access (allows MAX_CONCURRENT_JOBS parallel)
        with gpu_semaphore:
            print(f"Job {job_id}: Acquired GPU slot. Starting generation...")
            
            pipeline, predictor = get_pipeline()
            
            # Preprocess image
            processed_image = pipeline.preprocess_image(image, resolution=1024)
            
            # Generate normal map
            normal_image = predictor(
                processed_image,
                resolution=768,
                match_input_resolution=True,
                data_type='object'
            )
            
            # Set seed (use job-specific generator to avoid global state issues)
            seed = params.get('seed', -1)
            if seed == -1:
                seed = np.random.randint(0, MAX_SEED)
            
            # Generate mesh
            outputs = pipeline.run(
                normal_image,
                seed=seed,
                formats=["mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": params.get('ss_sampling_steps', 50),
                    "cfg_strength": params.get('ss_guidance_strength', 3.0),
                },
                slat_sampler_params={
                    "steps": params.get('slat_sampling_steps', 6),
                    "cfg_strength": params.get('slat_guidance_strength', 3.0),
                },
            )
            
            generated_mesh = outputs['mesh'][0]
        
        # Save mesh (outside GPU semaphore - doesn't need GPU)
        output_dir = os.path.join(TMP_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "mesh.glb")
        
        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        trimesh_mesh.export(mesh_path)
        
        # Update job status
        job.mesh_path = mesh_path
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        
        print(f"Job {job_id}: Completed successfully.")
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        print(f"Job {job_id}: Failed - {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/send", response_model=SendResponse)
async def send(request: SendRequest):
    """
    Start a mesh generation pipeline in the background.
    Returns a UUID for tracking the process.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = Job(job_id)
    jobs[job_id] = job
    
    # Prepare params
    params = {
        'seed': request.seed,
        'ss_guidance_strength': request.ss_guidance_strength,
        'ss_sampling_steps': request.ss_sampling_steps,
        'slat_guidance_strength': request.slat_guidance_strength,
        'slat_sampling_steps': request.slat_sampling_steps,
    }
    
    # Submit to background executor
    executor.submit(generate_mesh, job_id, image, params)
    
    return SendResponse(
        job_id=job_id,
        status="pending",
        message="Mesh generation started. Use /status to check progress."
    )


@app.post("/status", response_model=StatusResponse)
async def status(request: StatusRequest):
    """
    Check the status of a mesh generation job.
    Returns the mesh as base64 if completed.
    """
    job = jobs.get(request.job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = StatusResponse(
        job_id=job.job_id,
        status=job.status.value
    )
    
    if job.status == JobStatus.COMPLETED and job.mesh_path:
        # Read mesh and encode as base64
        try:
            with open(job.mesh_path, 'rb') as f:
                mesh_data = f.read()
            response.mesh_base64 = base64.b64encode(mesh_data).decode('utf-8')
        except Exception as e:
            response.status = "failed"
            response.error = f"Failed to read mesh: {str(e)}"
    
    elif job.status == JobStatus.FAILED:
        response.error = job.error
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns 200 if the server is running.
    """
    return HealthResponse(
        status="ok",
        message="Server is running"
    )


@app.get("/info")
async def info():
    """
    Get server configuration and current queue status.
    """
    # Count jobs by status
    status_counts = {status.value: 0 for status in JobStatus}
    for job in jobs.values():
        status_counts[job.status.value] += 1
    
    # Get GPU memory info if available
    gpu_info = {}
    try:
        if torch.cuda.is_available():
            gpu_info = {
                "device": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "allocated_memory_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "reserved_memory_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            }
    except:
        pass
    
    return {
        "config": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "available_slots": gpu_semaphore._value,
        },
        "queue": {
            "total_jobs": len(jobs),
            **status_counts
        },
        "gpu": gpu_info
    }


# =============================================================================
# Optional: Cleanup endpoint
# =============================================================================

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = jobs.pop(job_id, None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up files
    if job.mesh_path and os.path.exists(job.mesh_path):
        import shutil
        output_dir = os.path.dirname(job.mesh_path)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    return {"status": "deleted", "job_id": job_id}


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Pre-load pipeline on startup
    print("Pre-loading models...")
    get_pipeline()
    print("Models loaded. Starting server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

