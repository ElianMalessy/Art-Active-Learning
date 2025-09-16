from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import tempfile
import base64
import os
from typing import Optional, Tuple

from models.bayes import NormalBayes
from dataset.embeddings import EmbeddingDataset
from utils import device

app = FastAPI(title='Art Tinder', description='Tinder-style interface for art preference learning')

# Mount static files
app.mount('/static', StaticFiles(directory='static'), name='static')

# Templates
templates = Jinja2Templates(directory='templates')

# Global state
dataset = None
distribution = None
mask = None
current_session = None

class FeedbackRequest(BaseModel):
    image_idx: int
    rating: int  # 0 for dislike, 1 for like

class SessionData:
    def __init__(self):
        self.dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
        self.distribution = NormalBayes()
        self.mask = torch.ones(len(self.dataset), dtype=torch.bool).to(device)
        self.current_image_idx: Optional[int] = None
        self.current_image_data: Optional[str] = None
        self.current_prompt: Optional[str] = None

def get_next_image(session: SessionData) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    if not session.mask.any():
        return None, None, None
    
    # Get all latent embeddings
    dataloader = DataLoader(session.dataset, batch_size=len(session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        global_idxs = torch.nonzero(session.mask, as_tuple=False).squeeze(-1)
        candidates = latents[global_idxs]

        # Calculate probabilities
        logp = session.distribution.log_likelihood(candidates, y=1)
        p = torch.exp(logp).clamp(1e-6, 1 - 1e-6)
        
        # Calculate entropy
        bernoulli_entropy = -p*torch.log(p) - (1-p)*torch.log(1-p)
        
        # Select image with maximum entropy
        local_idx = int(torch.argmax(bernoulli_entropy).item())
        global_idx = int(global_idxs[local_idx].item())
        
        # Mark as shown
        session.mask[global_idx] = False
        
        # Get image and prompt
        image_tensor = session.dataset.get_image(global_idx)
        prompt = session.dataset.get_prompt(global_idx)
        
        # Convert image to base64 for web display
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        save_image(image_tensor, temp_path)
        
        with open(temp_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        os.unlink(temp_path)
        
        return global_idx, image_data, prompt

    return None, None, None

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/api/start')
async def start_session():
    '''Start a new session'''
    global current_session
    current_session = SessionData()
    return {'status': 'started'}

@app.get('/api/next-image')
async def get_next_image_endpoint():
    '''Get the next image to display'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session. Please start a session first.'}
    
    image_idx, image_data, prompt = get_next_image(current_session)
    
    if image_idx is None:
        return {'error': 'No more images available'}
    
    current_session.current_image_idx = image_idx
    current_session.current_image_data = image_data
    current_session.current_prompt = prompt
    
    return {
        'image_idx': image_idx,
        'image_data': image_data,
        'prompt': prompt
    }

@app.post('/api/feedback')
async def submit_feedback(feedback: FeedbackRequest):
    '''Submit user feedback for an image'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session'}
    
    if feedback.image_idx != current_session.current_image_idx:
        return {'error': 'Image ID mismatch'}
    
    # Get the latent embedding for this image
    dataloader = DataLoader(current_session.dataset, batch_size=len(current_session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        
        latent_embedding = latents[feedback.image_idx]
        current_session.distribution.update(latent_embedding, feedback.rating)
        break
    
    return {'status': f'feedback_received {current_session.distribution.counts}'}

@app.get('/api/stats')
async def get_stats():
    '''Get session statistics'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session'}
    
    total_images = len(current_session.dataset)
    remaining_images = current_session.mask.sum().item()
    shown_images = total_images - remaining_images
    
    return {
        'total_images': total_images,
        'remaining_images': remaining_images,
        'shown_images': shown_images,
        'likes': current_session.distribution.counts[1],
        'dislikes': current_session.distribution.counts[0]
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
