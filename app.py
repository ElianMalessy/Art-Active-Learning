from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import torch
from torchvision.utils import save_image

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
    image_id: int
    rating: int  # 0 for dislike, 1 for like

class SessionData:
    def __init__(self):
        self.dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
        self.distribution = NormalBayes()
        self.mask = torch.ones(len(self.dataset), dtype=torch.bool).to(device)
        self.current_image_id: Optional[int] = None
        self.current_image_data: Optional[str] = None
        self.current_prompt: Optional[str] = None

def get_next_image(session: SessionData) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    if not session.mask.any():
        return None, None, None
    
    # Get all latent embeddings
    dataloader = torch.utils.data.DataLoader(session.dataset, batch_size=len(session.dataset), shuffle=False, num_workers=0)
    
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
        x_local = int(torch.argmax(bernoulli_entropy).item())
        x_global = int(global_idxs[x_local].item())
        
        # Mark as shown
        session.mask[x_global] = False
        
        # Get image and prompt
        image_tensor = session.dataset.get_image(x_global)
        prompt = session.dataset.get_prompt(x_global)
        
        # Convert image to base64 for web display
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        save_image(image_tensor, temp_path)
        
        with open(temp_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        os.unlink(temp_path)
        
        return x_global, image_data, prompt

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
    
    image_id, image_data, prompt = get_next_image(current_session)
    
    if image_id is None:
        return {'error': 'No more images available'}
    
    current_session.current_image_id = image_id
    current_session.current_image_data = image_data
    current_session.current_prompt = prompt
    
    return {
        'image_id': image_id,
        'image_data': image_data,
        'prompt': prompt
    }

@app.post('/api/feedback')
async def submit_feedback(feedback: FeedbackRequest):
    '''Submit user feedback for an image'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session'}
    
    if feedback.image_id != current_session.current_image_id:
        return {'error': 'Image ID mismatch'}
    
    # Get the latent embedding for this image
    dataloader = torch.utils.data.DataLoader(current_session.dataset, batch_size=len(current_session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        global_idxs = torch.nonzero(current_session.mask, as_tuple=False).squeeze(-1)
        
        # Find the local index for this global index
        local_idx = None
        for i, global_idx in enumerate(global_idxs):
            if global_idx.item() == feedback.image_id:
                local_idx = i
                break
        
        if local_idx is not None:
            latent_embedding = latents[local_idx]
            current_session.distribution.update(latent_embedding, feedback.rating)
            break
    
    return {'status': 'feedback_received'}

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
