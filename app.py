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

from models.bradley_terry import BradleyTerryModel
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

class ComparisonRequest(BaseModel):
    pair_id: str
    preferred: str  # 'a' or 'b'

class SessionData:
    def __init__(self):
        self.dataset = EmbeddingDataset('latent_embeddings', return_metadata=True)
        self.model = BradleyTerryModel(input_dim=64)  # Using Bradley-Terry model
        self.mask = torch.ones(len(self.dataset), dtype=torch.bool).to(device)
        self.current_pair_indices: Optional[Tuple[int, int]] = None
        self.current_image_a_data: Optional[str] = None
        self.current_image_b_data: Optional[str] = None
        self.current_prompt_a: Optional[str] = None
        self.current_prompt_b: Optional[str] = None
        self.comparisons_made = 0

def get_next_comparison_pair(session: SessionData) -> Tuple[Optional[Tuple[int, int]], Optional[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    if session.mask.sum() < 2:  # Need at least 2 images for comparison
        return None, None, None, None, None, None
    
    # Get all latent embeddings
    dataloader = DataLoader(session.dataset, batch_size=len(session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        global_idxs = torch.nonzero(session.mask, as_tuple=False).squeeze(-1)
        candidates = latents[global_idxs]

        # Select most informative pair using Bradley-Terry model
        local_idx_a, local_idx_b = session.model.select_most_informative_pair(candidates)
        
        if local_idx_a is None or local_idx_b is None:
            return None, None, None, None, None, None
            
        global_idx_a = int(global_idxs[local_idx_a].item())
        global_idx_b = int(global_idxs[local_idx_b].item())
        
        # Mark both as shown
        session.mask[global_idx_a] = False
        session.mask[global_idx_b] = False
        
        # Get images and prompts
        image_tensor_a = session.dataset.get_image(global_idx_a)
        image_tensor_b = session.dataset.get_image(global_idx_b)
        prompt_a = session.dataset.get_prompt(global_idx_a)
        prompt_b = session.dataset.get_prompt(global_idx_b)
        
        # Convert images to base64 for web display
        def tensor_to_base64(image_tensor):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            save_image(image_tensor, temp_path)
            with open(temp_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            os.unlink(temp_path)
            return image_data
        
        image_data_a = tensor_to_base64(image_tensor_a)
        image_data_b = tensor_to_base64(image_tensor_b)
        
        # Calculate P(A beats B) for display
        prob_a_beats_b = session.model.comparison_probability(
            candidates[local_idx_a], 
            candidates[local_idx_b]
        ).item()

        
        return (global_idx_a, global_idx_b), image_data_a, image_data_b, prompt_a, prompt_b, prob_a_beats_b

    return None, None, None, None, None, None

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/api/start')
async def start_session():
    '''Start a new session'''
    global current_session
    current_session = SessionData()
    return {'status': 'started'}

@app.get('/api/next-comparison')
async def get_next_comparison_endpoint():
    '''Get the next pair of images to compare'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session. Please start a session first.'}
    
    pair_indices, image_data_a, image_data_b, prompt_a, prompt_b, prob_a_beats_b = get_next_comparison_pair(current_session)
    
    if pair_indices is None:
        return {'error': 'Not enough images available for comparison'}
    
    current_session.current_pair_indices = pair_indices
    current_session.current_image_a_data = image_data_a
    current_session.current_image_b_data = image_data_b
    current_session.current_prompt_a = prompt_a
    current_session.current_prompt_b = prompt_b
    
    return {
        'pair_id': f"{pair_indices[0]}_{pair_indices[1]}",
        'image_a': image_data_a,
        'image_b': image_data_b,
        'prompt_a': prompt_a,
        'prompt_b': prompt_b,
        'prob_a_beats_b': prob_a_beats_b
    }

@app.post('/api/comparison-feedback')
async def submit_comparison_feedback(feedback: ComparisonRequest):
    '''Submit user preference for a pair comparison'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session'}
    
    if current_session.current_pair_indices is None:
        return {'error': 'No active comparison'}
    
    # Parse pair_id to get indices
    expected_pair_id = f"{current_session.current_pair_indices[0]}_{current_session.current_pair_indices[1]}"
    if feedback.pair_id != expected_pair_id:
        return {'error': 'Pair ID mismatch'}
    
    if feedback.preferred not in ['a', 'b']:
        return {'error': 'Invalid preference. Must be "a" or "b"'}
    
    # Get the latent embeddings for both images
    dataloader = DataLoader(current_session.dataset, batch_size=len(current_session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        
        idx_a, idx_b = current_session.current_pair_indices
        embedding_a = latents[idx_a]
        embedding_b = latents[idx_b]
        
        # Update Bradley-Terry model
        a_wins = (feedback.preferred == 'a')
        loss = current_session.model.update(embedding_a, embedding_b, a_wins)
        current_session.comparisons_made += 1
        
        return {
            'status': 'feedback_received',
            'comparisons_made': current_session.comparisons_made,
            'training_loss': loss
        }
    
    return {'error': 'Failed to process feedback'}

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
        'comparisons_made': current_session.comparisons_made
    }

@app.get('/api/recommendations')
async def get_recommendations(k: int = 10):
    '''Get top k recommendations based on learned preferences'''
    global current_session
    
    if current_session is None:
        return {'error': 'No active session'}
    
    if current_session.comparisons_made == 0:
        return {'error': 'No comparisons made yet. Cannot generate recommendations.'}
    
    # Get all embeddings
    dataloader = DataLoader(current_session.dataset, batch_size=len(current_session.dataset), shuffle=False, num_workers=0)
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        
        # Get top recommendations
        top_indices, top_scores = current_session.model.get_top_recommendations(latents, k=k)
        
        recommendations = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            idx = int(idx)
            image_tensor = current_session.dataset.get_image(idx)
            prompt = current_session.dataset.get_prompt(idx)
            
            # Convert to base64
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            save_image(image_tensor, temp_path)
            with open(temp_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            os.unlink(temp_path)
            
            recommendations.append({
                'rank': i + 1,
                'image_idx': idx,
                'image_data': image_data,
                'prompt': prompt,
                'quality_score': float(score)
            })
        
        return {'recommendations': recommendations}
    
    return {'error': 'Failed to generate recommendations'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
