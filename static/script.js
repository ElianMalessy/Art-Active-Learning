class ArtComparisonApp {
    constructor() {
        this.currentPairId = null;
        this.sessionStarted = false;
        this.comparisonsCount = 0;

        this.initializeElements();
        this.bindEvents();
        this.autoStartSession();
    }

    initializeElements() {
        this.loading = document.getElementById('loading');
        this.comparisonPair = document.getElementById('comparison-pair');
        this.imageA = document.getElementById('image-a');
        this.imageB = document.getElementById('image-b');
        this.promptA = document.getElementById('prompt-a');
        this.promptB = document.getElementById('prompt-b');
        this.preferBtns = document.querySelectorAll('.prefer-btn');
        
        // Probability display elements
        this.probValue = document.getElementById('prob-value');
        this.probDescription = document.getElementById('prob-description');
        
        // Stats elements
        this.comparisonsCountEl = document.getElementById('comparisons-count');
        this.remainingCountEl = document.getElementById('remaining-count');
        
        // Recommendations elements
        this.recommendationsSection = document.getElementById('recommendations-section');
        this.recommendationsGrid = document.getElementById('recommendations-grid');
        this.continueBtn = document.getElementById('continue-btn');
    }

    bindEvents() {
        // Preference button events
        this.preferBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent card click event
                const choice = e.currentTarget.dataset.choice;
                this.handlePreference(choice);
            });
        });

        // Make entire cards clickable
        document.getElementById('card-a').addEventListener('click', () => {
            this.handlePreference('a');
        });
        
        document.getElementById('card-b').addEventListener('click', () => {
            this.handlePreference('b');
        });

        // Continue button event
        this.continueBtn.addEventListener('click', () => {
            this.hideRecommendations();
            this.getNextComparison();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'a' || e.key === 'A') {
                this.handlePreference('a');
            } else if (e.key === 'b' || e.key === 'B') {
                this.handlePreference('b');
            } else if (e.key === 'r' || e.key === 'R') {
                this.showRecommendations();
            }
        });
    }

    async autoStartSession() {
        console.log('🚀 Starting new art comparison session...');
        try {
            const response = await fetch('/api/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'started') {
                this.sessionStarted = true;
                console.log('✅ Session started successfully');
                await this.getNextComparison();
                await this.updateStats();
            } else {
                throw new Error('Failed to start session');
            }
        } catch (error) {
            console.error('❌ Error starting session:', error);
            this.showError('Failed to start session. Please refresh the page.');
        }
    }

    async getNextComparison() {
        if (!this.sessionStarted) {
            console.log('⚠️ No active session');
            return;
        }

        this.showLoading();
        
        try {
            console.log('🔍 Fetching next comparison...');
            const response = await fetch('/api/next-comparison');
            const data = await response.json();
            
            if (data.error) {
                if (data.error.includes('Not enough images')) {
                    this.showRecommendations();
                } else {
                    throw new Error(data.error);
                }
            } else {
                this.displayComparison(data);
            }
        } catch (error) {
            console.error('❌ Error fetching comparison:', error);
            this.showError('Failed to load comparison. Please try again.');
        }
    }

    displayComparison(data) {
        this.currentPairId = data.pair_id;
        
        // Set images
        this.imageA.src = `data:image/png;base64,${data.image_a}`;
        this.imageB.src = `data:image/png;base64,${data.image_b}`;
        
        // Set prompts
        this.promptA.textContent = data.prompt_a;
        this.promptB.textContent = data.prompt_b;
        
        // Set probability display
        const probPercent = (data.prob_a_beats_b * 100).toFixed(1);
        this.probValue.textContent = `${probPercent}%`;
        this.probDescription.textContent = `left wins vs right`;
        
        this.hideLoading();
        
        console.log('🖼️ Displaying comparison:', {
            pairId: data.pair_id,
            promptA: data.prompt_a.substring(0, 50) + '...',
            promptB: data.prompt_b.substring(0, 50) + '...',
            probABeatsB: data.prob_a_beats_b
        });
    }

    async handlePreference(choice) {
        if (!this.currentPairId) {
            console.log('⚠️ No active comparison');
            return;
        }

        // Visual feedback - highlight selected card
        const selectedCard = document.getElementById(`card-${choice}`);
        const otherCard = document.getElementById(choice === 'a' ? 'card-b' : 'card-a');
        
        selectedCard.classList.add('selected');
        otherCard.style.opacity = '0.5';

        try {
            console.log(`💖 User preferred ${choice.toUpperCase()}`);
            
            const response = await fetch('/api/comparison-feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pair_id: this.currentPairId,
                    preferred: choice
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            } else {
                console.log('✅ Feedback submitted successfully', {
                    comparisons: data.comparisons_made,
                    loss: data.training_loss
                });
                
                this.comparisonsCount = data.comparisons_made;
                await this.updateStats();
                
                // Reset visual states
                setTimeout(() => {
                    selectedCard.classList.remove('selected');
                    otherCard.style.opacity = '1';
                }, 500);
                
                // Show recommendations after every 5 comparisons
                if (this.comparisonsCount % 5 === 0 && this.comparisonsCount > 0) {
                    this.showRecommendations();
                } else {
                    // Get next comparison
                    setTimeout(() => {
                        this.getNextComparison();
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('❌ Error submitting feedback:', error);
            this.showError('Failed to submit preference. Please try again.');
            
            // Reset visual states on error
            selectedCard.classList.remove('selected');
            otherCard.style.opacity = '1';
        }
    }

    async updateStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (!data.error) {
                this.comparisonsCountEl.textContent = data.comparisons_made || 0;
                this.remainingCountEl.textContent = data.remaining_images || 0;
            }
        } catch (error) {
            console.error('❌ Error updating stats:', error);
        }
    }

    async showRecommendations() {
        try {
            console.log('🎯 Fetching recommendations...');
            const response = await fetch('/api/recommendations?k=10');
            const data = await response.json();
            
            if (data.error) {
                console.log('⚠️ Cannot show recommendations:', data.error);
                this.getNextComparison();
                return;
            }
            
            this.displayRecommendations(data.recommendations);
        } catch (error) {
            console.error('❌ Error fetching recommendations:', error);
        }
    }

    displayRecommendations(recommendations) {
        this.recommendationsGrid.innerHTML = '';
        
        recommendations.forEach((rec, index) => {
            const card = document.createElement('div');
            card.className = 'recommendation-card';
            card.innerHTML = `
                <div class="rank">${rec.rank}</div>
                <div class="quality-score">${rec.quality_score.toFixed(2)}</div>
                <div class="image-container">
                    <img src="data:image/png;base64,${rec.image_data}" alt="Recommended artwork">
                </div>
                <div class="card-info">
                    <h3>${rec.prompt}</h3>
                </div>
            `;
            this.recommendationsGrid.appendChild(card);
        });
        
        this.comparisonPair.style.display = 'none';
        this.loading.style.display = 'none';
        this.recommendationsSection.style.display = 'block';
        
        console.log('🎯 Showing recommendations:', recommendations.length);
    }

    hideRecommendations() {
        this.recommendationsSection.style.display = 'none';
    }

    showLoading() {
        this.loading.style.display = 'block';
        this.comparisonPair.style.display = 'none';
    }

    hideLoading() {
        this.loading.style.display = 'none';
        this.comparisonPair.style.display = 'flex';
    }

    showError(message) {
        console.error('💥 Error:', message);
        this.loading.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <p>${message}</p>
        `;
        this.loading.style.display = 'block';
        this.comparisonPair.style.display = 'none';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎨 Art Comparison App initializing...');
    new ArtComparisonApp();
});

// Global error handling
window.addEventListener('error', (e) => {
    console.error('💥 Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('💥 Unhandled promise rejection:', e.reason);
});