class TinderApp {
    constructor() {
        this.currentImageId = null;
        this.isDragging = false;
        this.startX = 0;
        this.startY = 0;
        this.currentX = 0;
        this.currentY = 0;
        this.threshold = 100;
        this.sessionStarted = false;

        this.initializeElements();
        this.bindEvents();
        this.autoStartSession();
    }

    initializeElements() {
        this.card = document.getElementById('main-card');
        this.imageContainer = document.getElementById('image-container');
        this.cardImage = document.getElementById('card-image');
        this.cardPrompt = document.getElementById('card-prompt');
        this.loading = document.getElementById('loading');
        this.likeBtn = document.getElementById('like-btn');
        this.dislikeBtn = document.getElementById('dislike-btn');
        this.likeIndicator = document.getElementById('like-indicator');
        this.dislikeIndicator = document.getElementById('dislike-indicator');

        this.likesCount = document.getElementById('likes-count');
        this.dislikesCount = document.getElementById('dislikes-count');
        this.remainingCount = document.getElementById('remaining-count');
    }

    bindEvents() {
        // Button events
        this.likeBtn.addEventListener('click', () => this.handleLike());
        this.dislikeBtn.addEventListener('click', () => this.handleDislike());

        // Add keyboard shortcut for testing
        document.addEventListener('keydown', (e) => {
            if (e.key === 't' || e.key === 'T') {
                console.log('🔧 Manual test triggered with T key');
                this.getNextImage();
            }
        });

        // Touch events for mobile
        this.card.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: false });
        this.card.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
        this.card.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: false });

        // Mouse events for desktop
        this.card.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.card.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.card.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.card.addEventListener('mouseleave', (e) => this.handleMouseUp(e));

        // Prevent context menu on long press
        this.card.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    async autoStartSession() {
        console.log('🚀 autoStartSession: Starting automatically...');
        try {
            console.log('🚀 autoStartSession: Fetching /api/start...');
            const response = await fetch('/api/start', { method: 'POST' });
            console.log('🚀 autoStartSession: Response received:', response.status);
            const data = await response.json();
            console.log('🚀 autoStartSession: Data received:', data);

            if (data.status === 'started') {
                console.log('🚀 autoStartSession: Session started successfully!');
                this.sessionStarted = true;
                console.log('🚀 autoStartSession: About to call getNextImage()...');
                await this.getNextImage();
                console.log('🚀 autoStartSession: getNextImage() completed, updating stats...');
                await this.updateStats();
                console.log('🚀 autoStartSession: All done!');
            } else {
                console.log('🚀 autoStartSession: Unexpected response:', data);
            }
        } catch (error) {
            console.log('🚀 autoStartSession: Error caught:', error);
            this.showMessage('Failed to start session', 'error');
            console.error('Error starting session:', error);
        }
    }

    async getNextImage() {
        console.log('getNextImage: Starting...');
        this.showLoading(true);

        try {
            console.log('getNextImage: Fetching from API...');
            const response = await fetch('/api/next-image');
            const data = await response.json();

            if (data.error) {
                console.log('getNextImage: API returned error:', data.error);
                this.showMessage(data.error, 'error');
                this.showLoading(false);
                return;
            }

            console.log('getNextImage: API success, image_id:', data.image_id);
            this.currentImageId = data.image_id;
            this.cardPrompt.textContent = data.prompt;

            // Create a new image to test loading before setting the actual image
            const testImage = new Image();

            testImage.onload = () => {
                console.log('getNextImage: Test image loaded successfully');
                // Image loaded successfully, now set it to the actual image element
                this.cardImage.src = testImage.src;
                this.showLoading(false);
                this.resetCardPosition();
                console.log('getNextImage: Image displayed, loading hidden');
            };

            testImage.onerror = () => {
                console.log('getNextImage: Test image failed to load');
                this.showMessage('Failed to load image', 'error');
                this.showLoading(false);
                console.error('Error loading image');
            };

            // Start loading the test image
            console.log('getNextImage: Starting image load...');
            testImage.src = `data:image/png;base64,${data.image_data}`;

        } catch (error) {
            console.log('getNextImage: Exception caught:', error);
            this.showMessage('Failed to load image', 'error');
            this.showLoading(false);
            console.error('Error loading image:', error);
        }
    }

    async handleLike() {
        console.log('❤️ handleLike: Called, currentImageId:', this.currentImageId, 'sessionStarted:', this.sessionStarted);
        if (this.currentImageId < 0 || !this.sessionStarted) {
            console.log('❤️ handleLike: Early return - missing imageId or session not started');
            return;
        }

        console.log('❤️ handleLike: Submitting feedback...');
        console.log('❤️ handleLike: Feedback submitted, animating card...');
        this.animateCard('right');
        this.showLoading(true);
        this.cardPrompt.textContent = 'Prompt will appear here'
        await this.submitFeedback(1);
    }

    async handleDislike() {
        console.log('❌ handleDislike: Called, currentImageId:', this.currentImageId, 'sessionStarted:', this.sessionStarted);
        if (!this.currentImageId || !this.sessionStarted) {
            console.log('❌ handleDislike: Early return - missing imageId or session not started');
            return;
        }

        console.log('❌ handleDislike: Submitting feedback...');
        console.log('❌ handleDislike: Feedback submitted, animating card...');
        this.animateCard('left');
        this.showLoading(true);
        this.cardPrompt.textContent = 'Prompt will appear here'
        await this.submitFeedback(0);
    }

    async submitFeedback(rating) {
        console.log('📤 submitFeedback: Submitting rating', rating, 'for image', this.currentImageId);
        try {
            const response = await fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_id: this.currentImageId,
                    rating: rating
                })
            });

            console.log('📤 submitFeedback: Response received:', response.status);
            const data = await response.json();
            console.log('📤 submitFeedback: Data received:', data);

            if (data.error) {
                console.log('📤 submitFeedback: Error in response:', data.error);
                this.showMessage(data.error, 'error');
                return;
            }

            console.log('📤 submitFeedback: Updating stats...');
            await this.updateStats();
            console.log('📤 submitFeedback: Stats updated successfully');

        } catch (error) {
            console.log('📤 submitFeedback: Exception caught:', error);
            this.showMessage('Failed to submit feedback', 'error');
            console.error('Error submitting feedback:', error);
        }
    }

    async updateStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();

            if (data.error) {
                return;
            }

            this.likesCount.textContent = data.likes;
            this.dislikesCount.textContent = data.dislikes;
            this.remainingCount.textContent = data.remaining_images;

        } catch (error) {
            console.error('Error updating stats:', error);
        }
    }

    // Touch event handlers
    handleTouchStart(e) {
        e.preventDefault();
        this.isDragging = true;
        this.startX = e.touches[0].clientX;
        this.startY = e.touches[0].clientY;
        this.card.classList.add('touching');
    }

    handleTouchMove(e) {
        if (!this.isDragging) return;
        e.preventDefault();

        this.currentX = e.touches[0].clientX;
        this.currentY = e.touches[0].clientY;

        const deltaX = this.currentX - this.startX;
        const deltaY = this.currentY - this.startY;

        this.updateCardPosition(deltaX, deltaY);
        this.updateSwipeIndicators(deltaX);
    }

    handleTouchEnd(e) {
        if (!this.isDragging) return;
        e.preventDefault();

        this.isDragging = false;
        this.card.classList.remove('touching');

        const deltaX = this.currentX - this.startX;
        this.handleSwipeEnd(deltaX);
    }

    // Mouse event handlers
    handleMouseDown(e) {
        e.preventDefault();
        this.isDragging = true;
        this.startX = e.clientX;
        this.startY = e.clientY;
        this.card.classList.add('touching');
    }

    handleMouseMove(e) {
        if (!this.isDragging) return;
        e.preventDefault();

        this.currentX = e.clientX;
        this.currentY = e.clientY;

        const deltaX = this.currentX - this.startX;
        const deltaY = this.currentY - this.startY;

        this.updateCardPosition(deltaX, deltaY);
        this.updateSwipeIndicators(deltaX);
    }

    handleMouseUp(e) {
        if (!this.isDragging) return;
        e.preventDefault();

        this.isDragging = false;
        this.card.classList.remove('touching');

        const deltaX = this.currentX - this.startX;
        this.handleSwipeEnd(deltaX);
    }

    updateCardPosition(deltaX, deltaY) {
        const rotation = deltaX * 0.1;
        this.card.style.transform = `translate(${deltaX}px, ${deltaY}px) rotate(${rotation}deg)`;
    }

    updateSwipeIndicators(deltaX) {
        if (deltaX > this.threshold) {
            this.likeIndicator.classList.add('show');
            this.dislikeIndicator.classList.remove('show');
        } else if (deltaX < -this.threshold) {
            this.dislikeIndicator.classList.add('show');
            this.likeIndicator.classList.remove('show');
        } else {
            this.likeIndicator.classList.remove('show');
            this.dislikeIndicator.classList.remove('show');
        }
    }

    handleSwipeEnd(deltaX) {
        if (deltaX > this.threshold) {
            this.handleLike();
        } else if (deltaX < -this.threshold) {
            this.handleDislike();
        } else {
            this.resetCardPosition();
        }
    }

    animateCard(direction) {
        console.log('🎬 animateCard: Starting animation for direction:', direction);
        const animationClass = direction === 'right' ? 'slide-out-right' : 'slide-out-left';
        console.log('🎬 animateCard: Adding class:', animationClass);
        this.card.classList.add(animationClass);

        setTimeout(async () => {
            console.log('🎬 animateCard: Animation timeout completed, removing class and getting next image');
            this.card.classList.remove(animationClass);
            this.resetCardPosition();
            console.log('🎯 animateCard: Getting next image after', direction, 'swipe');
            await this.getNextImage();
        }, 300);
    }

    resetCardPosition() {
        this.card.style.transform = '';
        this.likeIndicator.classList.remove('show');
        this.dislikeIndicator.classList.remove('show');
    }

    showLoading(show) {
        this.loading.style.display = show ? 'flex' : 'none';
        this.cardImage.style.display = show ? 'none' : 'block';
    }

    showMessage(text, type) {
        // Remove existing messages
        const existingMessage = document.querySelector('.message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        document.body.appendChild(message);

        // Show message
        setTimeout(() => message.classList.add('show'), 100);

        // Hide message after 3 seconds
        setTimeout(() => {
            message.classList.remove('show');
            setTimeout(() => message.remove(), 300);
        }, 3000);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 TinderApp initializing...');
    new TinderApp();
    console.log('✅ TinderApp initialized successfully!');
});
