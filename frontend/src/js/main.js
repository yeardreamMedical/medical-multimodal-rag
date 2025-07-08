// frontend/src/js/main.js

import { apiService } from './apiService.js';
import { UIManager } from './uiManager.js';
import { QuizLogic } from './quizLogic.js';
import { ModalManager } from './modalManager.js';

// Main application logic
class MedicubeApp {
    constructor() {
        this.uiManager = new UIManager();
        this.quizLogic = new QuizLogic(this.uiManager, apiService);
        this.modalManager = new ModalManager(this.uiManager, apiService);
        
        this.bindEvents();
        this.uiManager.showMainPage();
    }

    bindEvents() {
        // Main page events
        this.uiManager.elements.generateBtn.addEventListener('click', () => this.handleGenerateQuestion());
        this.uiManager.elements.topicInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleGenerateQuestion();
            }
        });
        
        // Example tag events
        this.uiManager.elements.exampleTags.forEach(tag => {
            tag.addEventListener('click', () => {
                const topic = tag.dataset.topic;
                this.uiManager.elements.topicInput.value = topic;
                this.handleGenerateQuestion();
            });
        });
        
        // Quiz page events
        this.uiManager.elements.backBtn.addEventListener('click', () => this.uiManager.showMainPage());
        this.uiManager.elements.showAnswerBtn.addEventListener('click', () => this.quizLogic.showAnswer());
        this.uiManager.elements.newQuestionBtn.addEventListener('click', () => this.uiManager.showMainPage());
        this.uiManager.elements.retryBtn.addEventListener('click', () => this.uiManager.showMainPage());
        
        // Similar images events
        this.uiManager.elements.similarImagesBtn.addEventListener('click', () => this.modalManager.showSimilarImages(this.quizLogic.currentQuestion.topic_analysis.estimated_topic));
        this.uiManager.elements.closeModalBtn.addEventListener('click', () => this.modalManager.hideSimilarImages());
        
        // Modal overlay click to close
        this.uiManager.elements.similarImagesModal.addEventListener('click', (e) => {
            if (e.target === this.uiManager.elements.similarImagesModal || e.target.classList.contains('modal-overlay')) {
                this.modalManager.hideSimilarImages();
            }
        });
        
        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.uiManager.elements.similarImagesModal.style.display === 'block') {
                this.modalManager.hideSimilarImages();
            }
        });
    }

    async handleGenerateQuestion() {
        const topic = this.uiManager.elements.topicInput.value.trim();
        await this.quizLogic.handleGenerateQuestion(topic);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MedicubeApp();
});
