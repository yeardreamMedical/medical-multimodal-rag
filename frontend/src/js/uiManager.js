// frontend/src/js/uiManager.js

export class UIManager {
    constructor() {
        this.elements = {};
        this.initializeElements();
    }

    initializeElements() {
        // Pages
        this.elements.mainPage = document.getElementById('main-page');
        this.elements.quizPage = document.getElementById('quiz-page');
        
        // Main page elements
        this.elements.topicInput = document.getElementById('medical-topic');
        this.elements.generateBtn = document.getElementById('generate-btn');
        this.elements.exampleTags = document.querySelectorAll('.example-tag');
        
        // Quiz page elements
        this.elements.backBtn = document.getElementById('back-btn');
        this.elements.quizTopic = document.getElementById('quiz-topic');
        this.elements.loadingState = document.getElementById('loading-state');
        this.elements.questionDisplay = document.getElementById('question-display');
        this.elements.answerDisplay = document.getElementById('answer-display');
        this.elements.errorDisplay = document.getElementById('error-display');
        
        // Question elements
        this.elements.relatedImage = document.getElementById('related-image');
        this.elements.questionImage = document.getElementById('question-image');
        this.elements.imageType = document.getElementById('image-type');
        this.elements.difficultyBadge = document.getElementById('difficulty-badge');
        this.elements.topicBadge = document.getElementById('topic-badge');
        this.elements.questionText = document.getElementById('question-text');
        this.elements.optionsContainer = document.getElementById('options-container');
        this.elements.showAnswerBtn = document.getElementById('show-answer-btn');
        
        // Similar images elements
        this.elements.similarImagesBtn = document.getElementById('similar-images-btn');
        this.elements.similarImagesModal = document.getElementById('similar-images-modal');
        this.elements.closeModalBtn = document.getElementById('close-modal-btn');
        this.elements.similarImagesGrid = document.getElementById('similar-images-grid');
        
        // Answer elements
        this.elements.correctAnswerText = document.getElementById('correct-answer-text');
        this.elements.explanationText = document.getElementById('explanation-text');
        this.elements.analysisTopic = document.getElementById('analysis-topic');
        this.elements.analysisDifficulty = document.getElementById('analysis-difficulty');
        this.elements.analysisRelevance = document.getElementById('analysis-relevance');
        this.elements.newQuestionBtn = document.getElementById('new-question-btn');
        
        // Error elements
        this.elements.errorMessage = document.getElementById('error-message');
        this.elements.retryBtn = document.getElementById('retry-btn');
    }

    showMainPage() {
        this.hideAllPages();
        this.elements.mainPage.classList.add('active');
        this.resetQuizState();
        this.elements.topicInput.focus();
    }

    showQuizPage() {
        this.hideAllPages();
        this.elements.quizPage.classList.add('active');
    }

    hideAllPages() {
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
    }

    resetQuizState() {
        // Hide all quiz sections
        this.elements.loadingState.style.display = 'none';
        this.elements.questionDisplay.style.display = 'none';
        this.elements.answerDisplay.style.display = 'none';
        this.elements.errorDisplay.style.display = 'none';
        this.elements.similarImagesModal.style.display = 'none';
        
        // Reset loading steps
        document.querySelectorAll('.step').forEach((step, index) => {
            step.classList.toggle('active', index === 0);
        });
        
        // Enable generate button
        this.elements.generateBtn.disabled = false;
        this.elements.generateBtn.innerHTML = '<span class="btn-text">문제 생성</span><span class="btn-icon">✨</span>';
    }

    showLoading() {
        this.elements.loadingState.style.display = 'block';
        this.elements.questionDisplay.style.display = 'none';
        this.elements.answerDisplay.style.display = 'none';
        this.elements.errorDisplay.style.display = 'none';
    }

    async simulateLoadingSteps() {
        const steps = document.querySelectorAll('.step');
        
        for (let i = 0; i < steps.length; i++) {
            // Activate current step
            steps.forEach((step, index) => {
                step.classList.toggle('active', index <= i);
            });
            
            // Wait for step duration
            await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
        }
    }

    displayQuestion(questionData) {
        this.elements.loadingState.style.display = 'none';
        this.elements.questionDisplay.style.display = 'block';
        
        // Display image if available
        if (questionData.selected_images && questionData.selected_images.length > 0) {
            const imageData = questionData.selected_images[0];
            this.elements.questionImage.src = imageData.image_path;
            this.elements.questionImage.alt = `${questionData.image_selection.korean_name} 관련 의료 이미지`;
            this.elements.imageType.textContent = `참고 이미지: ${questionData.image_selection.korean_name}`;
            this.elements.relatedImage.style.display = 'block';
        } else {
            this.elements.relatedImage.style.display = 'none';
        }
        
        // Display question metadata
        this.elements.difficultyBadge.textContent = questionData.topic_analysis.difficulty_level;
        this.elements.topicBadge.textContent = questionData.topic_analysis.estimated_topic;
        
        // Display question text
        this.elements.questionText.textContent = questionData.question;
        
        // Reset answer button
        this.elements.showAnswerBtn.disabled = true;
    }

    displayOptions(options, selectOptionCallback) {
        this.elements.optionsContainer.innerHTML = '';
        
        options.forEach((option, index) => {
            const optionElement = document.createElement('div');
            optionElement.className = 'option';
            optionElement.dataset.index = index;
            
            optionElement.innerHTML = `
                <div class="option-radio"></div>
                <div class="option-text">${option}</div>
            `;
            
            optionElement.addEventListener('click', () => selectOptionCallback(index));
            
            this.elements.optionsContainer.appendChild(optionElement);
        });
    }

    selectOptionUI(index) {
        // Remove previous selection
        document.querySelectorAll('.option').forEach(option => {
            option.classList.remove('selected');
        });
        
        // Add selection to clicked option
        const selectedOption = document.querySelector(`[data-index="${index}"]`);
        selectedOption.classList.add('selected');
    }

    markAnswerUI(correctAnswerIndex, selectedOptionIndex) {
        document.querySelectorAll('.option').forEach((option, index) => {
            const optionIndex = parseInt(option.dataset.index);
            
            if (optionIndex === correctAnswerIndex) {
                option.classList.add('correct');
            } else if (optionIndex === selectedOptionIndex && optionIndex !== correctAnswerIndex) {
                option.classList.add('incorrect');
            }
        });
    }

    displayAnswerSection(questionData) {
        const correctOption = questionData.options[questionData.answer];
        
        this.elements.correctAnswerText.textContent = `${questionData.answer + 1}. ${correctOption}`;
        this.elements.explanationText.textContent = questionData.explanation;
        
        // Display AI analysis
        this.elements.analysisTopic.textContent = questionData.topic_analysis.estimated_topic;
        this.elements.analysisDifficulty.textContent = questionData.topic_analysis.difficulty_level;
        this.elements.analysisRelevance.textContent = questionData.topic_analysis.clinical_relevance.toUpperCase();
        
        this.elements.answerDisplay.style.display = 'block';
    }

    showError(errorMessage) {
        this.elements.loadingState.style.display = 'none';
        this.elements.questionDisplay.style.display = 'none';
        this.elements.answerDisplay.style.display = 'none';
        this.elements.errorDisplay.style.display = 'block';
        
        this.elements.errorMessage.textContent = errorMessage;
    }
}
