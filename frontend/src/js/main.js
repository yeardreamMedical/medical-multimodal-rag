// Main application logic
class MedicubeApp {
    constructor() {
        this.currentQuestion = null;
        this.selectedOption = null;
        this.isAnswerShown = false;
        
        this.initializeElements();
        this.bindEvents();
        this.showMainPage();
    }

    initializeElements() {
        // Pages
        this.mainPage = document.getElementById('main-page');
        this.quizPage = document.getElementById('quiz-page');
        
        // Main page elements
        this.topicInput = document.getElementById('medical-topic');
        this.generateBtn = document.getElementById('generate-btn');
        this.exampleTags = document.querySelectorAll('.example-tag');
        
        // Quiz page elements
        this.backBtn = document.getElementById('back-btn');
        this.quizTopic = document.getElementById('quiz-topic');
        this.loadingState = document.getElementById('loading-state');
        this.questionDisplay = document.getElementById('question-display');
        this.answerDisplay = document.getElementById('answer-display');
        this.errorDisplay = document.getElementById('error-display');
        
        // Question elements
        this.relatedImage = document.getElementById('related-image');
        this.questionImage = document.getElementById('question-image');
        this.imageType = document.getElementById('image-type');
        this.difficultyBadge = document.getElementById('difficulty-badge');
        this.topicBadge = document.getElementById('topic-badge');
        this.questionText = document.getElementById('question-text');
        this.optionsContainer = document.getElementById('options-container');
        this.showAnswerBtn = document.getElementById('show-answer-btn');
        
        // Similar images elements
        this.similarImagesBtn = document.getElementById('similar-images-btn');
        this.similarImagesModal = document.getElementById('similar-images-modal');
        this.closeModalBtn = document.getElementById('close-modal-btn');
        this.similarImagesGrid = document.getElementById('similar-images-grid');
        
        // Answer elements
        this.correctAnswerText = document.getElementById('correct-answer-text');
        this.explanationText = document.getElementById('explanation-text');
        this.analysisTopic = document.getElementById('analysis-topic');
        this.analysisDifficulty = document.getElementById('analysis-difficulty');
        this.analysisRelevance = document.getElementById('analysis-relevance');
        this.newQuestionBtn = document.getElementById('new-question-btn');
        
        // Error elements
        this.errorMessage = document.getElementById('error-message');
        this.retryBtn = document.getElementById('retry-btn');
    }

    bindEvents() {
        // Main page events
        this.generateBtn.addEventListener('click', () => this.handleGenerateQuestion());
        this.topicInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleGenerateQuestion();
            }
        });
        
        // Example tag events
        this.exampleTags.forEach(tag => {
            tag.addEventListener('click', () => {
                const topic = tag.dataset.topic;
                this.topicInput.value = topic;
                this.handleGenerateQuestion();
            });
        });
        
        // Quiz page events
        this.backBtn.addEventListener('click', () => this.showMainPage());
        this.showAnswerBtn.addEventListener('click', () => this.showAnswer());
        this.newQuestionBtn.addEventListener('click', () => this.showMainPage());
        this.retryBtn.addEventListener('click', () => this.showMainPage());
        
        // Similar images events
        this.similarImagesBtn.addEventListener('click', () => this.showSimilarImages());
        this.closeModalBtn.addEventListener('click', () => this.hideSimilarImages());
        
        // Modal overlay click to close
        this.similarImagesModal.addEventListener('click', (e) => {
            if (e.target === this.similarImagesModal || e.target.classList.contains('modal-overlay')) {
                this.hideSimilarImages();
            }
        });
        
        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.similarImagesModal.style.display === 'block') {
                this.hideSimilarImages();
            }
        });
    }

    showMainPage() {
        this.hideAllPages();
        this.mainPage.classList.add('active');
        this.resetQuizState();
        this.topicInput.focus();
    }

    showQuizPage() {
        this.hideAllPages();
        this.quizPage.classList.add('active');
    }

    hideAllPages() {
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
    }

    resetQuizState() {
        this.currentQuestion = null;
        this.selectedOption = null;
        this.isAnswerShown = false;
        
        // Hide all quiz sections
        this.loadingState.style.display = 'none';
        this.questionDisplay.style.display = 'none';
        this.answerDisplay.style.display = 'none';
        this.errorDisplay.style.display = 'none';
        this.similarImagesModal.style.display = 'none';
        
        // Reset loading steps
        document.querySelectorAll('.step').forEach((step, index) => {
            step.classList.toggle('active', index === 0);
        });
        
        // Enable generate button
        this.generateBtn.disabled = false;
        this.generateBtn.innerHTML = '<span class="btn-text">문제 생성</span><span class="btn-icon">✨</span>';
    }

    async handleGenerateQuestion() {
        const topic = this.topicInput.value.trim();
        
        if (!topic) {
            alert('의료 주제를 입력해주세요.');
            return;
        }

        // Disable generate button and show loading
        this.generateBtn.disabled = true;
        this.generateBtn.innerHTML = '<span class="btn-text">생성 중...</span>';
        
        this.showQuizPage();
        this.showLoading();
        this.quizTopic.textContent = `"${topic}" 문제 풀이`;

        try {
            // Simulate loading steps
            await this.simulateLoadingSteps();
            
            // Generate question (mock API call for now)
            const questionData = await this.generateQuestionAPI(topic);
            
            if (questionData.error) {
                this.showError(questionData.error);
                return;
            }
            
            this.currentQuestion = questionData;
            this.displayQuestion(questionData);
            
        } catch (error) {
            console.error('Question generation failed:', error);
            this.showError('문제 생성 중 오류가 발생했습니다. 다시 시도해주세요.');
        }
    }

    showLoading() {
        this.loadingState.style.display = 'block';
        this.questionDisplay.style.display = 'none';
        this.answerDisplay.style.display = 'none';
        this.errorDisplay.style.display = 'none';
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

    async generateQuestionAPI(topic) {
        // Mock API call - replace with actual backend call
        return new Promise((resolve) => {
            setTimeout(() => {
                // Mock question data
                const mockQuestion = {
                    question: `65세 남성이 3일 전부터 발열, 기침, 가래를 주소로 내원하였다. 체온 38.5°C, 맥박 100회/분, 호흡수 24회/분, 혈압 130/80 mmHg이다. 흉부 청진에서 우하엽에 수포음이 들리며, 흉부 X선 검사에서 우하엽에 경화 소견이 관찰된다. 가장 적절한 진단은?`,
                    options: [
                        "바이러스성 상기도 감염",
                        "세균성 폐렴",
                        "폐결핵",
                        "기관지 천식",
                        "만성 폐쇄성 폐질환"
                    ],
                    answer: 1,
                    explanation: "환자는 발열, 기침, 가래의 전형적인 폐렴 증상을 보이고 있으며, 흉부 청진에서 수포음, 흉부 X선에서 경화 소견이 관찰되어 세균성 폐렴을 시사합니다. 급성 발병과 국소적인 경화 소견은 세균성 폐렴의 특징적인 소견입니다.",
                    topic_analysis: {
                        estimated_topic: "세균성 폐렴",
                        difficulty_level: "중급",
                        clinical_relevance: "high"
                    },
                    image_selection: {
                        selected_image_type: "Pneumonia",
                        korean_name: "폐렴",
                        reason: "세균성 폐렴의 전형적인 흉부 X선 소견을 보여주는 이미지가 문제 이해에 도움이 됩니다."
                    },
                    selected_images: [
                        {
                            image_path: "https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=800",
                            score: 0.95,
                            labels: ["pneumonia", "chest x-ray"]
                        }
                    ]
                };
                
                resolve(mockQuestion);
            }, 3000);
        });
    }

    displayQuestion(questionData) {
        this.loadingState.style.display = 'none';
        this.questionDisplay.style.display = 'block';
        
        // Display image if available
        if (questionData.selected_images && questionData.selected_images.length > 0) {
            const imageData = questionData.selected_images[0];
            this.questionImage.src = imageData.image_path;
            this.questionImage.alt = `${questionData.image_selection.korean_name} 관련 의료 이미지`;
            this.imageType.textContent = `참고 이미지: ${questionData.image_selection.korean_name}`;
            this.relatedImage.style.display = 'block';
        } else {
            this.relatedImage.style.display = 'none';
        }
        
        // Display question metadata
        this.difficultyBadge.textContent = questionData.topic_analysis.difficulty_level;
        this.topicBadge.textContent = questionData.topic_analysis.estimated_topic;
        
        // Display question text
        this.questionText.textContent = questionData.question;
        
        // Display options
        this.displayOptions(questionData.options);
        
        // Reset answer button
        this.showAnswerBtn.disabled = true;
        this.isAnswerShown = false;
    }

    displayOptions(options) {
        this.optionsContainer.innerHTML = '';
        
        options.forEach((option, index) => {
            const optionElement = document.createElement('div');
            optionElement.className = 'option';
            optionElement.dataset.index = index;
            
            optionElement.innerHTML = `
                <div class="option-radio"></div>
                <div class="option-text">${option}</div>
            `;
            
            optionElement.addEventListener('click', () => this.selectOption(index));
            
            this.optionsContainer.appendChild(optionElement);
        });
    }

    selectOption(index) {
        if (this.isAnswerShown) return;
        
        // Remove previous selection
        document.querySelectorAll('.option').forEach(option => {
            option.classList.remove('selected');
        });
        
        // Add selection to clicked option
        const selectedOption = document.querySelector(`[data-index="${index}"]`);
        selectedOption.classList.add('selected');
        
        this.selectedOption = index;
        this.showAnswerBtn.disabled = false;
    }

    showAnswer() {
        if (!this.currentQuestion || this.selectedOption === null) return;
        
        this.isAnswerShown = true;
        
        // Mark correct and incorrect options
        document.querySelectorAll('.option').forEach((option, index) => {
            const optionIndex = parseInt(option.dataset.index);
            
            if (optionIndex === this.currentQuestion.answer) {
                option.classList.add('correct');
            } else if (optionIndex === this.selectedOption && optionIndex !== this.currentQuestion.answer) {
                option.classList.add('incorrect');
            }
        });
        
        // Display answer section
        this.displayAnswerSection();
        
        // Scroll to answer section
        this.answerDisplay.scrollIntoView({ behavior: 'smooth' });
    }

    displayAnswerSection() {
        const correctOption = this.currentQuestion.options[this.currentQuestion.answer];
        
        this.correctAnswerText.textContent = `${this.currentQuestion.answer + 1}. ${correctOption}`;
        this.explanationText.textContent = this.currentQuestion.explanation;
        
        // Display AI analysis
        this.analysisTopic.textContent = this.currentQuestion.topic_analysis.estimated_topic;
        this.analysisDifficulty.textContent = this.currentQuestion.topic_analysis.difficulty_level;
        this.analysisRelevance.textContent = this.currentQuestion.topic_analysis.clinical_relevance.toUpperCase();
        
        this.answerDisplay.style.display = 'block';
    }

    showSimilarImages() {
        if (!this.currentQuestion || !this.currentQuestion.selected_images) {
            return;
        }

        // Generate mock similar images based on the current topic
        const currentTopic = this.currentQuestion.topic_analysis.estimated_topic;
        const similarImages = this.generateMockSimilarImages(currentTopic);
        
        // Clear previous content
        this.similarImagesGrid.innerHTML = '';
        
        // Add similar images to grid
        similarImages.forEach((imageData, index) => {
            const imageItem = document.createElement('div');
            imageItem.className = 'similar-image-item';
            
            imageItem.innerHTML = `
                <img src="${imageData.url}" alt="${imageData.description}" loading="lazy">
                <div class="similar-image-info">
                    <h4>${imageData.title}</h4>
                    <p>${imageData.description}</p>
                    <span class="similarity-score">유사도 ${imageData.similarity}%</span>
                </div>
            `;
            
            // Add click event to view larger image
            imageItem.addEventListener('click', () => {
                this.viewLargerImage(imageData);
            });
            
            this.similarImagesGrid.appendChild(imageItem);
        });
        
        // Show modal
        this.similarImagesModal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }

    hideSimilarImages() {
        this.similarImagesModal.style.display = 'none';
        document.body.style.overflow = ''; // Restore scrolling
    }

    generateMockSimilarImages(topic) {
        // Mock similar images based on topic
        const imageTemplates = {
            '세균성 폐렴': [
                {
                    title: '폐렴 - 우하엽 경화',
                    description: '우하엽에 국소적인 경화 소견이 관찰되는 전형적인 세균성 폐렴',
                    url: 'https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 94
                },
                {
                    title: '폐렴 - 좌하엽 침윤',
                    description: '좌하엽에 침윤성 음영을 보이는 폐렴 소견',
                    url: 'https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 89
                },
                {
                    title: '다발성 폐렴',
                    description: '양측 폐야에 다발성 침윤을 보이는 폐렴',
                    url: 'https://images.pexels.com/photos/4386370/pexels-photo-4386370.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 87
                },
                {
                    title: '폐렴 회복기',
                    description: '치료 후 호전되고 있는 폐렴의 흉부 X선 소견',
                    url: 'https://images.pexels.com/photos/4386444/pexels-photo-4386444.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 82
                }
            ],
            '기흉': [
                {
                    title: '자연기흉 - 좌측',
                    description: '좌측 폐의 부분적 허탈을 보이는 자연기흉',
                    url: 'https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 92
                },
                {
                    title: '긴장성 기흉',
                    description: '종격동 편위를 동반한 긴장성 기흉',
                    url: 'https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 88
                },
                {
                    title: '기흉 - 우측',
                    description: '우측 폐의 완전 허탈을 보이는 기흉',
                    url: 'https://images.pexels.com/photos/4386370/pexels-photo-4386370.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 85
                },
                {
                    title: '기흉 치료 후',
                    description: '흉관 삽입 후 폐 재팽창된 상태',
                    url: 'https://images.pexels.com/photos/4386444/pexels-photo-4386444.jpeg?auto=compress&cs=tinysrgb&w=400',
                    similarity: 79
                }
            ]
        };

        // Return images for the current topic, or default pneumonia images
        return imageTemplates[topic] || imageTemplates['세균성 폐렴'];
    }

    viewLargerImage(imageData) {
        // Create a temporary modal for larger image view
        const largeImageModal = document.createElement('div');
        largeImageModal.className = 'modal';
        largeImageModal.style.zIndex = '1001';
        
        largeImageModal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="modal-content" style="max-width: 80vw; max-height: 80vh;">
                <div class="modal-header">
                    <h3>${imageData.title}</h3>
                    <button class="close-modal-btn">×</button>
                </div>
                <div class="modal-body" style="text-align: center;">
                    <img src="${imageData.url}" alt="${imageData.description}" 
                         style="max-width: 100%; max-height: 60vh; object-fit: contain;">
                    <p style="margin-top: 1rem; color: var(--secondary-600);">${imageData.description}</p>
                    <span class="similarity-score" style="margin-top: 0.5rem;">유사도 ${imageData.similarity}%</span>
                </div>
            </div>
        `;
        
        document.body.appendChild(largeImageModal);
        largeImageModal.style.display = 'flex';
        
        // Close events
        const closeBtn = largeImageModal.querySelector('.close-modal-btn');
        const overlay = largeImageModal.querySelector('.modal-overlay');
        
        const closeLargeImage = () => {
            document.body.removeChild(largeImageModal);
        };
        
        closeBtn.addEventListener('click', closeLargeImage);
        overlay.addEventListener('click', closeLargeImage);
        
        // ESC key
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                closeLargeImage();
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
    }

    showError(errorMessage) {
        this.loadingState.style.display = 'none';
        this.questionDisplay.style.display = 'none';
        this.answerDisplay.style.display = 'none';
        this.errorDisplay.style.display = 'block';
        
        this.errorMessage.textContent = errorMessage;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MedicubeApp();
});