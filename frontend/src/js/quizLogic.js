// frontend/src/js/quizLogic.js

export class QuizLogic {
    constructor(uiManager, apiService) {
        this.uiManager = uiManager;
        this.apiService = apiService;
        this.currentQuestion = null;
        this.selectedOption = null;
        this.isAnswerShown = false;
    }

    resetQuizState() {
        this.currentQuestion = null;
        this.selectedOption = null;
        this.isAnswerShown = false;
        this.uiManager.resetQuizState();
    }

    async handleGenerateQuestion(topic) {
        if (!topic) {
            alert('의료 주제를 입력해주세요.');
            return;
        }

        this.uiManager.elements.generateBtn.disabled = true;
        this.uiManager.elements.generateBtn.innerHTML = '<span class="btn-text">생성 중...</span>';
        
        this.uiManager.showQuizPage();
        this.uiManager.showLoading();
        this.uiManager.elements.quizTopic.textContent = `"${topic}" 문제 풀이`;

        try {
            await this.uiManager.simulateLoadingSteps();
            
            const questionData = await this.apiService.generateQuestionAPI(topic);
            
            if (questionData.error) {
                this.uiManager.showError(questionData.error);
                return;
            }
            
            this.currentQuestion = questionData;
            this.uiManager.displayQuestion(questionData);
            this.uiManager.displayOptions(questionData.options, this.selectOption.bind(this));
            
        } catch (error) {
            console.error('Question generation failed:', error);
            this.uiManager.showError('문제 생성 중 오류가 발생했습니다. 다시 시도해주세요.');
        }
    }

    selectOption(index) {
        if (this.isAnswerShown) return;
        
        this.uiManager.selectOptionUI(index);
        
        this.selectedOption = index;
        this.uiManager.elements.showAnswerBtn.disabled = false;
    }

    showAnswer() {
        if (!this.currentQuestion || this.selectedOption === null) return;
        
        this.isAnswerShown = true;
        
        this.uiManager.markAnswerUI(this.currentQuestion.answer, this.selectedOption);
        this.uiManager.displayAnswerSection(this.currentQuestion);
        
        this.uiManager.elements.answerDisplay.scrollIntoView({ behavior: 'smooth' });
    }
}
