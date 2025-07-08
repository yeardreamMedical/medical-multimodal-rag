// frontend/src/js/apiService.js

export const apiService = {
    async generateQuestionAPI(topic) {
        try {
            const response = await fetch('/api/generate-question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ topic: topic }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '문제 생성에 실패했습니다.');
            }

            return await response.json();
        } catch (error) {
            console.error('Error generating question:', error);
            throw error;
        }
    },

    async getSimilarImagesAPI(topic) {
        try {
            const response = await fetch(`/api/similar-images/${encodeURIComponent(topic)}`);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '유사 이미지 검색에 실패했습니다.');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching similar images:', error);
            throw error;
        }
    }
};
