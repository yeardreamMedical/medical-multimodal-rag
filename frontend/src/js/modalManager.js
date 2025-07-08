// frontend/src/js/modalManager.js

export class ModalManager {
    constructor(uiManager, apiService) {
        this.uiManager = uiManager;
        this.apiService = apiService;
    }

    async showSimilarImages(currentTopic) {
        if (!currentTopic) {
            return;
        }

        try {
            const similarImagesResponse = await this.apiService.getSimilarImagesAPI(currentTopic);
            const similarImages = similarImagesResponse.images;
            
            this.uiManager.elements.similarImagesGrid.innerHTML = '';
            
            similarImages.forEach((imageData) => {
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
                
                imageItem.addEventListener('click', () => {
                    this.viewLargerImage(imageData);
                });
                
                this.uiManager.elements.similarImagesGrid.appendChild(imageItem);
            });
            
            this.uiManager.elements.similarImagesModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';

        } catch (error) {
            console.error('Error showing similar images:', error);
            this.uiManager.showError('유사 이미지 로드 중 오류가 발생했습니다.');
        }
    }

    hideSimilarImages() {
        this.uiManager.elements.similarImagesModal.style.display = 'none';
        document.body.style.overflow = '';
    }

    viewLargerImage(imageData) {
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
        
        const closeBtn = largeImageModal.querySelector('.close-modal-btn');
        const overlay = largeImageModal.querySelector('.modal-overlay');
        
        const closeLargeImage = () => {
            document.body.removeChild(largeImageModal);
        };
        
        closeBtn.addEventListener('click', closeLargeImage);
        overlay.addEventListener('click', closeLargeImage);
        
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                closeLargeImage();
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
    }
}
