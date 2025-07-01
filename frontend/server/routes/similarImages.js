import express from 'express';
import { getMockSimilarImages } from '../utils/mockData.js';

const router = express.Router();

// GET /api/similar-images/:topic
router.get('/similar-images/:topic', async (req, res) => {
    try {
        const { topic } = req.params;
        await new Promise(resolve => setTimeout(resolve, 500));
        const images = getMockSimilarImages(topic);
        res.json({
            topic: topic,
            total_count: images.length,
            images: images
        });
    } catch (error) {
        console.error('Similar images error:', error);
        res.status(500).json({ error: '비슷한 이미지 검색 중 오류가 발생했습니다.' });
    }
});

export default router;
