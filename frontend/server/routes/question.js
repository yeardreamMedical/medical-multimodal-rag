import express from 'express';
import { generateMockQuestion, generateGenericQuestion } from '../utils/mockData.js';

const router = express.Router();

// POST /api/generate-question
router.post('/generate-question', async (req, res) => {
    try {
        const { topic } = req.body;
        if (!topic) {
            return res.status(400).json({ error: '주제를 입력해주세요.' });
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
        const questionData = generateMockQuestion(topic) || generateGenericQuestion(topic);
        res.json(questionData);
    } catch (error) {
        console.error('Question generation error:', error);
        res.status(500).json({ error: '문제 생성 중 오류가 발생했습니다.' });
    }
});

export default router;
