import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import questionRouter from './routes/question.js';
import similarImagesRouter from './routes/similarImages.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../dist')));

// API routes
app.use('/api', questionRouter);
app.use('/api', similarImagesRouter);

// Serve frontend for all other routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../dist/index.html'));
});

export default app;
