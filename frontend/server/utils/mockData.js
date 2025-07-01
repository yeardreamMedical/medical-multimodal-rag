// Mock data and generator functions for questions and similar images

export function generateMockQuestion(topic) {
    const mockQuestions = {
        '폐렴': {
            question: "65세 남성이 3일 전부터 발열, 기침, 가래를 주소로 내원하였다. 체온 38.5°C, 맥박 100회/분, 호흡수 24회/분, 혈압 130/80 mmHg이다. 흉부 청진에서 우하엽에 수포음이 들리며, 흉부 X선 검사에서 우하엽에 경화 소견이 관찰된다. 가장 적절한 진단은?",
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
        },
        '기흉': {
            question: "25세 키 큰 남성이 갑작스런 흉통과 호흡곤란을 주소로 응급실에 내원하였다. 좌측 흉통이 심하며 호흡 시 악화된다고 한다. 활력징후는 혈압 120/80 mmHg, 맥박 110회/분, 호흡수 28회/분이다. 흉부 X선 검사 소견은?",
            options: [
                "좌측 폐의 허탈과 늑막강 내 공기 음영",
                "양측 폐야의 미만성 침윤",
                "심장 음영의 확대",
                "우측 늑막삼출",
                "정상 소견"
            ],
            answer: 0,
            explanation: "젊은 키 큰 남성에서 갑작스런 흉통과 호흡곤란은 자연기흉을 시사하는 전형적인 증상입니다. 흉부 X선에서 폐의 허탈과 늑막강 내 공기 음영이 관찰되는 것이 기흉의 특징적인 소견입니다.",
            topic_analysis: {
                estimated_topic: "자연기흉",
                difficulty_level: "중급",
                clinical_relevance: "high"
            },
            image_selection: {
                selected_image_type: "Pneumothorax",
                korean_name: "기흉",
                reason: "기흉의 전형적인 흉부 X선 소견을 보여주는 이미지가 진단에 필수적입니다."
            },
            selected_images: [
                {
                    image_path: "https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg?auto=compress&cs=tinysrgb&w=800",
                    score: 0.92,
                    labels: ["pneumothorax", "chest x-ray"]
                }
            ]
        }
    };
    return mockQuestions[topic];
}

export function generateGenericQuestion(topic) {
    return {
        question: `${topic}과 관련된 다음 환자 사례를 보고 가장 적절한 답을 선택하세요. 45세 환자가 관련 증상으로 내원하였습니다. 병력과 신체검사, 검사 소견을 종합할 때 가장 가능성이 높은 진단은?`,
        options: [
            `${topic}의 초기 단계`,
            `${topic}의 전형적인 형태`,
            `${topic}과 유사한 다른 질환`,
            `${topic}의 합병증`,
            `${topic}과 무관한 질환`
        ],
        answer: 1,
        explanation: `이 환자의 증상과 검사 소견은 ${topic}의 전형적인 특징을 보여줍니다. 임상 양상과 검사 결과를 종합하여 진단할 수 있습니다.`,
        topic_analysis: {
            estimated_topic: topic,
            difficulty_level: "중급",
            clinical_relevance: "medium"
        },
        image_selection: {
            selected_image_type: "None",
            korean_name: "이미지 없음",
            reason: "이 주제는 텍스트 기반 문제로 충분히 학습 효과를 얻을 수 있습니다."
        },
        selected_images: []
    };
}

export function getMockSimilarImages(topic) {
    const similarImagesData = {
        '폐렴': [
            {
                id: 'pneumonia_001',
                title: '폐렴 - 우하엽 경화',
                description: '우하엽에 국소적인 경화 소견이 관찰되는 전형적인 세균성 폐렴',
                url: 'https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 94,
                labels: ['pneumonia', 'consolidation', 'right lower lobe']
            },
            {
                id: 'pneumonia_002',
                title: '폐렴 - 좌하엽 침윤',
                description: '좌하엽에 침윤성 음영을 보이는 폐렴 소견',
                url: 'https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 89,
                labels: ['pneumonia', 'infiltrate', 'left lower lobe']
            },
            {
                id: 'pneumonia_003',
                title: '다발성 폐렴',
                description: '양측 폐야에 다발성 침윤을 보이는 폐렴',
                url: 'https://images.pexels.com/photos/4386370/pexels-photo-4386370.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 87,
                labels: ['pneumonia', 'bilateral', 'multiple']
            },
            {
                id: 'pneumonia_004',
                title: '폐렴 회복기',
                description: '치료 후 호전되고 있는 폐렴의 흉부 X선 소견',
                url: 'https://images.pexels.com/photos/4386444/pexels-photo-4386444.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 82,
                labels: ['pneumonia', 'resolving', 'treatment response']
            }
        ],
        '기흉': [
            {
                id: 'pneumothorax_001',
                title: '자연기흉 - 좌측',
                description: '좌측 폐의 부분적 허탈을 보이는 자연기흉',
                url: 'https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 92,
                labels: ['pneumothorax', 'left', 'spontaneous']
            },
            {
                id: 'pneumothorax_002',
                title: '긴장성 기흉',
                description: '종격동 편위를 동반한 긴장성 기흉',
                url: 'https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 88,
                labels: ['pneumothorax', 'tension', 'mediastinal shift']
            },
            {
                id: 'pneumothorax_003',
                title: '기흉 - 우측',
                description: '우측 폐의 완전 허탈을 보이는 기흉',
                url: 'https://images.pexels.com/photos/4386370/pexels-photo-4386370.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 85,
                labels: ['pneumothorax', 'right', 'complete collapse']
            },
            {
                id: 'pneumothorax_004',
                title: '기흉 치료 후',
                description: '흉관 삽입 후 폐 재팽창된 상태',
                url: 'https://images.pexels.com/photos/4386444/pexels-photo-4386444.jpeg?auto=compress&cs=tinysrgb&w=400',
                similarity: 79,
                labels: ['pneumothorax', 'post treatment', 'chest tube']
            }
        ]
    };
    return similarImagesData[topic] || similarImagesData['폐렴'];
}
