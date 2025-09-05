
"""
Comprehensive test scenarios for the Preventive Health Copilot
"""

TEST_SCENARIOS = {
    "basic_diet_advice": [
        "I want to eat healthier but don't know where to start",
        "What foods should I avoid for better health?",
        "Give me a simple meal plan for the week"
    ],
    
    "condition_specific": [
        "I'm pre-diabetic and need dietary guidance",
        "I have high blood pressure, what foods should I eat?",
        "My doctor said I have high cholesterol, help with diet",
        "I'm trying to lose weight safely"
    ],
    
    "reminder_scheduling": [
        "I keep forgetting to take my vitamins",
        "Set up reminders for my blood pressure medication",
        "I need to remember to check my blood sugar",
        "Help me remember to exercise regularly"
    ],
    
    "complex_queries": [
        "I'm diabetic and have high blood pressure, need comprehensive diet plan with reminders",
        "I'm planning to start a new workout routine but have heart conditions, guide me",
        "My family has history of diabetes, what preventive measures should I take?"
    ]
}

EVALUATION_CRITERIA = {
    "accuracy": {
        "description": "Medical accuracy and evidence-based recommendations",
        "weight": 0.3,
        "keywords": ["evidence-based", "research", "studies", "medical professional"]
    },
    
    "safety": {
        "description": "Appropriate disclaimers and safety considerations",
        "weight": 0.25,
        "keywords": ["consult doctor", "healthcare professional", "medical advice"]
    },
    
    "completeness": {
        "description": "Structured reasoning and comprehensive response",
        "weight": 0.25,
        "keywords": ["OBSERVE", "THINK", "ACT", "REFLECT"]
    },
    
    "functionality": {
        "description": "Appropriate use of function calls",
        "weight": 0.2,
        "keywords": ["schedule_reminder", "retrieve_diet_tips"]
    }
}