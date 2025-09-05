import os
import json
import datetime
import logging
import asyncio
import time
import statistics
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod


try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Running in console mode only.")

# Core dependencies
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums and Data Classes
class ReasoningStep(Enum):
    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    REFLECT = "reflect"

@dataclass
class HealthMetrics:
    response_time: float
    reasoning_steps: int
    function_calls: int
    accuracy_score: float
    user_satisfaction: float
    completeness_score: float

@dataclass
class FunctionCall:
    name: str
    parameters: Dict[str, Any]
    result: Any = None
    execution_time: float = 0.0

@dataclass
class EvaluationMetrics:
    response_time: float
    reasoning_completeness: float
    function_call_accuracy: float
    medical_accuracy: float
    safety_compliance: float
    clarity_score: float
    empathy_score: float
    actionability_score: float
    token_usage: int
    api_calls: int
    error_count: int

# Function Implementations
class HealthFunction(ABC):
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class ScheduleReminderFunction(HealthFunction):
    def __init__(self):
        self.reminders = []
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        reminder_type = kwargs.get('reminder_type', 'general')
        frequency = kwargs.get('frequency', 'daily')
        time_of_day = kwargs.get('time_of_day', '09:00')
        message = kwargs.get('message', 'Health reminder')
        
        reminder_id = f"rem_{len(self.reminders) + 1}"
        reminder = {
            'id': reminder_id,
            'type': reminder_type,
            'frequency': frequency,
            'time': time_of_day,
            'message': message,
            'created_at': datetime.datetime.now().isoformat(),
            'active': True
        }
        
        self.reminders.append(reminder)
        
        return {
            'success': True,
            'reminder_id': reminder_id,
            'message': f'Reminder scheduled successfully for {frequency} at {time_of_day}',
            'details': reminder
        }

class RetrieveDietTipsFunction(HealthFunction):
    def __init__(self):
        self.diet_database = {
            'diabetes': [
                'Focus on complex carbohydrates like whole grains',
                'Include fiber-rich vegetables in every meal',
                'Monitor portion sizes and eat regular meals',
                'Choose lean proteins and healthy fats'
            ],
            'hypertension': [
                'Reduce sodium intake to less than 2300mg daily',
                'Increase potassium-rich foods like bananas and leafy greens',
                'Limit processed and packaged foods',
                'Follow the DASH diet principles'
            ],
            'heart_health': [
                'Include omega-3 fatty acids from fish twice weekly',
                'Eat a variety of colorful fruits and vegetables',
                'Choose whole grains over refined grains',
                'Limit saturated and trans fats'
            ],
            'weight_management': [
                'Practice portion control using smaller plates',
                'Eat protein with every meal to maintain satiety',
                'Stay hydrated with water throughout the day',
                'Include physical activity with dietary changes'
            ],
            'general': [
                'Eat 5-9 servings of fruits and vegetables daily',
                'Choose whole foods over processed options',
                'Stay hydrated with 8-10 glasses of water daily',
                'Practice mindful eating and regular meal times'
            ]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        condition = kwargs.get('condition', 'general').lower()
        count = kwargs.get('count', 3)
        
        tips = self.diet_database.get(condition, self.diet_database['general'])
        selected_tips = tips[:count] if count <= len(tips) else tips
        
        return {
            'success': True,
            'condition': condition,
            'tips': selected_tips,
            'total_available': len(tips)
        }

# Main Copilot Class
class PreventiveHealthCopilot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize function registry
        self.functions = {
            'schedule_reminder': ScheduleReminderFunction(),
            'retrieve_diet_tips': RetrieveDietTipsFunction()
        }
        
        self.conversation_history = []
        self.metrics_history = []
        
        # System prompt for multi-step reasoning
        self.system_prompt = """
You are a Preventive Health Copilot designed to help users maintain and improve their health through evidence-based recommendations. You use a structured reasoning approach:

**REASONING FRAMEWORK:**
1. OBSERVE: Analyze the user's health query and context
2. THINK: Apply medical knowledge and reasoning
3. ACT: Execute appropriate functions or provide recommendations  
4. REFLECT: Evaluate the response and suggest next steps

**AVAILABLE FUNCTIONS:**
- schedule_reminder(reminder_type, frequency, time_of_day, message): Schedule health reminders
- retrieve_diet_tips(condition, count): Get dietary recommendations for specific conditions

**GUIDELINES:**
- Always prioritize user safety and recommend consulting healthcare professionals for serious concerns
- Use evidence-based information and cite reliable sources when possible
- Break down complex health topics into understandable steps
- Be empathetic and supportive while maintaining professional boundaries
- Use the reasoning framework explicitly in your responses

**RESPONSE FORMAT:**
🔍 OBSERVE: [Your observation of the user's request]
🤔 THINK: [Your reasoning and analysis]
⚡ ACT: [Actions taken or recommendations provided]
💭 REFLECT: [Evaluation and next steps]
"""

    async def _call_function(self, function_name: str, parameters: Dict[str, Any]) -> FunctionCall:
        """Execute a function call and measure performance."""
        start_time = time.time()
        
        if function_name not in self.functions:
            return FunctionCall(
                name=function_name,
                parameters=parameters,
                result={'error': f'Function {function_name} not found'},
                execution_time=time.time() - start_time
            )
        
        try:
            result = await self.functions[function_name].execute(**parameters)
            execution_time = time.time() - start_time
            
            return FunctionCall(
                name=function_name,
                parameters=parameters,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            return FunctionCall(
                name=function_name,
                parameters=parameters,
                result={'error': str(e)},
                execution_time=time.time() - start_time
            )

    def _extract_function_calls(self, response_text: str, user_query: str) -> List[Dict[str, Any]]:
        """Extract function calls based on query analysis."""
        function_calls = []
        query_lower = user_query.lower()
        
        # Check for reminder needs
        reminder_keywords = ['remind', 'forget', 'schedule', 'medication', 'pill', 'take', 'remember']
        if any(keyword in query_lower for keyword in reminder_keywords):
            reminder_type = 'medication' if any(word in query_lower for word in ['medication', 'pill', 'drug']) else 'general'
            function_calls.append({
                'name': 'schedule_reminder',
                'parameters': {
                    'reminder_type': reminder_type,
                    'frequency': 'daily',
                    'time_of_day': '08:00',
                    'message': f'Daily {reminder_type} reminder'
                }
            })
        
        # Check for diet tips needs
        diet_keywords = ['diet', 'food', 'eat', 'nutrition', 'meal', 'dietary']
        if any(keyword in query_lower for keyword in diet_keywords):
            condition = 'general'
            if 'diabetes' in query_lower or 'diabetic' in query_lower:
                condition = 'diabetes'
            elif 'hypertension' in query_lower or 'blood pressure' in query_lower:
                condition = 'hypertension'
            elif 'heart' in query_lower or 'cardiac' in query_lower:
                condition = 'heart_health'
            elif 'weight' in query_lower or 'lose' in query_lower:
                condition = 'weight_management'
            
            function_calls.append({
                'name': 'retrieve_diet_tips',
                'parameters': {
                    'condition': condition,
                    'count': 4
                }
            })
        
        return function_calls

    async def process_query(self, user_query: str) -> Tuple[str, HealthMetrics]:
        """Process a user query with multi-step reasoning and function calling."""
        start_time = time.time()
        
        # Prepare the full prompt
        full_prompt = f"{self.system_prompt}\n\nUser Query: {user_query}\n\nProvide a structured response using the reasoning framework."
        
        try:
            # Generate initial response
            response = self.model.generate_content(full_prompt)
            initial_response = response.text
            
            # Extract and execute function calls
            function_calls = self._extract_function_calls(initial_response, user_query)
            executed_functions = []
            
            for func_call in function_calls:
                executed_func = await self._call_function(
                    func_call['name'], 
                    func_call['parameters']
                )
                executed_functions.append(executed_func)
            
            # Generate final response with function results
            if executed_functions:
                function_results = "\n".join([
                    f"✅ Function {func.name} executed: {func.result.get('message', 'Success')}"
                    for func in executed_functions if func.result.get('success')
                ])
                
                if function_results:
                    final_prompt = f"""
Based on your previous response and these function execution results:

{function_results}

Please provide a comprehensive final response that incorporates these results and maintains your structured reasoning format.

Original query: {user_query}
"""
                    
                    final_response = self.model.generate_content(final_prompt)
                    final_text = f"{initial_response}\n\n📋 **Actions Taken:**\n{function_results}\n\n{final_response.text}"
                else:
                    final_text = initial_response
            else:
                final_text = initial_response
            
            # Calculate metrics
            response_time = time.time() - start_time
            reasoning_steps = self._count_reasoning_steps(final_text)
            
            metrics = HealthMetrics(
                response_time=response_time,
                reasoning_steps=reasoning_steps,
                function_calls=len(executed_functions),
                accuracy_score=self._evaluate_accuracy(user_query, final_text),
                user_satisfaction=0.0,  # To be set by user feedback
                completeness_score=self._evaluate_completeness(final_text)
            )
            
            # Store in history
            self.conversation_history.append({
                'query': user_query,
                'response': final_text,
                'functions_called': [func.name for func in executed_functions],
                'timestamp': datetime.datetime.now().isoformat()
            })
            self.metrics_history.append(metrics)
            
            return final_text, metrics
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact support if the issue persists.\n\nError details: {str(e)}"
            return error_response, HealthMetrics(
                response_time=time.time() - start_time,
                reasoning_steps=0,
                function_calls=0,
                accuracy_score=0.0,
                user_satisfaction=0.0,
                completeness_score=0.0
            )

    def _count_reasoning_steps(self, response: str) -> int:
        """Count explicit reasoning steps in the response."""
        steps = ['OBSERVE:', 'THINK:', 'ACT:', 'REFLECT:']
        return sum(1 for step in steps if step in response)

    def _evaluate_accuracy(self, query: str, response: str) -> float:
        """Evaluate response accuracy (simplified heuristic)."""
        accuracy_keywords = [
            'evidence-based', 'research shows', 'studies indicate',
            'consult your doctor', 'healthcare professional', 'medical advice'
        ]
        
        score = 0.0
        for keyword in accuracy_keywords:
            if keyword in response.lower():
                score += 0.2
        
        # Check if response addresses the query appropriately
        if len(response.split()) > 50:  # Substantial response
            score += 0.3
        
        return min(score, 1.0)

    def _evaluate_completeness(self, response: str) -> float:
        """Evaluate response completeness."""
        required_elements = [
            ('OBSERVE:', 0.25), ('THINK:', 0.25), 
            ('ACT:', 0.25), ('REFLECT:', 0.25)
        ]
        
        score = 0.0
        for element, weight in required_elements:
            if element in response:
                score += weight
        
        return score

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregate performance metrics."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        metrics = self.metrics_history
        
        return {
            'total_queries': len(metrics),
            'average_response_time': sum(m.response_time for m in metrics) / len(metrics),
            'average_reasoning_steps': sum(m.reasoning_steps for m in metrics) / len(metrics),
            'total_function_calls': sum(m.function_calls for m in metrics),
            'average_accuracy': sum(m.accuracy_score for m in metrics) / len(metrics),
            'average_completeness': sum(m.completeness_score for m in metrics) / len(metrics),
            'queries_with_functions': sum(1 for m in metrics if m.function_calls > 0)
        }

# Evaluation System
class HealthCopilotEvaluator:
    def __init__(self):
        self.evaluation_history = []
        self.medical_keywords = {
            "evidence_based": [
                "research shows", "studies indicate", "clinical trials",
                "peer-reviewed", "evidence-based", "scientific literature"
            ],
            "safety_disclaimers": [
                "consult your doctor", "healthcare professional", "medical advice",
                "not a substitute", "seek professional", "emergency"
            ]
        }
    
    async def evaluate_response(
        self, 
        query: str, 
        response: str, 
        function_calls: List[str],
        response_time: float
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of a single response"""
        
        # Calculate all metrics
        metrics = EvaluationMetrics(
            response_time=response_time,
            reasoning_completeness=self._evaluate_reasoning_completeness(response),
            function_call_accuracy=self._evaluate_function_calls(query, function_calls),
            medical_accuracy=self._evaluate_medical_accuracy(response),
            safety_compliance=self._evaluate_safety_compliance(response),
            clarity_score=self._evaluate_clarity(response),
            empathy_score=self._evaluate_empathy(response),
            actionability_score=self._evaluate_actionability(response),
            token_usage=len(response.split()),
            api_calls=1 + len(function_calls),
            error_count=0
        )
        
        self.evaluation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'response_length': len(response),
            'function_calls': function_calls,
            'metrics': asdict(metrics)
        })
        
        return metrics
    
    def _evaluate_reasoning_completeness(self, response: str) -> float:
        """Evaluate reasoning framework completeness."""
        steps = ['OBSERVE:', 'THINK:', 'ACT:', 'REFLECT:']
        present_steps = sum(1 for step in steps if step in response)
        return present_steps / len(steps)
    
    def _evaluate_function_calls(self, query: str, function_calls: List[str]) -> float:
        """Evaluate function call appropriateness."""
        query_lower = query.lower()
        score = 0.0
        
        # Check if reminder function was appropriately used
        needs_reminder = any(word in query_lower for word in ['remind', 'forget', 'medication'])
        if 'schedule_reminder' in function_calls and needs_reminder:
            score += 0.5
        elif 'schedule_reminder' not in function_calls and needs_reminder:
            score += 0.1  # Missed opportunity
        elif 'schedule_reminder' not in function_calls and not needs_reminder:
            score += 0.5  # Correctly didn't use
        
        # Check if diet tips function was appropriately used
        needs_diet = any(word in query_lower for word in ['diet', 'food', 'eat', 'nutrition'])
        if 'retrieve_diet_tips' in function_calls and needs_diet:
            score += 0.5
        elif 'retrieve_diet_tips' not in function_calls and needs_diet:
            score += 0.1  # Missed opportunity
        elif 'retrieve_diet_tips' not in function_calls and not needs_diet:
            score += 0.5  # Correctly didn't use
        
        return min(score, 1.0)
    
    def _evaluate_medical_accuracy(self, response: str) -> float:
        """Evaluate medical accuracy."""
        response_lower = response.lower()
        score = 0.0
        
        # Evidence-based language
        evidence_count = sum(1 for keyword in self.medical_keywords["evidence_based"] 
                           if keyword in response_lower)
        score += min(evidence_count * 0.2, 0.4)
        
        # Safety considerations
        safety_count = sum(1 for keyword in self.medical_keywords["safety_disclaimers"]
                         if keyword in response_lower)
        score += min(safety_count * 0.3, 0.6)
        
        return min(score, 1.0)
    
    def _evaluate_safety_compliance(self, response: str) -> float:
        """Evaluate safety compliance."""
        safety_keywords = self.medical_keywords["safety_disclaimers"]
        safety_mentions = sum(1 for keyword in safety_keywords 
                            if keyword in response.lower())
        return min(safety_mentions * 0.33, 1.0)
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity."""
        # Simple heuristic based on structure and length
        has_structure = any(marker in response for marker in ['1.', '•', 'OBSERVE:', 'THINK:'])
        reasonable_length = 100 < len(response) < 1000
        return (0.5 if has_structure else 0.0) + (0.5 if reasonable_length else 0.2)
    
    def _evaluate_empathy(self, response: str) -> float:
        """Evaluate empathetic language."""
        empathy_words = ['understand', 'help', 'support', 'encourage', 'you']
        response_lower = response.lower()
        empathy_score = sum(0.2 for word in empathy_words if word in response_lower)
        return min(empathy_score, 1.0)
    
    def _evaluate_actionability(self, response: str) -> float:
        """Evaluate actionability."""
        action_words = ['start', 'try', 'consider', 'include', 'avoid', 'schedule']
        response_lower = response.lower()
        action_score = sum(0.15 for word in action_words if word in response_lower)
        return min(action_score, 1.0)

# Console Interface
async def console_interface():
    """Interactive console interface for testing"""
    print("🏥 Preventive Health Copilot - Interactive Console")
    print("=" * 50)
    print("Enter your health questions (type 'quit' to exit, 'metrics' for performance data)")
    print()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")
    
    # Initialize systems
    copilot = PreventiveHealthCopilot(api_key)
    evaluator = HealthCopilotEvaluator()
    
    while True:
        try:
            query = input("\n💬 Your health question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'metrics':
                metrics = copilot.get_performance_metrics()
                print("\n📊 Performance Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.3f}")
                    else:
                        print(f"   {key}: {value}")
                continue
            elif not query:
                continue
            
            print("\n🤔 Processing your question...")
            
            # Process query
            response, basic_metrics = await copilot.process_query(query)
            
            # Get function calls
            function_calls = []
            if copilot.conversation_history:
                function_calls = copilot.conversation_history[-1].get('functions_called', [])
            
            # Detailed evaluation
            detailed_metrics = await evaluator.evaluate_response(
                query, response, function_calls, basic_metrics.response_time
            )
            
            # Display response
            print("\n" + "="*60)
            print("🏥 HEALTH COPILOT RESPONSE:")
            print("="*60)
            print(response)
            print("\n" + "="*60)
            
            # Display metrics
            print("\n📊 Response Metrics:")
            print(f"   ⏱️  Response Time: {detailed_metrics.response_time:.2f}s")
            print(f"   🧠 Reasoning Steps: {basic_metrics.reasoning_steps}")
            print(f"   ⚡ Functions Called: {len(function_calls)}")
            print(f"   🎯 Medical Accuracy: {detailed_metrics.medical_accuracy:.2f}")
            print(f"   🛡️  Safety Compliance: {detailed_metrics.safety_compliance:.2f}")
            print(f"   📝 Clarity Score: {detailed_metrics.clarity_score:.2f}")
            
            if function_calls:
                print(f"   🔧 Functions Used: {', '.join(function_calls)}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Stay healthy!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# FastAPI Web Interface (optional)
if FASTAPI_AVAILABLE:
    class HealthQuery(BaseModel):
        query: str = Field(..., min_length=1, max_length=1000)
        user_id: Optional[str] = None

    class HealthResponse(BaseModel):
        response: str
        function_calls: List[str]
        metrics: Dict[str, Any]
        timestamp: str

    # Global instance for FastAPI
    copilot_instance = None

    app = FastAPI(title="Preventive Health Copilot API", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        global copilot_instance
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            copilot_instance = PreventiveHealthCopilot(api_key)
            print("🚀 Health Copilot API started")

    @app.post("/query", response_model=HealthResponse)
    async def process_query(query_request: HealthQuery):
        if not copilot_instance:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            response, metrics = await copilot_instance.process_query(query_request.query)
            function_calls = []
            if copilot_instance.conversation_history:
                function_calls = copilot_instance.conversation_history[-1].get('functions_called', [])
            
            return HealthResponse(
                response=response,
                function_calls=function_calls,
                metrics=asdict(metrics),
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Preventive Health Copilot API", "status": "running"}

# Main execution
async def main():
    """Main function to run the application"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "web" and FASTAPI_AVAILABLE:
            print("🌐 Starting web server...")
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        elif sys.argv[1] == "test":
            # Run test scenarios
            await run_test_scenarios()
        else:
            print("Available commands: web, test, or no argument for console")
    else:
        # Default: console interface
        await console_interface()

async def run_test_scenarios():
    """Run predefined test scenarios"""
    print("🧪 Running Test Scenarios")
    print("=" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return
    
    copilot = PreventiveHealthCopilot(api_key)
    evaluator = HealthCopilotEvaluator()
    
    test_queries = [
        "I'm pre-diabetic and want to improve my diet. Can you help?",
        "I keep forgetting to take my blood pressure medication",
        "What are some heart-healthy foods I should include?",
        "I have both diabetes and high blood pressure - comprehensive plan?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔬 Test {i}: {query}")
        print("-" * 40)
        
        response, metrics = await copilot.process_query(query)
        function_calls = copilot.conversation_history[-1].get('functions_called', [])
        
        detailed_metrics = await evaluator.evaluate_response(
            query, response, function_calls, metrics.response_time
        )
        
        print(f"✅ Response generated ({len(response)} chars)")
        print(f"⏱️  Time: {detailed_metrics.response_time:.2f}s")
        print(f"🎯 Medical Accuracy: {detailed_metrics.medical_accuracy:.2f}")
        print(f"🛡️  Safety: {detailed_metrics.safety_compliance:.2f}")
        print(f"🔧 Functions: {function_calls}")
        
        await asyncio.sleep(1)  # Rate limiting
    
    print(f"\n📊 Overall Performance:")
    perf_metrics = copilot.get_performance_metrics()
    for key, value in perf_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())