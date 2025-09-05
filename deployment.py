
"""
Production deployment utilities
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any

class ProductionLogger:
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("health_copilot")
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler
        fh = logging.FileHandler(f"health_copilot_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_query(self, query: str, response: str, metrics: Dict[str, Any]):
        """Log query processing details"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "response_length": len(response),
            "metrics": metrics
        }
        self.logger.info(f"Query processed: {json.dumps(log_data)}")
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(f"Error: {error}, Context: {json.dumps(context or {})}")

class HealthMetricsCollector:
    def __init__(self):
        self.daily_stats = {}
        self.performance_trends = []
    
    def record_interaction(self, metrics: Dict[str, Any]):
        """Record user interaction metrics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'total_queries': 0,
                'avg_response_time': 0,
                'function_calls': 0,
                'user_satisfaction': []
            }
        
        stats = self.daily_stats[today]
        stats['total_queries'] += 1
        stats['avg_response_time'] = (
            (stats['avg_response_time'] * (stats['total_queries'] - 1) + 
             metrics.get('response_time', 0)) / stats['total_queries']
        )
        stats['function_calls'] += metrics.get('function_calls', 0)
    
    def get_daily_report(self, date: str = None) -> Dict[str, Any]:
        """Get daily performance report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.daily_stats.get(date, {})