import json
import statistics
from datetime import datetime
from collections import defaultdict

from logs.logger import get_logger

logger = get_logger(__name__)

FEEDBACK_LOG = "models/slailm/logs/human_feedback_log.jsonl"

class HumanEval:
    """
    Handles the collection, storage, and basic analysis of human feedback 
    on model responses.
    """

    RATING_CATEGORIES = ["coherence", "safety", "helpfulness", "fluency", "relevance", "factuality"]
    SAFETY_KEYWORDS = {"badword", "harmful", "unsafe"} # Simple example

    @staticmethod
    def get_human_ratings(prompt: str, response: str) -> dict:
        """
        Simulates gathering multiple ratings from a human evaluator.

        Args:
            prompt: The input prompt given to the model.
            response: The model's generated response.

        Returns:
            A dictionary containing ratings for various categories (1-5 scale).
        """
        print("-" * 20)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 20)
        
        ratings = {}
        for category in HumanEval.RATING_CATEGORIES:
            while True:
                try:
                    # In a real scenario, this would come from a UI or API
                    rating_input = input(f"Enter rating for '{category}' (1-5): ")
                    rating = float(rating_input)
                    if 1.0 <= rating <= 5.0:
                        ratings[category] = rating
                        break
                    else:
                        print("Rating must be between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        # Log the collected feedback
        HumanEval.log_feedback(prompt, response, ratings)
        return ratings

    @staticmethod
    def get_safety_score_basic(response: str) -> float:
        """
        Provides a basic automated safety check based on keywords.
        Returns a score (e.g., 1.0 for potentially unsafe, 5.0 for likely safe).
        """
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in HumanEval.SAFETY_KEYWORDS):
            return 1.0 # Indicates potential safety issue
        # Add more sophisticated checks here if needed (e.g., toxicity model)
        return 5.0 # Assumed safe otherwise

    @staticmethod
    def log_feedback(prompt: str, response: str, ratings: dict):
        """
        Logs the collected human feedback to a persistent store (e.g., file).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "ratings": ratings
        }
        try:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            print(f"Error logging human feedback: {e}")

    @staticmethod
    def get_average_ratings(last_n: int = 100) -> dict:
        """
        Calculates average ratings from the logged feedback.

        Args:
            last_n: The number of recent feedback entries to consider.

        Returns:
            A dictionary with average scores for each rating category.
        """
        all_ratings = defaultdict(list)
        try:
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            recent_lines = lines[-last_n:]
            for line in recent_lines:
                try:
                    entry = json.loads(line)
                    for category, rating in entry.get("ratings", {}).items():
                         if category in HumanEval.RATING_CATEGORIES:
                            all_ratings[category].append(rating)
                except json.JSONDecodeError:
                    continue # Skip corrupted lines

        except FileNotFoundError:
            print(f"Feedback log '{FEEDBACK_LOG}' not found.")
            return {cat: 0.0 for cat in HumanEval.RATING_CATEGORIES}
        except IOError as e:
            print(f"Error reading feedback log: {e}")
            return {cat: 0.0 for cat in HumanEval.RATING_CATEGORIES}
            
        average_scores = {}
        for category, scores in all_ratings.items():
            if scores:
                average_scores[category] = statistics.mean(scores)
            else:
                average_scores[category] = 0.0
                
        # Ensure all categories are present, even if no data exists
        for cat in HumanEval.RATING_CATEGORIES:
            if cat not in average_scores:
                 average_scores[cat] = 0.0

        return average_scores
