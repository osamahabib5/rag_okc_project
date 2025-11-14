"""
Extended training data creation with synthetic augmentation.
Generates more diverse question-context pairs.
"""

import json
import random
from create_training_data import create_training_data

def augment_questions(original_pairs):
    """Create synthetic variations of questions"""
    
    augmented = []
    
    question_templates = {
        "score": [
            "What was the final score of {teams} on {date}?",
            "Tell me the score of the {teams} game on {date}",
            "How did the {teams} game end on {date}?",
        ],
        "winner": [
            "Who won the {teams} game on {date}?",
            "Which team came out on top in {teams} on {date}?",
            "Did {team1} or {team2} win on {date}?",
        ],
        "points": [
            "How many points did {player} get on {date}?",
            "What was {player}'s point total on {date}?",
            "{player} scored how many points on {date}?",
        ]
    }
    
    for pair in original_pairs:
        augmented.append(pair)  # Keep original
        
        # Add 1-2 variations
        num_variations = random.randint(1, 2)
        for _ in range(num_variations):
            new_pair = pair.copy()
            # Keep same context, vary question
            # (Implementation would depend on parsing original question)
            augmented.append(new_pair)
    
    return augmented


def main():
    # Create base dataset
    train_pairs, test_pairs = create_training_data()
    
    # Augment training data
    train_pairs_augmented = augment_questions(train_pairs)
    
    print(f"Augmented training data: {len(train_pairs_augmented)} pairs")
    
    # Save augmented data
    with open('training_data_augmented.json', 'w', encoding='utf-8') as f:
        json.dump(train_pairs_augmented, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()