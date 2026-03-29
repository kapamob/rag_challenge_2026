import json
import sys
from pathlib import Path

def validate_submission(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return False

    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format: {e}")
            return False

    answers = data.get("answers", [])
    if not answers:
        print("Warning: No answers found in submission.")
        return False

    total = len(answers)
    null_count = 0
    empty_retrieval_count = 0
    format_errors = 0

    # Load questions to know expected types
    questions_path = path.parent / "starter_kit/questions.json"
    if not questions_path.exists():
        questions_path = Path("starter_kit/questions.json")
    
    questions = {}
    if questions_path.exists():
        with open(questions_path, 'r') as f:
            questions = {q['id']: q for q in json.load(f)}

    for ans in answers:
        q_id = ans.get("question_id")
        answer = ans.get("answer")
        
        q_meta = questions.get(q_id, {})
        q_type = q_meta.get("answer_type", "free_text")
        
        retrieval = ans.get("telemetry", {}).get("retrieval", {})
        pages = retrieval.get("retrieved_chunk_pages", [])

        # Check for null answers
        if answer is None:
            null_count += 1
            continue
        
        # Check for empty retrieval when answer exists
        if not pages:
            empty_retrieval_count += 1
            print(f"  [Grounding Alert] Question {q_id}: Answer exists but retrieval is empty!")

        # Format Validation (Phase 13.2) - Type Aware
        import re
        if q_type == "names":
            if not isinstance(answer, list):
                format_errors += 1
                print(f"  [Type Error] Question {q_id}: Expected names (list), got {type(answer).__name__}")
            else:
                for item in answer:
                    if not isinstance(item, str):
                        format_errors += 1
                        print(f"  [Type Error] Question {q_id}: Names list contains non-string: {item}")
        elif q_type == "number":
             if not isinstance(answer, (int, float)):
                 format_errors += 1
                 print(f"  [Type Error] Question {q_id}: Expected number, got {type(answer).__name__}")
        elif q_type == "boolean":
             if not isinstance(answer, bool):
                 format_errors += 1
                 print(f"  [Type Error] Question {q_id}: Expected boolean, got {type(answer).__name__}")
        elif q_type == "date":
             if not isinstance(answer, str):
                 format_errors += 1
                 print(f"  [Type Error] Question {q_id}: Expected date string, got {type(answer).__name__}")
             elif not re.match(r"^\d{4}-\d{2}-\d{2}$", answer):
                 format_errors += 1
                 print(f"  [Format Error] Question {q_id}: Date format should be YYYY-MM-DD, got: {answer}")
        elif q_type == "name":
             if not isinstance(answer, str):
                 format_errors += 1
                 print(f"  [Type Error] Question {q_id}: Expected name string, got {type(answer).__name__}")

    null_percent = (null_count / total) * 100
    
    print(f"\n--- Validation Report: {path.name} ---")
    print(f"Total Questions: {total}")
    print(f"Null Answers: {null_count} ({null_percent:.1f}%)")
    print(f"Answered with Empty Retrieval: {empty_retrieval_count}")
    print(f"Format/Type Errors: {format_errors}")
    # Threshold checks
    if null_percent > 25:
        print("CRITICAL: Too many NULL answers (>25%). Check retrieval/LLM logs.")
    
    if empty_retrieval_count > 0:
        print(f"WARNING: {empty_retrieval_count} answers lack grounding pages. This will lower your G-score.")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_file>")
    else:
        validate_submission(sys.argv[1])
