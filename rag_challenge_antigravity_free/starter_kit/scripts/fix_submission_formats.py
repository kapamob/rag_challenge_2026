import os
import json
import sys
from pathlib import Path
from llama_index.llms.openai_like import OpenAILike

# Fix path for arlc
# Assuming script is in starter_kit/scripts/fix_submission_formats.py
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import get_config
CONFIG = get_config()

# OpenRouter setup for refiner
LLM = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    max_tokens=512
)

def refactor_answer(raw_value, target_type):
    if raw_value is None:
        return None
    
    # 15.3 Quote Scrubbing (Initial)
    if isinstance(raw_value, str):
        # Remove nested quotes if they exist (e.g. '"val"' or '""val""')
        raw_value = raw_value.strip()
        while (raw_value.startswith('"') and raw_value.endswith('"')) or (raw_value.startswith("'") and raw_value.endswith("'")):
             raw_value = raw_value[1:-1].strip()
        
    prompt = f"""
You are a legal formatting expert. Reformat the following raw answer for a RAG challenge.
Target Type: {target_type}
Raw Answer: {raw_value}

RULES:
1. date: Return ONLY YYYY-MM-DD.
2. names: Return ONLY a JSON list of strings.
3. name: Return ONLY the precise value (e.g., Case Number "SCT 295/2025"). No extra quotes.
4. number: Return ONLY a float or int.
5. boolean: Return ONLY true/false.
6. free_text: Keep legal faithfulness but ENSURE total length is UNDER 280 characters. 
   If the answer is too long, rewrite it to be more concise without losing key facts.

Return ONLY the reformatted/summarized value.
"""
    response = LLM.complete(prompt).text.strip()
    
    # Final cleanup of quotes from LLM response
    clean_res = response
    while (clean_res.startswith('"') and clean_res.endswith('"')) or (clean_res.startswith("'") and clean_res.endswith("'")):
         clean_res = clean_res[1:-1].strip()

    try:
        if target_type == "names":
            clean = clean_res.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        elif target_type == "number":
             return float(clean_res.replace(",", ""))
        elif target_type == "boolean":
             return clean_res.lower() == "true"
        elif target_type == "date":
             import re
             if re.match(r"^\d{4}-\d{2}-\d{2}$", clean_res):
                 return clean_res
             return clean_res
        elif target_type == "free_text":
             return clean_res[:280] # Safety truncate
        else:
            return clean_res
    except Exception as e:
        print(f"  [Error] Failed to parse refiner output: {e}")
        return raw_value

def main():
    questions_path = ROOT_DIR / "questions.json"
    submission_path = ROOT_DIR / "submission_v20_strict.json"
    output_path = ROOT_DIR / "submission_v20_strict_final.json"
    
    with open(questions_path, 'r') as f:
        questions = {q['id']: q for q in json.load(f)}
        
    with open(submission_path, 'r') as f:
        submission = json.load(f)
        
    print(f"Refining {len(submission['answers'])} answers...")
    
    fixed_count = 0
    for ans in submission['answers']:
        q_id = ans['question_id']
        q_meta = questions.get(q_id, {})
        target_type = q_meta.get('answer_type', 'free_text')
        raw_ans = ans['answer']
        
        needs_fix = False
        if target_type == 'date' and (raw_ans and len(str(raw_ans)) != 10): needs_fix = True
        if target_type == 'names' and not isinstance(raw_ans, list): needs_fix = True
        # Quote cleanup needed if it starts/ends with duplicates
        if isinstance(raw_ans, str) and (raw_ans.startswith('"') and raw_ans.endswith('"')): needs_fix = True
        # Length guard for free_text
        if target_type == 'free_text' and raw_ans and len(str(raw_ans)) > 280: needs_fix = True
        # Type cleanup
        if target_type in ['name', 'number', 'boolean'] and raw_ans and len(str(raw_ans)) > 100: needs_fix = True

        if needs_fix:
            print(f"  Fixing {q_id} ({target_type}): {str(raw_ans)[:50]}...")
            new_ans = refactor_answer(raw_ans, target_type)
            ans['answer'] = new_ans
            fixed_count += 1

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Fixed {fixed_count} answers. Saved to {output_path}")

if __name__ == "__main__":
    main()
