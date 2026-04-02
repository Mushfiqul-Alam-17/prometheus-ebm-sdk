import json
import logging
import time
import os
import random
import re
from collections import defaultdict
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DATASET = "prometheus_200_multimodel_dataset.json"
OUT_DATASET = "prometheus_1000_dataset.json"
API_KEY = os.environ.get("GEMINI_API_KEY")

TARGET_PER_GROUP = 50  # 5 domains x 4 classes x 50 = 1000 total
BATCH_SIZE = 5

def extract_json_array(text):
    """Safely extract JSON array from LLM output."""
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing whole text
    try:
        if text.startswith('```json'):
            text = text[7:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        return []

def main():
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running: $env:GEMINI_API_KEY = 'your_key_here'")
        return

    client = genai.Client(api_key=API_KEY)
    
    # Load base dataset
    try:
        with open(BASE_DATASET, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        logging.info(f"Loaded {len(base_data)} existing problems.")
    except Exception as e:
        logging.error(f"Could not load base dataset: {e}")
        return

    # Group by domain and class
    inventory = defaultdict(list)
    for p in base_data:
        inventory[(p['domain'], p['problem_class'])].append(p)

    final_dataset = list(base_data)
    
    # Resume safety
    if os.path.exists(OUT_DATASET):
        with open(OUT_DATASET, 'r', encoding='utf-8') as f:
            existing_final = json.load(f)
            if len(existing_final) > len(final_dataset):
                final_dataset = existing_final
                logging.info(f"Resumed from {OUT_DATASET} with {len(final_dataset)} problems.")
                
                # Update inventory from resumed dataset
                inventory = defaultdict(list)
                for p in final_dataset:
                    inventory[(p['domain'], p['problem_class'])].append(p)

    # Generation loop
    for domain, options in [
        ('medical', 'MED'), ('financial', 'FIN'), 
        ('legal', 'LEG'), ('environmental', 'ENV'), ('social', 'SOC')
    ]:
        for p_class in ["DETERMINATE", "UNDERDETERMINED", "INSUFFICIENT", "CONTRADICTORY"]:
            group_key = (domain, p_class)
            current_count = len(inventory[group_key])
            
            if current_count >= TARGET_PER_GROUP:
                logging.info(f"Skipping {domain}/{p_class}: already have {current_count} items.")
                continue
                
            needed = TARGET_PER_GROUP - current_count
            logging.info(f"Generating {needed} items for {domain}/{p_class}...")
            
            # Use original ones as few-shot
            few_shot = [p for p in base_data if p['domain'] == domain and p['problem_class'] == p_class]
            
            while needed > 0:
                batch = min(BATCH_SIZE, needed)
                logging.info(f"  -> Requesting batch of {batch}...")
                
                examples_str = json.dumps(random.sample(few_shot, min(3, len(few_shot))), indent=2)
                
                prompt = (
                    f"You are a dataset curator for the PROMETHEUS-EBM reasoning benchmark.\n"
                    f"Generate {batch} NEW epistemic reasoning problems in the domain of '{domain}' "
                    f"with the problem class '{p_class}'.\n\n"
                    f"Here are reference examples to match tone, difficulty, and structure:\n{examples_str}\n\n"
                    f"REQUIREMENTS:\n"
                    f"1. Generate highly realistic, expert-level scenarios. No simple/trivial cases.\n"
                    f"2. Output format MUST be a pure JSON array of {batch} objects.\n"
                    f"3. Do not include markdown formatting.\n"
                    f"4. Structure Keys: problem_id (use prefix PBM-{options}-NEW), domain, problem_class, "
                    f"correct_solvability_class, system, user, ground_truth_answer, branching_factor."
                )
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-pro',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            response_mime_type="application/json"
                        )
                    )
                    
                    new_items = extract_json_array(response.text)
                    if len(new_items) > 0:
                        for idx, item in enumerate(new_items):
                            item['problem_id'] = f"PBM-{options}-{int(time.time()*1000)+idx}"
                            final_dataset.append(item)
                            inventory[group_key].append(item)
                        
                        needed -= len(new_items)
                        logging.info(f"  -> Success. Got {len(new_items)}. {needed} left for this group.")
                        
                        # Save checkpoint
                        with open(OUT_DATASET, 'w', encoding='utf-8') as f:
                            json.dump(final_dataset, f, indent=2)
                    else:
                        logging.warning("  -> Empty or invalid JSON received. Retrying...")
                except Exception as e:
                    logging.error(f"  -> API Error: {e}")
                    time.sleep(5)
                
                time.sleep(2)  # Rate limit safety

    logging.info(f"COMPLETED! Total dataset size: {len(final_dataset)}")

if __name__ == "__main__":
    main()
