import json
import glob
import os

# Get all JSON log files
log_files = glob.glob("logs/*.json")

data = []

for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Check if samples exist at root level
        if 'samples' not in log_data:
            print(f"Warning: No 'samples' found in {log_file}")
            continue
            
        model_name = log_data['eval']['model']
        task_name = log_data['eval']['task']
        task_id = log_data['eval']['task_id']
        
        for sample in log_data['samples']:
            try:
                question = sample['input']
                
                # Check if choices array exists and has content
                if 'output' not in sample or 'choices' not in sample['output'] or len(sample['output']['choices']) == 0:
                    print(f"Warning: No choices found in sample {sample.get('id', 'unknown')} in {log_file}")
                    continue
                    
                answer = sample['output']['choices'][0]['message']['content']
                target = sample['target']
                
                # Handle different score structures
                score_value = None
                judge_explanation = ''
                judge_answer = ''
                
                if 'scores' in sample and 'mask_combined_scorer' in sample['scores']:
                    score_data = sample['scores']['mask_combined_scorer']
                    score_value = score_data.get('value')
                    judge_explanation = score_data.get('explanation', '')
                    judge_answer = score_data.get('answer', '')
                
                formatted_ground_truth = sample['metadata'].get('formatted_ground_truth', '')
                
                # Extract additional metadata
                proposition = sample['metadata'].get('proposition', '')
                proposition_type = sample['metadata'].get('proposition_type', '')
                proposition_data = sample['metadata'].get('proposition_data', [])
                system_prompt = sample['metadata'].get('system_prompt', '')
                belief_elicit_1 = sample['metadata'].get('belief_elicit_1', '')
                belief_elicit_2 = sample['metadata'].get('belief_elicit_2', '')
                belief_elicit_3 = sample['metadata'].get('belief_elicit_3', '')
                is_doubling_down = sample['metadata'].get('is_doubling_down', False)
                
                # Extract model usage stats
                model_usage = sample.get('model_usage', {})
                total_time = sample.get('total_time', 0)
                working_time = sample.get('working_time', 0)
                
                data.append({
                    'task': task_name,
                    'sample_id': sample['id'],
                    'model': model_name,
                    'question': question,
                    'answer': answer,
                    'target': target,
                    'score': score_value,
                    'ground_truth': formatted_ground_truth,
                    
                    # Proposition metadata
                    'proposition': proposition,
                    'proposition_type': proposition_type,
                    'proposition_data': proposition_data,
                    'system_prompt': system_prompt,
                    
                    # Belief elicitation questions
                    'belief_elicit_1': belief_elicit_1,
                    'belief_elicit_2': belief_elicit_2,
                    'belief_elicit_3': belief_elicit_3,
                    'is_doubling_down': is_doubling_down,
                    
                    # Judge analysis
                    'judge_explanation': judge_explanation,
                    'judge_answer': judge_answer,
                    
                    # Performance metrics
                    'model_usage': model_usage,
                    'total_time': total_time,
                    'working_time': working_time,
                })
                
            except Exception as e:
                print(f"Error processing sample in {log_file}: {e}")
                continue
                
    except Exception as e:
        print(f"Error processing file {log_file}: {e}")
        continue

# Write to file
with open('parsed_jsons.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Parsed {len(data)} samples from {len(log_files)} files to parsed_jsons.json")