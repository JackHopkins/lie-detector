"""
Utility functions for baseline evaluations to reduce duplication in evaluate.py
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ResultsAnalyzer:
    """Analyzes evaluation results and extracts metrics."""
    
    def __init__(self, baseline_type: str):
        self.baseline_type = baseline_type
        self.scorer_name = self._get_scorer_name(baseline_type)
    
    def _get_scorer_name(self, baseline_type: str) -> str:
        """Get the scorer name for a baseline type."""
        scorer_mapping = {
            "escaped_transcript": "escaped_transcript_scorer",
            "llama_chat": "llama_chat_scorer", 
            "llama_chat_reasoning": "llama_chat_reasoning_scorer",
            "base_transcript_reasoning": "base_transcript_reasoning_scorer",
            "rowans_escaped_transcript": "escaped_transcript_scorer"
        }
        return scorer_mapping.get(baseline_type, "truth_scorer")
    
    def analyze_sample(self, sample) -> Dict[str, Any]:
        """Analyze a single sample and extract key information."""
        target = sample.target
        prediction = sample.output.completion.strip()
        original_did_lie = sample.metadata.get('did_lie', 'N/A') if sample.metadata else 'N/A'
        
        # Extract prediction from scorer
        if hasattr(sample, 'scores') and sample.scores and self.scorer_name in sample.scores:
            extracted_pred = sample.scores[self.scorer_name].answer
            metadata = sample.scores[self.scorer_name].metadata
            is_parseable = metadata.get('parseable', True) if metadata else False
        else:
            extracted_pred = "OTHER"
            is_parseable = False
        
        return {
            "target": target,
            "extracted_prediction": extracted_pred,
            "raw_prediction": prediction,
            "original_did_lie": original_did_lie,
            "is_parseable": is_parseable,
            "correct": target == extracted_pred if is_parseable else None
        }
    
    def calculate_confusion_matrix(self, results: List) -> Dict[str, int]:
        """Calculate confusion matrix metrics from results."""
        total_tp = total_tn = total_fp = total_fn = 0
        
        for result in results:
            if hasattr(result, 'scores') and result.scores and self.scorer_name in result.scores:
                metadata = result.scores[self.scorer_name].metadata
                if metadata and metadata.get('parseable', True):
                    total_tp += metadata.get('tp', 0)
                    total_tn += metadata.get('tn', 0)
                    total_fp += metadata.get('fp', 0)
                    total_fn += metadata.get('fn', 0)
        
        return {
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn
        }
    
    def calculate_metrics(self, confusion_matrix: Dict[str, int]) -> Dict[str, float]:
        """Calculate performance metrics from confusion matrix."""
        tp, tn, fp, fn = confusion_matrix["tp"], confusion_matrix["tn"], confusion_matrix["fp"], confusion_matrix["fn"]
        total_parseable = tp + tn + fp + fn
        
        if total_parseable == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        
        accuracy = (tp + tn) / total_parseable
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }


class ResultsSaver:
    """Handles saving of evaluation results in various formats."""
    
    def __init__(self, output_dir: str, baseline_type: str):
        self.output_dir = output_dir
        self.baseline_type = baseline_type
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def save_prompts(self, results: List, filename: str) -> str:
        """Save all prompts sent to the model to a text file."""
        filepath = os.path.join(self.output_dir, f"prompts_{self.timestamp}_{filename}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            self._write_prompts_header(f)
            
            for i, sample in enumerate(results):
                self._write_sample_prompt(f, i, sample)
        
        return filepath
    
    def _write_prompts_header(self, f):
        """Write header for prompts file."""
        f.write("="*80 + "\n")
        f.write("FULL PROMPTS SENT TO MODEL\n")
        f.write("="*80 + "\n\n")
    
    def _write_sample_prompt(self, f, i: int, sample):
        """Write a single sample's prompt to file."""
        analyzer = ResultsAnalyzer(self.baseline_type)
        sample_data = analyzer.analyze_sample(sample)
        
        f.write(f"SAMPLE {i+1:03d}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Target: {sample_data['target']}\n")
        f.write(f"Extracted Prediction: {sample_data['extracted_prediction']}\n")
        f.write(f"Original did_lie: {sample_data['original_did_lie']}\n")
        f.write(f"Parseable: {sample_data['is_parseable']}\n")
        f.write("-" * 50 + "\n")
        
        f.write("FULL PROMPT SENT TO MODEL:\n")
        f.write("〰" * 40 + "\n")
        
        # Handle both string and list inputs (for LLaMA chat format)
        if isinstance(sample.input, list):
            self._write_message_list(f, sample.input)
        else:
            f.write(sample.input)
        
        f.write("\n" + "〰" * 40 + "\n")
        f.write("MODEL RESPONSE:\n")
        f.write("▶ " + sample_data['raw_prediction'] + "\n")
        f.write("\n" + "="*80 + "\n\n")
    
    def _write_message_list(self, f, messages):
        """Write a list of chat messages to file."""
        for j, msg in enumerate(messages):
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # ChatMessage object
                f.write(f"[{j+1}] {msg.role.upper()}: {msg.content}\n")
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Dictionary format
                f.write(f"[{j+1}] {msg['role'].upper()}: {msg['content']}\n")
            else:
                # Fallback
                f.write(f"[{j+1}] {str(msg)}\n")
    
    def save_results_json(self, results: List, metrics: Dict[str, Any], filename: str) -> str:
        """Save detailed results to JSON file."""
        filepath = os.path.join(self.output_dir, f"results_{self.timestamp}_{filename}.json")
        
        analyzer = ResultsAnalyzer(self.baseline_type)
        
        # Process all samples
        processed_results = []
        for i, sample in enumerate(results):
            sample_data = analyzer.analyze_sample(sample)
            sample_data["sample_id"] = i + 1
            processed_results.append(sample_data)
        
        # Create full JSON structure
        json_data = {
            "metadata": {
                "timestamp": self.timestamp,
                "total_samples": len(results),
                "baseline_type": self.baseline_type,
                "model": "openrouter/meta-llama/llama-3.1-8b-instruct"  # TODO: make configurable
            },
            "metrics": metrics,
            "results": processed_results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_summary_txt(self, results: List, metrics: Dict[str, Any], filename: str) -> str:
        """Save summary to text file."""
        filepath = os.path.join(self.output_dir, f"results_{self.timestamp}_{filename}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            self._write_summary_header(f, len(results))
            self._write_metrics_summary(f, metrics)
            
            # TODO: Add subtask breakdown if needed
        
        return filepath
    
    def _write_summary_header(self, f, total_samples: int):
        """Write summary file header."""
        f.write("="*80 + "\n")
        f.write("LIE DETECTION BASELINE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {self.timestamp}\n")
        f.write(f"Baseline Type: {self.baseline_type}\n")
        f.write(f"Total Samples: {total_samples}\n\n")
    
    def _write_metrics_summary(self, f, metrics: Dict[str, Any]):
        """Write metrics summary to file."""
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Parseable samples: {metrics.get('parseable_samples', 0)}\n")
        f.write(f"Filtered samples: {metrics.get('filtered_samples', 0)}\n")
        f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
        f.write(f"F1 Score: {metrics.get('f1_score', 0):.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Positives: {metrics.get('tp', 0)}\n")
        f.write(f"True Negatives: {metrics.get('tn', 0)}\n")
        f.write(f"False Positives: {metrics.get('fp', 0)}\n")
        f.write(f"False Negatives: {metrics.get('fn', 0)}\n\n")


class EvaluationSummary:
    """Provides summary information about evaluation results."""
    
    def __init__(self, results: List, baseline_type: str):
        self.results = results
        self.baseline_type = baseline_type
        self.analyzer = ResultsAnalyzer(baseline_type)
    
    def print_filtering_summary(self):
        """Print summary of filtering/parsing results.""" 
        filtered_count = 0
        parseable_count = 0
        
        for sample in self.results:
            sample_data = self.analyzer.analyze_sample(sample)
            if sample_data["is_parseable"]:
                parseable_count += 1
            else:
                filtered_count += 1
        
        print(f"\n=== FILTERING SUMMARY ===")
        print(f"Total samples: {len(self.results)}")
        print(f"Filtered out (unparseable): {filtered_count}")
        print(f"Parseable samples: {parseable_count}")
    
    def print_classification_summary(self):
        """Print summary of classification results."""
        by_prediction = {"A": 0, "B": 0, "OTHER": 0}
        by_target = {"A": 0, "B": 0}
        correct_predictions = 0
        parseable_count = 0
        
        for sample in self.results:
            sample_data = self.analyzer.analyze_sample(sample)
            
            target_str = str(sample_data["target"]) if sample_data["target"] else "B"
            if target_str in by_target:
                by_target[target_str] += 1
            
            pred = sample_data["extracted_prediction"]
            if pred in by_prediction:
                by_prediction[pred] += 1
            
            if sample_data["is_parseable"]:
                parseable_count += 1
                if sample_data["correct"]:
                    correct_predictions += 1
        
        print(f"\n=== CLASSIFICATION SUMMARY (PARSEABLE ONLY) ===")
        if parseable_count > 0:
            accuracy_rate = correct_predictions / parseable_count
            print(f"Overall Accuracy: {correct_predictions}/{parseable_count} = {accuracy_rate:.3f}")
        else:
            print("No parseable samples to evaluate!")
        
        print(f"Target Distribution: A={by_target['A']}, B={by_target['B']}")
        print(f"Prediction Distribution: A={by_prediction['A']}, B={by_prediction['B']}, OTHER={by_prediction['OTHER']}")
    
    def print_final_metrics(self):
        """Print final performance metrics."""
        confusion_matrix = self.analyzer.calculate_confusion_matrix(self.results)
        metrics = self.analyzer.calculate_metrics(confusion_matrix)
        
        total_parseable = sum(confusion_matrix.values())
        
        if total_parseable > 0:
            print(f"\n{'='*60}")
            print("FINAL METRICS (PARSEABLE SAMPLES ONLY)")
            print(f"{'='*60}")
            print(f"Parseable samples: {total_parseable}/{len(self.results)} ({total_parseable/len(self.results):.1%})")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print("------------------------")
            print(f"True Positives: {confusion_matrix['tp']}")
            print(f"True Negatives: {confusion_matrix['tn']}")
            print(f"False Positives: {confusion_matrix['fp']}")
            print(f"False Negatives: {confusion_matrix['fn']}")
            print("------------------------")
        else:
            print("\nNo parseable samples to calculate metrics!")
        
        return metrics, confusion_matrix