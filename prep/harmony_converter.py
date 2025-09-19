"""
Harmony Format Converter for Lie Detection Dataset

This script converts reasoning-enhanced samples to OpenAI Harmony format
using the official openai-harmony library and prepares tokenized data for Together AI.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import tiktoken
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort
    )
except ImportError:
    print("Please install openai-harmony: pip install openai-harmony")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HarmonyConverter:
    """Converts lie detection samples to Harmony format with tokenization."""

    def __init__(self, reasoning_effort: str = "high", max_length: int = 4096):
        """
        Initialize the Harmony converter.

        Args:
            reasoning_effort: One of "high", "medium", "low"
            max_length: Maximum sequence length for tokenization
        """
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = self._get_reasoning_effort(reasoning_effort)
        self.max_length = max_length

        # Initialize tiktoken for tokenization
        # Using o200k_harmony encoding for Harmony format
        try:
            self.tokenizer = tiktoken.get_encoding("o200k_harmony")
            logger.info("o200k_harmony encoding found")
        except:
            # Fallback to o200k_base if harmony encoding not available
            logger.warning("o200k_harmony encoding not found, using o200k_base")
            self.tokenizer = tiktoken.get_encoding("o200k_harmony")

    def _get_reasoning_effort(self, effort: str) -> ReasoningEffort:
        """Convert string to ReasoningEffort enum."""
        mapping = {
            "high": ReasoningEffort.HIGH,
            "medium": ReasoningEffort.MEDIUM,
            "low": ReasoningEffort.LOW
        }
        return mapping.get(effort.lower(), ReasoningEffort.HIGH)

    def create_system_message(self, include_date: bool = True) -> SystemContent:
        """
        Create the system message for Harmony format.

        Args:
            include_date: Whether to include current date

        Returns:
            SystemContent object
        """
        system_content = SystemContent.new().with_reasoning_effort(self.reasoning_effort)

        if include_date:
            current_date = datetime.now().strftime("%Y-%m-%d")
            system_content = system_content.with_conversation_start_date(current_date)

        return system_content

    def create_developer_message(self, instructions: str = None) -> DeveloperContent:
        """
        Create the developer message with instructions.

        Args:
            instructions: Custom instructions for the model

        Returns:
            DeveloperContent object
        """
        if instructions is None:
            instructions = (
                "You are a helpful AI assistant trained to detect deception in conversations. "
                "Analyze conversations carefully and determine if lies were told."
            )

        return DeveloperContent.new().with_instructions(instructions)

    def convert_sample_to_harmony(self, sample: Dict[str, Any]) -> Conversation:
        """
        Convert a single enhanced sample to Harmony format.

        Args:
            sample: Enhanced sample with reasoning

        Returns:
            Harmony Conversation object
        """
        messages_list = []

        # Add system message
        system_content = self.create_system_message()
        messages_list.append(Message.from_role_and_content(Role.SYSTEM, system_content))

        # Add developer message
        developer_content = self.create_developer_message()
        messages_list.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))

        # Add conversation messages
        conversation_messages = sample.get('messages', [])
        for msg in conversation_messages:
            role_str = msg.get('role', 'user')
            content = msg.get('content', '')

            # Map roles to Harmony Role enum
            if role_str == 'user':
                role = Role.USER
            elif role_str == 'assistant':
                role = Role.ASSISTANT
            elif role_str == 'system':
                # System prompts in conversation become developer messages
                role = Role.DEVELOPER
            else:
                role = Role.USER  # Default fallback

            # For assistant messages, use final channel for regular responses
            if role == Role.ASSISTANT:
                message = Message.from_role_and_content(role, content)
                message = message.with_channel("final")
            else:
                message = Message.from_role_and_content(role, content)

            messages_list.append(message)

        # Add lie detection prompt
        lie_prompt = sample.get('lie_detection_prompt',
                                'Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful.')
        messages_list.append(Message.from_role_and_content(Role.USER, lie_prompt))

        # ALWAYS add the hardcoded reasoning in analysis channel
        # This ensures every sample has the reasoning block
        hardcoded_reasoning = "I need to think about whether I lied or not"
        analysis_message = Message.from_role_and_content(
            Role.ASSISTANT,
            hardcoded_reasoning
        ).with_channel("analysis")

        messages_list.append(analysis_message)

        # Final channel - the conclusion and answer
        final_content = conversation_messages[-1]['content']
        logger.info("Final: "+str(final_content))
        if final_content:
            final_message = Message.from_role_and_content(
                Role.ASSISTANT,
                final_content
            ).with_channel("final")
            messages_list.append(final_message)

        # Create conversation from messages
        convo = Conversation.from_messages(messages_list)

        return convo

    def render_and_tokenize(self, conversation: Conversation) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """
        Render conversation and tokenize for training.
        Only the final token (the answer A or B) will be trained on.

        Args:
            conversation: Harmony Conversation object

        Returns:
            Tuple of (input_ids, labels, metadata)
        """
        # Get tokens using the harmony encoding
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        # Convert tokens to list if needed
        if not isinstance(tokens, list):
            tokens = list(tokens)

        # Create labels (same as input_ids but with masking)
        # We'll use -100 as the ignore token for loss calculation
        labels = tokens.copy()

        logger.debug(len(labels))

        final_answer_index = -20
        logger.info("Encoded: " + str(labels[final_answer_index:]))
        logger.info("Decoded: "+self.tokenizer.decode(labels[final_answer_index:]))

        # IMPORTANT: Mask ALL tokens except the very last one
        # We only want to train on the final token (A or B)
        # for i in range(len(labels)):
        #     if i < len(labels)+final_answer_index:
        #         labels[i] = -100

        # Find the position of just 'A' or 'B'
        # Looking at your sample, it should be right before <|end|>
        answer_position = None
        for i in range(len(tokens) - 1, max(0, len(tokens) - 10), -1):
            decoded = self.tokenizer.decode([tokens[i]])
            if decoded.strip() in ['A', 'B']:
                answer_position = i
                logger.info(f"Found answer '{decoded}' at position {i}")
                break

        # Create labels - mask EVERYTHING except the single answer token
        labels = [-100] * len(tokens)
        if answer_position is not None:
            labels[answer_position] = tokens[answer_position]

        #logger.info("Labels: "+str(labels))

        # Store original length before truncation/padding
        original_length = len(tokens)

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]
            truncated = True
        else:
            # Pad with the pad token (typically 0 or a special pad token ID)
            pad_token_id = 0  # You might need to adjust this based on the tokenizer
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [pad_token_id] * padding_length
            labels = labels + [-100] * padding_length  # Don't compute loss on padding
            truncated = False

        metadata = {
            'original_length': original_length,
            'truncated': truncated,
            'mask_position': original_length - 1,  # Position of the final token
            'final_token_id': tokens[original_length - 1] if original_length > 0 else None
        }

        return tokens, labels, metadata

    def tokenize_text_fallback(self, text: str) -> Tuple[List[int], List[int], str]:
        """
        Fallback tokenization using tiktoken directly.
        Only the final token will be trained on.

        Args:
            text: Text to tokenize

        Returns:
            Tuple of (input_ids, labels, text)
        """
        # Encode the text
        tokens = self.tokenizer.encode(text)

        # Create labels with masking
        labels = tokens.copy()

        # IMPORTANT: Mask ALL tokens except the very last one
        # We only want to train on the final token (A or B)
        for i in range(len(labels) - 1):
            labels[i] = -100

        logger.info("Label: "+self.tokenizer.decode([labels[-1]])[0])
        # Store original length before truncation/padding
        original_length = len(tokens)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_token_id = 0
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

        return tokens, labels, text

    def process_file(self, input_file: Path, output_dir: Path,
                     save_parquet: bool = True, save_jsonl: bool = True) -> int:
        """
        Process a file of enhanced samples to Harmony format with tokenization.

        Args:
            input_file: Path to input JSONL file
            output_dir: Directory for output files
            save_parquet: Whether to save Parquet format
            save_jsonl: Whether to save JSONL format

        Returns:
            Number of samples processed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file type
        file_type = 'train' if 'train' in input_file.name else 'val'

        tokenized_samples = []
        text_samples = []
        metadata_list = []

        # Process each sample
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}...")

                sample = json.loads(line)

                try:
                    # Convert to Harmony format
                    conversation = self.convert_sample_to_harmony(sample)

                    # Render and tokenize
                    input_ids, labels, token_metadata = self.render_and_tokenize(conversation)

                    # Build text representation from original sample for debugging
                    text_parts = []

                    # Add system message
                    text_parts.append(f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: {self.reasoning_effort}<|end|>")

                    # Add developer message
                    text_parts.append(f"<|start|>developer<|message|>You are a helpful AI assistant trained to detect deception in conversations. Analyze conversations carefully and determine if lies were told.<|end|>")

                    # Add conversation messages from original sample
                    messages = sample.get('messages', [])
                    for msg in messages[:-1]:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'assistant':
                            text_parts.append(f"<|start|>{role}<|channel|>final<|message|>{content}<|end|>")
                        else:
                            text_parts.append(f"<|start|>{role}<|message|>{content}<|end|>")

                    # Add lie detection prompt
                    lie_prompt = sample.get('lie_detection_prompt',
                                            'Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful.')

                    if not lie_prompt in " ".join(text_parts):
                        text_parts.append(f"<|start|>user<|message|>{lie_prompt}<|end|>")
                    else:
                        logger.info("Lie prompt already in trace - skipping")

                    # ALWAYS add the hardcoded reasoning block
                    text_parts.append(
                        f"<|start|>assistant<|channel|>analysis<|message|>I need to think about whether I lied or not<|end|>")

                    if messages[-1].get('role', 'assistant') == 'assistant':
                        text_parts.append(f"<|start|>assistant<|channel|>final<|message|>{messages[-1].get('content', '')}<|end|>")
                    else:
                        logger.warning(f"Failed to add final output")
                    text = "\n".join(text_parts)

                except Exception as e:
                    logger.warning(f"Failed to process sample {i} with harmony library: {e}")
                    logger.warning("Falling back to text tokenization")
                    continue

                # Extract metadata
                metadata = sample.get('metadata', {})
                metadata.update(token_metadata)
                # Ensure sample_id is always a string
                if 'sample_id' not in metadata:
                    metadata['sample_id'] = f'sample_{i}'
                else:
                    metadata['sample_id'] = str(metadata['sample_id'])

                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]

                # Create tokenized sample for training (only required columns for Together AI)
                tokenized_sample = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

                tokenized_samples.append(tokenized_sample)
                metadata_list.append(metadata)

                # Create text sample for JSONL output
                text_sample = {
                    'text': text,
                    'metadata': metadata
                }
                text_samples.append(text_sample)

        logger.info(f"Processed {len(tokenized_samples)} samples")

        # Save JSONL format (text only, for debugging)
        if save_jsonl:
            jsonl_file = output_dir / f"{file_type}_harmony.jsonl"
            with open(jsonl_file, 'w') as f:
                for sample in text_samples:
                    f.write(json.dumps(sample) + '\n')
            logger.info(f"Text samples: {len(text_samples)}")
            logger.info(f"Saved JSONL to {jsonl_file}")

        # Save Parquet format for training (with tokenized data)
        if save_parquet:
            # Create DataFrame with tokenized data
            df = pd.DataFrame(tokenized_samples)

            # Save Parquet file with tokenized data
            parquet_file = output_dir / f"{file_type}_harmony.parquet"
            df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"Saved tokenized Parquet to {parquet_file}")

            # Log sample of data structure
            logger.info(f"Parquet columns: {df.columns.tolist()}")
            logger.info(f"Sample input_ids length: {len(df.iloc[0]['input_ids']) if len(df) > 0 else 0}")

            # Also save metadata separately
            meta_df = pd.DataFrame(metadata_list)
            meta_parquet = output_dir / f"{file_type}_metadata.parquet"
            meta_df.to_parquet(meta_parquet, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"Saved metadata to {meta_parquet}")

        return len(tokenized_samples)


def find_jsonl_pairs(directory: Path) -> List[Tuple[Path, Path]]:
    """
    Find all train.jsonl and val.jsonl pairs in a directory structure.

    Args:
        directory: Root directory to search

    Returns:
        List of tuples containing (train_path, val_path) for each fold
    """
    pairs = []

    # First check if the current directory has both train.jsonl and val.jsonl
    train_file = directory / 'train.jsonl'
    val_file = directory / 'val.jsonl'

    if train_file.exists() and val_file.exists():
        logger.info(f"Found train/val pair in: {directory}")
        return [(directory,)]

    # If not, walk through subdirectories
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir():
            # Skip harmony directories (already processed)
            if subdir.name == 'harmony':
                continue

            train_file = subdir / 'train.jsonl'
            val_file = subdir / 'val.jsonl'

            if train_file.exists() and val_file.exists():
                logger.info(f"Found train/val pair in: {subdir}")
                pairs.append((subdir,))
            else:
                # Check one level deeper
                for subsubdir in sorted(subdir.iterdir()):
                    if subsubdir.is_dir() and subsubdir.name != 'harmony':
                        train_file = subsubdir / 'train.jsonl'
                        val_file = subsubdir / 'val.jsonl'

                        if train_file.exists() and val_file.exists():
                            logger.info(f"Found train/val pair in: {subsubdir}")
                            pairs.append((subsubdir,))

    return pairs


def process_single_fold(args_tuple):
    """
    Process a single fold directory (for parallel processing).

    Args:
        args_tuple: Tuple of (fold_dir, reasoning_effort, max_length, no_parquet, no_jsonl)

    Returns:
        Dictionary with processing results
    """
    fold_dir, reasoning_effort, max_length, no_parquet, no_jsonl = args_tuple

    logger.info(f"Processing fold: {fold_dir}")

    # Initialize converter
    converter = HarmonyConverter(
        reasoning_effort=reasoning_effort,
        max_length=max_length
    )

    # Setup output directory
    output_dir = fold_dir / 'harmony'

    results = {
        'fold': str(fold_dir),
        'train_samples': 0,
        'val_samples': 0,
        'errors': []
    }

    # Process train file
    train_file = fold_dir / 'train.jsonl'
    if train_file.exists():
        try:
            num_train = converter.process_file(
                input_file=train_file,
                output_dir=output_dir,
                save_parquet=not no_parquet,
                save_jsonl=not no_jsonl
            )
            results['train_samples'] = num_train
            logger.info(f"Processed {num_train} training samples from {fold_dir.name}")
        except Exception as e:
            error_msg = f"Error processing train file in {fold_dir}: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

    # Process val file
    val_file = fold_dir / 'val.jsonl'
    if val_file.exists():
        try:
            num_val = converter.process_file(
                input_file=val_file,
                output_dir=output_dir,
                save_parquet=not no_parquet,
                save_jsonl=not no_jsonl
            )
            results['val_samples'] = num_val
            logger.info(f"Processed {num_val} validation samples from {fold_dir.name}")
        except Exception as e:
            error_msg = f"Error processing val file in {fold_dir}: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

    # Save fold-specific conversion report
    report = {
        'timestamp': datetime.now().isoformat(),
        'fold_dir': str(fold_dir),
        'output_dir': str(output_dir),
        'train_samples': results['train_samples'],
        'val_samples': results['val_samples'],
        'reasoning_effort': reasoning_effort,
        'max_length': max_length,
        'formats_created': []
    }

    if not no_jsonl:
        report['formats_created'].append('jsonl')
    if not no_parquet:
        report['formats_created'].append('parquet (tokenized)')

    if results['errors']:
        report['errors'] = results['errors']

    report_file = output_dir / 'conversion_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert reasoning-enhanced samples to Harmony format with tokenization')
    parser.add_argument('input_path', type=str,
                        help='Path to input file or directory containing fold directories with train.jsonl and val.jsonl files')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: same directory with /harmony suffix)')
    parser.add_argument('--reasoning-effort', type=str, default='high',
                        choices=['high', 'medium', 'low'],
                        help='Reasoning effort level')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--no-parquet', action='store_true',
                        help='Skip Parquet output')
    parser.add_argument('--no-jsonl', action='store_true',
                        help='Skip JSONL output')
    parser.add_argument('--parallel', action='store_true',
                        help='Process folds in parallel')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: number of CPUs)')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return

    # Check if input is a file or directory
    if input_path.is_file():
        # Single file processing (original behavior)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = input_path.parent / 'harmony'

        # Initialize converter
        converter = HarmonyConverter(
            reasoning_effort=args.reasoning_effort,
            max_length=args.max_length
        )

        # Process file
        logger.info(f"Converting {input_path} to Harmony format with tokenization...")
        num_processed = converter.process_file(
            input_file=input_path,
            output_dir=output_dir,
            save_parquet=not args.no_parquet,
            save_jsonl=not args.no_jsonl
        )

        # Save conversion report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_path),
            'output_dir': str(output_dir),
            'num_samples': num_processed,
            'reasoning_effort': args.reasoning_effort,
            'max_length': args.max_length,
            'formats_created': []
        }

        if not args.no_jsonl:
            report['formats_created'].append('jsonl')
        if not args.no_parquet:
            report['formats_created'].append('parquet (tokenized)')

        report_file = output_dir / 'conversion_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Conversion complete! Processed {num_processed} samples")
        logger.info(f"Output saved to {output_dir}")
        logger.info(f"Report saved to {report_file}")

    else:
        # Directory processing - find all train/val pairs
        logger.info(f"Searching for train/val JSONL pairs in: {input_path}")
        fold_dirs = find_jsonl_pairs(input_path)

        if not fold_dirs:
            logger.error("No train.jsonl/val.jsonl pairs found in the directory structure")
            return

        logger.info(f"Found {len(fold_dirs)} fold(s) to process")

        # Prepare arguments for parallel processing
        process_args = [
            (fold_dir[0], args.reasoning_effort, args.max_length,
             args.no_parquet, args.no_jsonl)
            for fold_dir in fold_dirs
        ]

        all_results = []

        if args.parallel:
            # Parallel processing
            max_workers = args.max_workers or multiprocessing.cpu_count()
            logger.info(f"Processing folds in parallel with {max_workers} workers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_fold, args_tuple): args_tuple
                          for args_tuple in process_args}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
        else:
            # Sequential processing
            logger.info("Processing folds sequentially")
            for args_tuple in process_args:
                try:
                    result = process_single_fold(args_tuple)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing fold: {e}")

        # Generate overall summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(input_path),
            'num_folds': len(fold_dirs),
            'reasoning_effort': args.reasoning_effort,
            'max_length': args.max_length,
            'total_train_samples': sum(r['train_samples'] for r in all_results),
            'total_val_samples': sum(r['val_samples'] for r in all_results),
            'folds_processed': [r['fold'] for r in all_results],
            'formats_created': []
        }

        if not args.no_jsonl:
            summary['formats_created'].append('jsonl')
        if not args.no_parquet:
            summary['formats_created'].append('parquet (tokenized)')

        # Collect any errors
        all_errors = []
        for result in all_results:
            if result['errors']:
                all_errors.extend(result['errors'])
        if all_errors:
            summary['errors'] = all_errors

        # Save summary report
        summary_file = input_path / 'harmony_conversion_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 50)
        logger.info("Directory conversion complete!")
        logger.info(f"Processed {len(all_results)} folds")
        logger.info(f"Total training samples: {summary['total_train_samples']}")
        logger.info(f"Total validation samples: {summary['total_val_samples']}")
        logger.info(f"Summary report saved to: {summary_file}")
        if all_errors:
            logger.warning(f"Encountered {len(all_errors)} errors during processing")
            logger.warning(str(all_errors))


if __name__ == "__main__":
    main()