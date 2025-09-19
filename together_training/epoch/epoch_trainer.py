#!/usr/bin/env python3
"""
Epoch-by-Epoch Trainer for TogetherAI

This module implements the main training orchestrator that manages
incremental training runs with automatic resumption and state persistence.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from together import Together

from .training_state import TrainingState
from .file_manager import FileManager

load_dotenv()

class EpochTrainer:
    """Main orchestrator for epoch-by-epoch training."""
    
    def __init__(self, 
                 api_key: str,
                 base_model: str = "openai/gpt-oss-120b",
                 learning_rate: float = 1e-4,
                 wandb_api_key: Optional[str] = None,
                 wandb_project_name: Optional[str] = None):
        """
        Initialize the epoch trainer.
        
        Args:
            api_key: TogetherAI API key
            base_model: Base model to start training from
            learning_rate: Learning rate for training
            wandb_api_key: Optional Weights & Biases API key for logging
            wandb_project_name: Optional Weights & Biases project name
        """
        self.client = Together(api_key=api_key)
        self.api_key = api_key
        self.base_model = base_model
        self.learning_rate = learning_rate
        
        # Check for wandb keys in environment if not provided
        self.wandb_api_key = wandb_api_key or os.getenv('WANDB_API_KEY')
        self.wandb_project_name = wandb_project_name or os.getenv('WANDB_PROJECT') or "lie-detection-epoch-training"
    
    def _normalize_status(self, api_status) -> str:
        """
        Convert TogetherAI status enum to normalized string.
        
        Args:
            api_status: Status from TogetherAI API (may be enum or string)
            
        Returns:
            Normalized status string
        """
        # Convert enum to string if needed
        status_str = str(api_status)
        
        # Map TogetherAI status values to our internal status
        status_mapping = {
            'FinetuneJobStatus.STATUS_PENDING': 'pending',
            'FinetuneJobStatus.STATUS_RUNNING': 'running', 
            'FinetuneJobStatus.STATUS_COMPLETED': 'completed',
            'FinetuneJobStatus.STATUS_CANCELLED': 'cancelled',
            'FinetuneJobStatus.STATUS_USER_ERROR': 'failed',
            'FinetuneJobStatus.STATUS_SYSTEM_ERROR': 'failed',
            'FinetuneJobStatus.STATUS_ERROR': 'failed',
            # Also handle direct string values
            'pending': 'pending',
            'running': 'running',
            'completed': 'completed',
            'cancelled': 'cancelled',
            'failed': 'failed',
            'error': 'failed'
        }
        
        normalized = status_mapping.get(status_str, status_str.lower())
        return normalized
    
    def _get_job_error_details(self, job_id: str) -> Optional[str]:
        """
        Get error details from a failed training job.
        
        Args:
            job_id: TogetherAI job ID
            
        Returns:
            Error message if found, None otherwise
        """
        try:
            events = self.client.fine_tuning.list_events(job_id)
            if events and hasattr(events, 'data') and events.data:
                # Look for error messages in events (most recent first)
                for event in reversed(events.data):
                    if hasattr(event, 'message'):
                        message = event.message.lower()
                        if any(keyword in message for keyword in ['error', 'failed', 'exception', 'invalid']):
                            return event.message
                            
                # If no error events found, return the last event
                if events.data:
                    return events.data[-1].message
            
            return None
            
        except Exception:
            return None
    
    def train_single_epoch(self, 
                          fold_path: str,
                          max_epochs: int = 5,
                          resume: bool = True) -> Dict[str, Any]:
        """
        Train for a single epoch, handling state management and resumption.
        
        Args:
            fold_path: Path to the fold directory containing train.jsonl and val.jsonl
            max_epochs: Maximum number of epochs to train
            resume: Whether to resume from existing state
            
        Returns:
            Dictionary with training results and next steps
        """
        fold_path = Path(fold_path).resolve()
        
        # Initialize training state
        training_state = TrainingState(str(fold_path), self.base_model)
        file_manager = FileManager(self.api_key, training_state, self.base_model)
        
        print(f"=== Epoch Training for {fold_path.name} ===")
        
        # Get current training status
        current_epoch = training_state.get_current_epoch()
        
        if current_epoch >= max_epochs:
            return {
                'status': 'completed',
                'message': f'Training completed. All {max_epochs} epochs finished.',
                'current_epoch': current_epoch,
                'models': training_state.get_all_models(),
                'summary': training_state.get_training_summary()
            }
        
        print(f"Next epoch to train: {current_epoch}")
        print(f"Progress: {current_epoch}/{max_epochs} epochs completed")
        
        # Validate and upload files
        try:
            train_file_id, val_file_id = file_manager.validate_and_upload_files(fold_path)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'File validation/upload failed: {e}',
                'current_epoch': current_epoch
            }
        
        # Determine if we need to continue from a checkpoint or start from base model
        if current_epoch == 0:
            # First epoch - start from base model
            base_model_for_epoch = self.base_model
            from_checkpoint = None
            print(f"Base model for epoch {current_epoch}: {base_model_for_epoch}")
        else:
            # Find the previous completed epoch's job ID for checkpoint
            prev_epoch = current_epoch - 1
            while prev_epoch >= 0:
                prev_epoch_info = training_state.get_epoch_info(prev_epoch)
                if (prev_epoch_info and 
                    prev_epoch_info.status == "completed" and 
                    prev_epoch_info.job_id):
                    from_checkpoint = prev_epoch_info.job_id
                    base_model_for_epoch = None  # Not used when from_checkpoint is set
                    print(f"Continuing from checkpoint (epoch {prev_epoch} job): {from_checkpoint}")
                    break
                prev_epoch -= 1
            else:
                # Fallback to base model if no previous checkpoint found
                base_model_for_epoch = self.base_model
                from_checkpoint = None
                print(f"No previous checkpoint found, using base model: {base_model_for_epoch}")
        
        # Create training job
        try:
            job_id = self._create_training_job(
                train_file_id=train_file_id,
                val_file_id=val_file_id,
                model=base_model_for_epoch,
                epoch_num=current_epoch,
                fold_name=fold_path.name,
                from_checkpoint=from_checkpoint
            )
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to create training job: {e}',
                'current_epoch': current_epoch
            }
        
        # Calculate learning rate for this epoch (for record keeping)
        epoch_learning_rate = self.learning_rate * (0.67 ** current_epoch)
        
        # Update training state
        training_state.add_epoch(current_epoch, job_id, epoch_learning_rate)
        
        print(f"✓ Training job started: {job_id}")
        print(f"Epoch {current_epoch} training in progress...")
        
        return {
            'status': 'training',
            'message': f'Epoch {current_epoch} training started',
            'current_epoch': current_epoch,
            'job_id': job_id,
            'base_model': base_model_for_epoch,
            'next_epoch': current_epoch + 1,
            'training_state': training_state,
            'summary': training_state.get_training_summary()
        }
    
    def monitor_and_wait(self, 
                        fold_path: str, 
                        timeout_minutes: int = 60) -> Dict[str, Any]:
        """
        Monitor the current training job until completion.
        
        Args:
            fold_path: Path to the fold directory
            timeout_minutes: Maximum time to wait for completion
            
        Returns:
            Dictionary with completion status and results
        """
        fold_path = Path(fold_path).resolve()
        training_state = TrainingState(str(fold_path), self.base_model)
        
        current_epoch = max(training_state.epochs.keys()) if training_state.epochs else 0
        epoch_info = training_state.get_epoch_info(current_epoch)
        
        if not epoch_info or not epoch_info.job_id:
            return {
                'status': 'error',
                'message': 'No active training job found'
            }
        
        if epoch_info.status != 'running':
            return {
                'status': 'error',
                'message': f'Job is not running (status: {epoch_info.status})'
            }
        
        job_id = epoch_info.job_id
        print(f"Monitoring job {job_id} for epoch {current_epoch}...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 30  # Start with 30 second intervals
        max_interval = 300   # Max 5 minute intervals
        
        while True:
            try:
                # Check if we've exceeded timeout
                if time.time() - start_time > timeout_seconds:
                    return {
                        'status': 'timeout',
                        'message': f'Training timeout after {timeout_minutes} minutes',
                        'job_id': job_id,
                        'epoch': current_epoch
                    }
                
                # Get job status
                job_info = self.client.fine_tuning.retrieve(job_id)
                raw_status = job_info.status
                status = self._normalize_status(raw_status)
                
                print(f"Epoch {current_epoch} - Job status: {raw_status} → {status}")
                
                if hasattr(job_info, 'output_name') and job_info.output_name:
                    print(f"Model ID: {job_info.output_name}")
                
                # Show recent events
                try:
                    events = self.client.fine_tuning.list_events(job_id)
                    if events and hasattr(events, 'data') and events.data:
                        recent_events = events.data[-3:]  # Last 3 events
                        for event in recent_events:
                            if hasattr(event, 'message'):
                                print(f"  Event: {event.message}")
                except Exception:
                    pass  # Events are optional
                
                if status == "completed":
                    model_id = job_info.output_name
                    training_state.update_epoch(current_epoch, model_id, "completed")
                    
                    print(f"✓ Epoch {current_epoch} completed successfully!")
                    print(f"Model ID: {model_id}")
                    
                    return {
                        'status': 'completed',
                        'message': f'Epoch {current_epoch} training completed',
                        'epoch': current_epoch,
                        'model_id': model_id,
                        'job_id': job_id,
                        'next_epoch': current_epoch + 1,
                        'summary': training_state.get_training_summary()
                    }
                
                elif status in ["failed", "cancelled"]:
                    training_state.update_epoch(current_epoch, status=status)
                    
                    # Try to get failure details
                    error_details = "No details available"
                    try:
                        events = self.client.fine_tuning.list_events(job_id)
                        if events and hasattr(events, 'data') and events.data:
                            for event in reversed(events.data):
                                if hasattr(event, 'message') and 'error' in event.message.lower():
                                    error_details = event.message
                                    break
                    except Exception:
                        pass
                    
                    return {
                        'status': 'failed',
                        'message': f'Epoch {current_epoch} training failed: {error_details}',
                        'epoch': current_epoch,
                        'job_id': job_id,
                        'error_details': error_details
                    }
                
                else:
                    # Still running - wait and check again
                    print(f"Waiting {check_interval} seconds before next check...")
                    time.sleep(check_interval)
                    
                    # Gradually increase check interval
                    check_interval = min(check_interval + 10, max_interval)
                
            except Exception as e:
                print(f"Error checking job status: {e}")
                return {
                    'status': 'error',
                    'message': f'Error monitoring job: {e}',
                    'job_id': job_id,
                    'epoch': current_epoch
                }
    
    def train_until_complete(self, 
                           fold_path: str,
                           max_epochs: int = 5,
                           timeout_per_epoch: int = 60) -> Dict[str, Any]:
        """
        Train all epochs sequentially until completion.
        
        Args:
            fold_path: Path to the fold directory
            max_epochs: Maximum number of epochs to train
            timeout_per_epoch: Timeout per epoch in minutes
            
        Returns:
            Dictionary with final training results
        """
        fold_path = Path(fold_path).resolve()
        results = []
        
        print(f"Starting complete training for {fold_path.name}")
        print(f"Target: {max_epochs} epochs")
        
        while True:
            # Start next epoch
            train_result = self.train_single_epoch(fold_path, max_epochs)
            
            if train_result['status'] == 'completed':
                print("✓ All epochs completed!")
                return {
                    'status': 'completed',
                    'message': 'All epochs completed successfully',
                    'epochs_completed': max_epochs,
                    'results': results,
                    'final_summary': train_result['summary']
                }
            
            elif train_result['status'] == 'error':
                print(f"✗ Training failed: {train_result['message']}")
                return train_result
            
            elif train_result['status'] == 'training':
                # Monitor this epoch until completion
                monitor_result = self.monitor_and_wait(fold_path, timeout_per_epoch)
                results.append(monitor_result)
                
                if monitor_result['status'] != 'completed':
                    print(f"✗ Epoch failed: {monitor_result['message']}")
                    return monitor_result
                
                print(f"✓ Epoch {monitor_result['epoch']} completed")
                continue
            
            else:
                return {
                    'status': 'error',
                    'message': f'Unexpected training status: {train_result["status"]}',
                    'results': results
                }
    
    def _create_training_job(self,
                           train_file_id: str,
                           val_file_id: str,
                           model: str,
                           epoch_num: int,
                           fold_name: str,
                           from_checkpoint: str = None) -> str:
        """Create a TogetherAI fine-tuning job for one epoch."""
        suffix = f"lie-{fold_name}-epoch{epoch_num}-{int(time.time())}"
        
        # Calculate learning rate with 15% decay per epoch
        current_learning_rate = self.learning_rate * (0.85 ** epoch_num)
        
        job_params = {
            'training_file': train_file_id,
            'validation_file': val_file_id,
            'n_epochs': 1,  # Always train for exactly 1 epoch
            'learning_rate': current_learning_rate,
            'train_on_inputs': 'auto',  # Use auto for better compatibility
            'lora': True,
            'suffix': suffix,
            'n_evals': 2,  # Match your example (more evaluations)
            'n_checkpoints': 1,  # Match your example (more checkpoints)
            'warmup_ratio': 0,  # Match your example
            # 'lr_scheduler_type': 'linear',
            # 'lr_scheduler_args': {
            #     'min_lr_ratio': 0.66  # End at 66% of starting learning rate
            # },
            "lr_scheduler_type": "linear",
            "min_lr_ratio": 0.66
        }
        
        # Use from_checkpoint for continuing training or model for new training
        if from_checkpoint:
            job_params['from_checkpoint'] = from_checkpoint
        else:
            job_params['model'] = model
        
        # Add wandb logging if available
        if self.wandb_api_key:
            job_params['wandb_api_key'] = self.wandb_api_key
            # Create wandb project name with fold suffix
            wandb_project_with_fold = self.wandb_project_name+f"-{fold_name}"
            job_params['wandb_project_name'] = wandb_project_with_fold
        
        try:
            print(f"Creating training job with parameters:")
            if from_checkpoint:
                print(f"  From checkpoint: {from_checkpoint}")
            else:
                print(f"  Model: {model}")
            print(f"  Epochs: 1")
            print(f"  Learning Rate: {current_learning_rate} (base: {self.learning_rate}, decay: {0.67 ** epoch_num:.4f})")
            print(f"  LR Scheduler: linear (end at 66% of start rate)")
            print(f"  Suffix: {suffix}")
            print(f"  N Evals: {job_params['n_evals']}")
            print(f"  N Checkpoints: {job_params['n_checkpoints']}")
            if self.wandb_api_key:
                wandb_project_with_fold = self.wandb_project_name+f"-{fold_name}"
                print(f"  WANDB Project: {wandb_project_with_fold}")
                print(f"  WANDB Logging: Enabled")
            else:
                print(f"  WANDB Logging: Disabled")
            
            job_response = self.client.fine_tuning.create(**job_params)
            return job_response.id
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if "model" in error_msg.lower():
                raise Exception(f"Invalid model '{model}'. Check model name and availability.")
            elif "file" in error_msg.lower():
                raise Exception(f"File upload issue. Train: {train_file_id}, Val: {val_file_id}")
            elif "rate limit" in error_msg.lower():
                raise Exception("Rate limit exceeded. Please wait before starting new training.")
            else:
                raise Exception(f"Training job creation failed: {error_msg}")
    
    def get_fold_status(self, fold_path: str, sync_with_api: bool = True) -> Dict[str, Any]:
        """
        Get current status of training for a fold, optionally syncing with TogetherAI API.
        
        Args:
            fold_path: Path to the fold directory
            sync_with_api: Whether to fetch live status from TogetherAI API
            
        Returns:
            Dictionary with current training status
        """
        fold_path = Path(fold_path).resolve()
        training_state = TrainingState(str(fold_path), self.base_model)
        
        # Get base summary from local state
        summary = training_state.get_training_summary()
        
        if sync_with_api:
            print("Syncing with TogetherAI API...")
            try:
                # Update status for all tracked epochs
                updated_any = False
                
                for epoch, epoch_info in training_state.epochs.items():
                    if not epoch_info.job_id:
                        continue
                    
                    # Skip if already completed or failed
                    if epoch_info.status in ["completed", "failed", "cancelled"]:
                        continue
                    
                    try:
                        print(f"  Checking epoch {epoch} job: {epoch_info.job_id}")
                        
                        # Get live status from API
                        job_info = self.client.fine_tuning.retrieve(epoch_info.job_id)
                        raw_api_status = job_info.status
                        api_status = self._normalize_status(raw_api_status)
                        
                        print(f"    API status: {raw_api_status} → {api_status}")
                        
                        # Update local state if status changed
                        if api_status != epoch_info.status:
                            print(f"    Updating status: {epoch_info.status} → {api_status}")
                            
                            if api_status == "completed" and hasattr(job_info, 'output_name'):
                                model_id = job_info.output_name
                                training_state.update_epoch(epoch, model_id, api_status)
                                print(f"    Model ID: {model_id}")
                            else:
                                training_state.update_epoch(epoch, status=api_status)
                                
                                # Show failure details for failed jobs
                                if api_status == "failed":
                                    try:
                                        error_details = self._get_job_error_details(epoch_info.job_id)
                                        if error_details:
                                            print(f"    Error: {error_details}")
                                    except Exception:
                                        pass
                            
                            updated_any = True
                        
                        # Show recent events for running jobs
                        if api_status == "running":
                            try:
                                events = self.client.fine_tuning.list_events(epoch_info.job_id)
                                if events and hasattr(events, 'data') and events.data:
                                    recent_event = events.data[-1]
                                    if hasattr(recent_event, 'message'):
                                        print(f"    Latest: {recent_event.message}")
                            except Exception:
                                pass
                    
                    except Exception as e:
                        print(f"    Error checking job {epoch_info.job_id}: {e}")
                        # Don't update state on API errors - keep local state
                        continue
                
                if updated_any:
                    # Refresh summary after updates
                    summary = training_state.get_training_summary()
                    print("✓ Local state synchronized with API")
                else:
                    print("✓ Local state is up to date")
                    
            except Exception as e:
                print(f"Warning: API sync failed: {e}")
                print("Showing cached local state...")
        
        return {
            'fold_path': str(fold_path),
            'summary': summary,
            'next_epoch': training_state.get_current_epoch(),
            'completed_models': training_state.get_all_models(),
            'synced_with_api': sync_with_api
        }
    
    def cleanup_failed_training(self, fold_path: str) -> None:
        """Clean up failed training epochs to allow retry."""
        fold_path = Path(fold_path).resolve()
        training_state = TrainingState(str(fold_path), self.base_model)
        training_state.cleanup_failed_epochs()
        print(f"Cleaned up failed epochs for {fold_path.name}")
    
    def deploy_epoch_models(self, fold_path: str) -> Dict[str, Any]:
        """Get deployment-ready information for all completed epoch models."""
        fold_path = Path(fold_path).resolve()
        training_state = TrainingState(str(fold_path), self.base_model)
        
        models = training_state.get_all_models()
        
        deployment_info = {
            'fold_name': fold_path.name,
            'base_model': self.base_model,
            'total_epochs': len(models),
            'models': {}
        }
        
        for epoch, model_id in models.items():
            deployment_info['models'][f'epoch_{epoch}'] = {
                'model_id': model_id,
                'endpoint_name': f"{fold_path.name}-epoch-{epoch}",
                'description': f"Lie detection model for {fold_path.name} trained for {epoch + 1} epochs"
            }
        
        return deployment_info
    
    def abort_fold_training(self, fold_path: str) -> Dict[str, Any]:
        """
        Cancel all running/queued training jobs in a fold.
        
        Args:
            fold_path: Path to the fold directory
            
        Returns:
            Dictionary with cancellation results
        """
        fold_path = Path(fold_path).resolve()
        training_state = TrainingState(str(fold_path), self.base_model)
        
        results = {
            'fold_name': fold_path.name,
            'cancelled_jobs': [],
            'failed_cancellations': [],
            'no_action_needed': []
        }
        
        print(f"=== Aborting Training Jobs for {fold_path.name} ===")
        
        if not training_state.epochs:
            print("No training jobs found for this fold.")
            return results
        
        for epoch, epoch_info in training_state.epochs.items():
            if not epoch_info.job_id:
                results['no_action_needed'].append({
                    'epoch': epoch,
                    'reason': 'No job ID found'
                })
                continue
            
            # Skip already terminated jobs
            if epoch_info.status in ["completed", "failed", "cancelled"]:
                results['no_action_needed'].append({
                    'epoch': epoch,
                    'job_id': epoch_info.job_id,
                    'status': epoch_info.status,
                    'reason': f'Job already {epoch_info.status}'
                })
                print(f"  Epoch {epoch}: Already {epoch_info.status} (job: {epoch_info.job_id})")
                continue
            
            # Attempt to cancel running/pending jobs
            try:
                print(f"  Cancelling epoch {epoch} job: {epoch_info.job_id}")
                
                # Cancel the job through TogetherAI API
                cancel_response = self.client.fine_tuning.cancel(epoch_info.job_id)
                
                # Update local state
                training_state.update_epoch(epoch, status="cancelled")
                
                results['cancelled_jobs'].append({
                    'epoch': epoch,
                    'job_id': epoch_info.job_id,
                    'previous_status': epoch_info.status
                })
                
                print(f"    ✓ Successfully cancelled job {epoch_info.job_id}")
                
            except Exception as e:
                error_msg = str(e)
                results['failed_cancellations'].append({
                    'epoch': epoch,
                    'job_id': epoch_info.job_id,
                    'error': error_msg
                })
                
                print(f"    ✗ Failed to cancel job {epoch_info.job_id}: {error_msg}")
                
                # Don't update local state if cancellation failed
        
        # Print summary
        print(f"\n=== Cancellation Summary ===")
        print(f"Successfully cancelled: {len(results['cancelled_jobs'])} jobs")
        print(f"Failed to cancel: {len(results['failed_cancellations'])} jobs")
        print(f"No action needed: {len(results['no_action_needed'])} jobs")
        
        if results['cancelled_jobs']:
            print(f"\nCancelled jobs:")
            for job in results['cancelled_jobs']:
                print(f"  Epoch {job['epoch']}: {job['job_id']}")
        
        if results['failed_cancellations']:
            print(f"\nFailed cancellations:")
            for job in results['failed_cancellations']:
                print(f"  Epoch {job['epoch']}: {job['job_id']} - {job['error']}")
        
        return results