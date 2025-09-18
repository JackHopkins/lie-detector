#!/usr/bin/env python3
"""
Endpoint Manager for TogetherAI Evaluation

This module manages dedicated endpoints for model evaluation, including:
- Creating endpoints for trained models
- Tracking endpoint state in eval.json
- Managing endpoint lifecycle and cleanup
- Handling endpoint failures and retries
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class EndpointInfo:
    """Information about a deployed evaluation endpoint."""
    model_id: str
    endpoint_id: str
    endpoint_name: str
    display_name: str
    state: str
    hardware: str
    created_time: float
    last_used: float
    inactive_timeout: int
    auto_cleanup_after: float


class EvaluationState:
    """Manages evaluation state and endpoint metadata in eval.json files."""

    def __init__(self, fold_path: str, base_model: str):
        """
        Initialize evaluation state manager.

        Args:
            fold_path: Path to the fold directory
            base_model: Base model name for reference
        """
        self.fold_path = Path(fold_path).resolve()
        self.base_model = base_model
        self.eval_json_path = self.fold_path / "eval.json"
        self._load_state()

    def _load_state(self) -> None:
        """Load evaluation state from eval.json or initialize new state."""
        if self.eval_json_path.exists():
            try:
                with open(self.eval_json_path, 'r') as f:
                    self.state = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._initialize_state()
        else:
            self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize new evaluation state."""
        self.state = {
            "fold_path": str(self.fold_path),
            "base_model": self.base_model,
            "created_time": time.time(),
            "updated_time": time.time(),
            "endpoints": {}
        }
        self._save_state()

    def _save_state(self) -> None:
        """Save current state to eval.json."""
        self.state["updated_time"] = time.time()

        # Ensure directory exists
        self.eval_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = self.eval_json_path.with_suffix('.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.state, f, indent=2)
        temp_path.replace(self.eval_json_path)

    def get_endpoint_info(self, epoch: int) -> Optional[EndpointInfo]:
        """Get endpoint information for a specific epoch."""
        epoch_str = str(epoch)
        if epoch_str not in self.state["endpoints"]:
            return None

        endpoint_data = self.state["endpoints"][epoch_str]
        return EndpointInfo(
            model_id=endpoint_data["model_id"],
            endpoint_id=endpoint_data["endpoint_id"],
            endpoint_name=endpoint_data["endpoint_name"],
            display_name=endpoint_data["display_name"],
            state=endpoint_data["state"],
            hardware=endpoint_data["hardware"],
            created_time=endpoint_data["created_time"],
            last_used=endpoint_data["last_used"],
            inactive_timeout=endpoint_data["inactive_timeout"],
            auto_cleanup_after=endpoint_data["auto_cleanup_after"]
        )

    def add_endpoint(self, epoch: int, model_id: str, endpoint_id: str,
                    endpoint_name: str, display_name: str, state: str,
                    hardware: str, inactive_timeout: int = 15) -> None:
        """Add or update endpoint information."""
        epoch_str = str(epoch)
        current_time = time.time()
        cleanup_time = current_time + (inactive_timeout * 60)  # Convert minutes to seconds

        self.state["endpoints"][epoch_str] = {
            "model_id": model_id,
            "endpoint_id": endpoint_id,
            "endpoint_name": endpoint_name,
            "display_name": display_name,
            "state": state,
            "hardware": hardware,
            "created_time": current_time,
            "last_used": current_time,
            "inactive_timeout": inactive_timeout,
            "auto_cleanup_after": cleanup_time
        }
        self._save_state()

    def update_endpoint_state(self, epoch: int, state: str, last_used: Optional[float] = None) -> None:
        """Update endpoint state and usage time."""
        epoch_str = str(epoch)
        if epoch_str in self.state["endpoints"]:
            self.state["endpoints"][epoch_str]["state"] = state
            if last_used is not None:
                self.state["endpoints"][epoch_str]["last_used"] = last_used
            else:
                self.state["endpoints"][epoch_str]["last_used"] = time.time()
            self._save_state()

    def remove_endpoint(self, epoch: int) -> None:
        """Remove endpoint information."""
        epoch_str = str(epoch)
        if epoch_str in self.state["endpoints"]:
            del self.state["endpoints"][epoch_str]
            self._save_state()

    def get_all_endpoints(self) -> Dict[int, EndpointInfo]:
        """Get all endpoint information."""
        endpoints = {}
        for epoch_str, endpoint_data in self.state["endpoints"].items():
            epoch = int(epoch_str)
            endpoints[epoch] = EndpointInfo(
                model_id=endpoint_data["model_id"],
                endpoint_id=endpoint_data["endpoint_id"],
                endpoint_name=endpoint_data["endpoint_name"],
                display_name=endpoint_data["display_name"],
                state=endpoint_data["state"],
                hardware=endpoint_data["hardware"],
                created_time=endpoint_data["created_time"],
                last_used=endpoint_data["last_used"],
                inactive_timeout=endpoint_data["inactive_timeout"],
                auto_cleanup_after=endpoint_data["auto_cleanup_after"]
            )
        return endpoints

    def get_expired_endpoints(self) -> List[int]:
        """Get list of epochs with expired endpoints that should be cleaned up."""
        current_time = time.time()
        expired = []

        for epoch_str, endpoint_data in self.state["endpoints"].items():
            if endpoint_data["auto_cleanup_after"] < current_time:
                expired.append(int(epoch_str))

        return expired


# !/usr/bin/env python3
"""
Enhanced Endpoint Manager for TogetherAI Evaluation

This module manages dedicated endpoints for model evaluation, including:
- Creating endpoints for trained models
- Discovering existing endpoints
- Tracking endpoint state in eval.json
- Managing endpoint lifecycle and cleanup
"""

import json
import time
import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any

from together import Together


class EndpointManager:
    """Enhanced EndpointManager with endpoint creation capabilities."""

    def __init__(self, api_key: str, default_hardware: str = "4x_nvidia_h100_80gb_sxm"):
        """
        Initialize endpoint manager.

        Args:
            api_key: TogetherAI API key
            default_hardware: Default hardware configuration for endpoints
        """
        self.client = Together(api_key=api_key)
        self.api_key = api_key
        self.default_hardware = default_hardware
        self.api_base_url = "https://api.together.xyz/v1"

    def find_or_create_endpoint_for_epoch(
            self,
            fold_path: str,
            epoch: int,
            fold_name: str,
            inactive_timeout: int = 360,
            create_if_missing: bool = True
    ) -> Optional[str]:
        """
        Find an existing endpoint or create one for a specific epoch.

        Args:
            fold_path: Path to the fold directory
            epoch: Epoch number
            fold_name: Name of the fold (e.g., 'mask-factual')
            inactive_timeout: Minutes before endpoint auto-stops (default 6 hours)
            create_if_missing: Whether to create endpoint if not found

        Returns:
            Endpoint name if found/created, None otherwise
        """
        from .training_state import TrainingState
        from .epoch_eval import EvaluationState

        # Get model ID from training state
        training_state = TrainingState(fold_path, "")
        completed_models = training_state.get_all_models()

        if epoch not in completed_models:
            print(f"No completed model found for epoch {epoch}")
            return None

        model_id = completed_models[epoch]
        print(f"Looking for endpoint for epoch {epoch}, model: {model_id}")

        # Check eval.json for cached endpoint
        eval_state = EvaluationState(fold_path, model_id.split('/')[0])
        endpoint_info = eval_state.get_endpoint_info(epoch)

        if endpoint_info:
            # Check if endpoint is still active
            status = self._check_endpoint_status(endpoint_info.endpoint_id)
            if status == "STARTED":
                print(f"Found active cached endpoint: {endpoint_info.endpoint_name}")
                eval_state.update_endpoint_state(epoch, "STARTED")
                return endpoint_info.endpoint_name
            elif status in ["PENDING", "STARTING"]:
                print(f"Endpoint is starting: {endpoint_info.endpoint_name}")
                return endpoint_info.endpoint_name

        # Try to discover existing endpoint
        discovered = self._discover_model_endpoint(model_id)
        if discovered:
            print(f"Found existing endpoint: {discovered['name']}")
            eval_state.add_endpoint(
                epoch=epoch,
                model_id=model_id,
                endpoint_id=discovered['id'],
                endpoint_name=discovered['name'],
                display_name=discovered.get('display_name', discovered['name']),
                state=discovered['state'],
                hardware=self.default_hardware,
                inactive_timeout=inactive_timeout // 60  # Convert to minutes for storage
            )
            return discovered['name']

        # Create new endpoint if requested
        if create_if_missing:
            print(f"No existing endpoint found. Creating new endpoint for epoch {epoch}...")
            created_endpoint = self._create_endpoint(
                model_id=model_id,
                display_name=f"{fold_name}-epoch{epoch}",
                inactive_timeout=inactive_timeout
            )

            if created_endpoint:
                eval_state.add_endpoint(
                    epoch=epoch,
                    model_id=model_id,
                    endpoint_id=created_endpoint['id'],
                    endpoint_name=created_endpoint['name'],
                    display_name=created_endpoint['display_name'],
                    state=created_endpoint['state'],
                    hardware=self.default_hardware,
                    inactive_timeout=inactive_timeout // 60
                )

                # Wait for endpoint to be ready
                if created_endpoint['state'] != "STARTED":
                    print("Waiting for endpoint to start...")
                    self._wait_for_endpoint(created_endpoint['id'], max_wait=600)

                return created_endpoint['name']

        return None

    def find_models_for_fold(self, fold_name: str, max_epochs: int = 10) -> Dict[int, str]:
        """
        Find all fine-tuned models for a specific fold.

        Args:
            fold_name: Name of the fold (e.g., 'mask-factual')
            max_epochs: Maximum number of epochs to search for

        Returns:
            Dictionary mapping epoch numbers to model IDs
        """
        print(f"Searching for models containing '{fold_name}'...")

        models_response = self.client.models.list()

        if hasattr(models_response, 'data'):
            models = models_response.data
        else:
            models = models_response

        found_models = {}

        for model in models:
            model_id = model.id if hasattr(model, 'id') else str(model)

            # Check if this model matches our fold
            if fold_name in model_id:
                # Extract epoch number from model ID
                for epoch in range(max_epochs):
                    epoch_pattern = f"-epoch{epoch}-"
                    if epoch_pattern in model_id:
                        found_models[epoch] = model_id
                        print(f"  Found epoch {epoch}: {model_id}")
                        break

        return found_models

    def _create_endpoint(
            self,
            model_id: str,
            display_name: str,
            inactive_timeout: int = 360,
            min_replicas: int = 1,
            max_replicas: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new endpoint using the Together API.

        Args:
            model_id: ID of the model to deploy
            display_name: Human-readable name for the endpoint
            inactive_timeout: Minutes before endpoint auto-stops
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas

        Returns:
            Endpoint info dict if created successfully, None otherwise
        """
        endpoint_data = {
            "model": model_id,
            "display_name": display_name,
            "hardware": self.default_hardware,
            "inactive_timeout": inactive_timeout,
            "autoscaling": {
                "min_replicas": min_replicas,
                "max_replicas": max_replicas
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/endpoints",
                json=endpoint_data,
                headers=headers
            )

            if response.status_code == 200 or response.status_code == 201:
                endpoint = response.json()
                print(f"Successfully created endpoint: {endpoint.get('name', endpoint.get('id'))}")

                return {
                    'id': endpoint.get('id'),
                    'name': endpoint.get('name', endpoint.get('id')),
                    'display_name': display_name,
                    'state': endpoint.get('state', 'PENDING'),
                    'model': model_id
                }
            else:
                print(f"Failed to create endpoint: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"Error creating endpoint: {e}")
            return None

    def _wait_for_endpoint(self, endpoint_id: str, max_wait: int = 600, check_interval: int = 10):
        """
        Wait for an endpoint to become active.

        Args:
            endpoint_id: ID of the endpoint to wait for
            max_wait: Maximum seconds to wait
            check_interval: Seconds between status checks
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self._check_endpoint_status(endpoint_id)

            if status == "STARTED":
                print(f"Endpoint {endpoint_id} is now active!")
                return True
            elif status in ["ERROR", "FAILED"]:
                print(f"Endpoint {endpoint_id} failed to start (status: {status})")
                return False
            else:
                print(f"Endpoint status: {status}, waiting...")
                time.sleep(check_interval)

        print(f"Timeout waiting for endpoint {endpoint_id} to start")
        return False

    def _check_endpoint_status(self, endpoint_id: str) -> str:
        """Check the current status of an endpoint."""
        try:
            endpoint = self.client.endpoints.get(endpoint_id)
            return endpoint.state if hasattr(endpoint, 'state') else "UNKNOWN"
        except Exception as e:
            if "not found" in str(e).lower():
                return "NOT_FOUND"
            return "ERROR"

    def _discover_model_endpoint(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Discover endpoint for a specific model by listing all endpoints.

        Args:
            model_id: Model ID to find endpoint for

        Returns:
            Endpoint info dict if found and active, None otherwise
        """
        try:
            endpoints_response = self.client.endpoints.list()

            if hasattr(endpoints_response, 'data'):
                endpoints = endpoints_response.data
            else:
                endpoints = endpoints_response

            for endpoint in endpoints:
                if (hasattr(endpoint, 'model') and endpoint.model == model_id and
                        hasattr(endpoint, 'state') and endpoint.state in ["STARTED", "PENDING", "STARTING"]):
                    return {
                        'id': endpoint.id,
                        'name': endpoint.name,
                        'model': endpoint.model,
                        'state': endpoint.state,
                        'display_name': getattr(endpoint, 'display_name', endpoint.name)
                    }

            return None

        except Exception as e:
            print(f"Error listing endpoints: {e}")
            return None

    def create_all_endpoints_for_fold(
            self,
            fold_path: str,
            fold_name: str,
            max_epochs: int = 10,
            inactive_timeout: int = 360
    ) -> Dict[str, Any]:
        """
        Create endpoints for all epochs of a fold that don't have one.

        Args:
            fold_path: Path to the fold directory
            fold_name: Name of the fold
            max_epochs: Maximum epochs to check
            inactive_timeout: Minutes before endpoints auto-stop

        Returns:
            Summary of created endpoints
        """
        from .training_state import TrainingState
        from .epoch_eval import EvaluationState

        print(f"\n{'=' * 60}")
        print(f"Creating endpoints for {fold_name}")
        print(f"{'=' * 60}")

        # Get completed models
        training_state = TrainingState(fold_path, "")
        completed_models = training_state.get_all_models()

        if not completed_models:
            # Try to find models by searching
            print("No models in training.json, searching for models...")
            found_models = self.find_models_for_fold(fold_name, max_epochs)
            if found_models:
                completed_models = found_models
            else:
                print("No models found")
                return {"created": 0, "skipped": 0, "failed": 0}

        summary = {
            "created": 0,
            "skipped": 0,
            "failed": 0,
            "endpoints": {}
        }

        for epoch, model_id in sorted(completed_models.items()):
            print(f"\nEpoch {epoch}: {model_id}")

            # Check if endpoint already exists
            discovered = self._discover_model_endpoint(model_id)
            if discovered:
                print(f"  → Endpoint already exists: {discovered['name']}")
                summary["skipped"] += 1
                summary["endpoints"][epoch] = {
                    "status": "exists",
                    "endpoint_name": discovered['name']
                }
                continue

            # Create endpoint
            print(f"  → Creating endpoint...")
            created = self._create_endpoint(
                model_id=model_id,
                display_name=f"{fold_name}-epoch{epoch}",
                inactive_timeout=inactive_timeout
            )

            if created:
                print(f"  → Created: {created['name']}")
                summary["created"] += 1
                summary["endpoints"][epoch] = {
                    "status": "created",
                    "endpoint_name": created['name'],
                    "endpoint_id": created['id']
                }

                # Save to eval.json
                eval_state = EvaluationState(fold_path, model_id.split('/')[0])
                eval_state.add_endpoint(
                    epoch=epoch,
                    model_id=model_id,
                    endpoint_id=created['id'],
                    endpoint_name=created['name'],
                    display_name=created['display_name'],
                    state=created['state'],
                    hardware=self.default_hardware,
                    inactive_timeout=inactive_timeout // 60
                )
            else:
                print(f"  → Failed to create endpoint")
                summary["failed"] += 1
                summary["endpoints"][epoch] = {
                    "status": "failed",
                    "model_id": model_id
                }

        print(f"\n{'=' * 60}")
        print(f"Summary: Created {summary['created']}, Skipped {summary['skipped']}, Failed {summary['failed']}")
        print(f"{'=' * 60}")

        return summary

    def get_or_find_endpoint(self, fold_path: str, epoch: int, model_id: str,
                             fold_name: str) -> Optional[str]:
        """
        Backwards compatible method that tries to find or create an endpoint.

        This method maintains compatibility with the existing code while
        adding the ability to create endpoints if they don't exist.
        """
        return self.find_or_create_endpoint_for_epoch(
            fold_path=fold_path,
            epoch=epoch,
            fold_name=fold_name,
            create_if_missing=True,
            inactive_timeout=360  # 6 hours default
        )


# Utility function for standalone endpoint creation
def create_endpoints_for_evaluation(
        fold_path: str,
        fold_name: str,
        api_key: Optional[str] = None,
        inactive_timeout: int = 360
) -> Dict[str, Any]:
    """
    Standalone function to create all endpoints needed for evaluation.

    Args:
        fold_path: Path to the fold directory
        fold_name: Name of the fold (e.g., 'mask-factual')
        api_key: Together API key (uses env var if not provided)
        inactive_timeout: Minutes before endpoints auto-stop (default 6 hours)

    Returns:
        Summary of created endpoints
    """
    if not api_key:
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")

    manager = EndpointManager(api_key)
    return manager.create_all_endpoints_for_fold(
        fold_path=fold_path,
        fold_name=fold_name,
        inactive_timeout=inactive_timeout
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage Together AI endpoints")
    parser.add_argument("--fold-path", required=True, help="Path to fold directory")
    parser.add_argument("--fold-name", required=True, help="Name of the fold")
    parser.add_argument("--timeout", type=int, default=360,
                        help="Inactive timeout in minutes (default: 360)")
    parser.add_argument("--create-all", action="store_true",
                        help="Create endpoints for all epochs")

    args = parser.parse_args()

    if args.create_all:
        summary = create_endpoints_for_evaluation(
            fold_path=args.fold_path,
            fold_name=args.fold_name,
            inactive_timeout=args.timeout
        )
        print(json.dumps(summary, indent=2))