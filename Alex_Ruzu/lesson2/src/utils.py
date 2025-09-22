"""
Utility functions for the cluster-driven pipeline.

Includes error handling, retry logic, and LLM interaction utilities.
"""

import json
import logging
import time
from typing import List, Any, Dict, Optional, Callable
from functools import wraps
from pathlib import Path
from datetime import datetime
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class LLMError(PipelineError):
    """Exception for LLM-related errors."""
    pass


class ClusteringError(PipelineError):
    """Exception for clustering-related errors."""
    pass


class ValidationError(PipelineError):
    """Exception for data validation errors."""
    pass


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (APIError, RateLimitError, APIConnectionError)
):
    """Decorator for retrying functions on specific exceptions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
                except Exception as e:
                    # Don't retry for non-specified exceptions
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            raise LLMError(f"Failed after {max_retries + 1} attempts: {last_exception}")

        return wrapper
    return decorator


class LLMClient:
    """Wrapper for OpenAI client with retry logic and error handling."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @retry_on_exception(max_retries=3, delay=2.0)
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a chat completion request with retry logic."""
        try:
            logger.debug(f"Making OpenAI request with {len(messages)} messages")

            response: ChatCompletion = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 1500)
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("OpenAI returned empty response")

            logger.debug(f"OpenAI response received (length: {len(content)})")
            return content.strip()

        except APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            raise LLMError(f"OpenAI API Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI request: {e}")
            raise LLMError(f"Unexpected error: {e}")

    def ask(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Simple prompt-based interaction."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat_completion(messages, **kwargs)


def safe_json_parse(text: str, context: str = "") -> Dict[str, Any]:
    """Safely parse JSON with helpful error messages."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON array or object
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        error_msg = f"Failed to parse JSON{' for ' + context if context else ''}: {e}"
        logger.error(f"{error_msg}. Text preview: {text[:200]}...")
        raise ValidationError(error_msg)


def validate_required_fields(data: Dict[str, Any], required_fields: List[str], context: str = "") -> None:
    """Validate that required fields are present in data."""
    missing_fields = [field for field in required_fields if field not in data or not data[field]]

    if missing_fields:
        error_msg = f"Missing required fields{' in ' + context if context else ''}: {missing_fields}"
        logger.error(error_msg)
        raise ValidationError(error_msg)


def safe_file_write(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """Safely write content to file with error handling."""
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)

        logger.info(f"Successfully wrote file: {file_path}")

    except IOError as e:
        error_msg = f"Failed to write file {file_path}: {e}"
        logger.error(error_msg)
        raise PipelineError(error_msg)


def safe_file_read(file_path: str, encoding: str = 'utf-8') -> str:
    """Safely read content from file with error handling."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        logger.debug(f"Successfully read file: {file_path} (length: {len(content)})")
        return content

    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise PipelineError(error_msg)
    except IOError as e:
        error_msg = f"Failed to read file {file_path}: {e}"
        logger.error(error_msg)
        raise PipelineError(error_msg)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ArtifactManager:
    """Manages timestamped artifacts for pipeline runs"""

    def __init__(self, base_dir: str = "docs"):
        """Initialize with timestamped session directory"""
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / "runs" / self.timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Also maintain latest symlink for convenience
        self.latest_dir = self.base_dir / "latest"
        if self.latest_dir.is_symlink() or self.latest_dir.exists():
            if self.latest_dir.is_symlink():
                self.latest_dir.unlink()
            elif self.latest_dir.is_dir():
                import shutil
                shutil.rmtree(self.latest_dir)

        # Create symlink to latest run (Windows compatible)
        try:
            self.latest_dir.symlink_to(f"runs/{self.timestamp}", target_is_directory=True)
        except OSError:
            # Fallback for systems without symlink support (common on Windows)
            logger.debug("Could not create latest symlink - symlinks not supported. This is expected on Windows without developer mode.")

    def save_artifact(self, name: str, content: Any, format: str = "json") -> Path:
        """Save artifact to session directory"""
        if format == "json":
            file_path = self.session_dir / f"{name}.json"
            content_str = json.dumps(content, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            safe_file_write(str(file_path), content_str)
        elif format == "md":
            file_path = self.session_dir / f"{name}.md"
            safe_file_write(str(file_path), str(content))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved artifact: {file_path}")
        return file_path

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about current session"""
        return {
            "session_id": self.timestamp,
            "session_dir": str(self.session_dir),
            "base_dir": str(self.base_dir),
            "created_at": datetime.now().isoformat()
        }

    def list_previous_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List previous pipeline runs"""
        runs_dir = self.base_dir / "runs"
        if not runs_dir.exists():
            return []

        runs = []
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name != self.timestamp:
                try:
                    # Parse timestamp from directory name
                    timestamp = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")

                    # Count artifacts
                    artifacts = list(run_dir.glob("*.json")) + list(run_dir.glob("*.md"))

                    runs.append({
                        "session_id": run_dir.name,
                        "created_at": timestamp.isoformat(),
                        "artifacts_count": len(artifacts),
                        "path": str(run_dir)
                    })

                    if len(runs) >= limit:
                        break

                except ValueError:
                    # Skip directories that don't match timestamp format
                    continue

        return runs


# Global variable to track if logging has been initialized
_logging_initialized = False

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration with proper separation of concerns."""
    global _logging_initialized

    root_logger = logging.getLogger()

    # Only initialize once - subsequent calls just add file handlers
    if not _logging_initialized:
        # Clear any existing handlers to start fresh
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatters
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler - only user-facing messages
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)

        # Custom filter for console - only essential user progress
        class ConsoleFilter(logging.Filter):
            def filter(self, record):
                # Only show essential user-facing progress, block all technical details
                essential_patterns = [
                    'Configuration validated successfully',
                    'Starting Complete Pipeline Execution',
                    'Phase 1:', 'Phase 2:', 'Phase 3:', 'Phase 4:', 'Phase 5:',
                    'PIPELINE COMPLETED SUCCESSFULLY',
                    'Virtual board simulation completed successfully!',
                    'SUCCESS',
                    'FAILED'
                ]

                # Block all clustering, technical, and detailed messages
                blocked_patterns = [
                    'Analyzing Cluster', 'Generated', 'Personas:', 'features',
                    'Starting feature formulation', 'Pain points identified',
                    'Saving', 'artifacts', 'session:', 'Start time:', 'End time:',
                    'Total duration:', 'Session ID:', 'CSV path:', 'Configuration:',
                    'Recent previous runs:', 'STEP', 'Loading', 'Creating',
                    'Finding', 'Performing', 'Cluster distribution', 'TF-IDF',
                    'Silhouette Score', 'Starting round', 'Initializing',
                    'Loaded', 'Generated Files:', 'Architecture Benefits:',
                    'Starting virtual board simulation', 'Structured simulation completed'
                ]

                message = record.getMessage()

                # First check if it should be blocked
                if any(pattern in message for pattern in blocked_patterns):
                    return False

                # Then check if it's an essential message
                return any(pattern in message for pattern in essential_patterns)

        console_handler.addFilter(ConsoleFilter())

        # Configure root logger
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)

        # Suppress noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        _logging_initialized = True

    # Add file handler if specified and not already present
    if log_file:
        try:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Check if handler for this file already exists
            existing_handler = None
            for handler in root_logger.handlers:
                if (isinstance(handler, logging.FileHandler) and
                    hasattr(handler, 'baseFilename') and
                    handler.baseFilename.endswith(os.path.basename(log_file))):
                    existing_handler = handler
                    break

            if not existing_handler:
                detailed_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

                file_handler = logging.FileHandler(log_file, 'w', encoding='utf-8')
                file_handler.setFormatter(detailed_formatter)
                file_handler.setLevel(level)

                # Apply different filters based on log file purpose
                if 'main.log' in log_file:
                    # Main log: Application orchestration and user-facing information
                    class MainLogFilter(logging.Filter):
                        def filter(self, record):
                            message = record.getMessage()

                            # Include main application flow and results
                            include_patterns = [
                                'Configuration validated successfully',
                                'Starting Complete Pipeline Execution',
                                'Phase 1:', 'Phase 2:', 'Phase 3:', 'Phase 4:', 'Phase 5:',
                                'STARTING ENHANCED CLUSTER-DRIVEN PIPELINE',
                                'PIPELINE COMPLETED SUCCESSFULLY',
                                'Virtual board simulation completed successfully',
                                'SUCCESS', 'FAILED',
                                'Total Execution Time:',
                                'Generated Files:',
                                'Session ID:', 'Start time:', 'End time:',
                                'STEP 1:', 'STEP 2:', 'STEP 3:', 'STEP 4:',
                                '   - docs/', '   -'  # Include file list items
                            ]

                            # Exclude all technical implementation details
                            exclude_patterns = [
                                'Pain points identified from clustering',
                                'Starting feature formulation',
                                'Successfully generated and validated',
                                'Saving 5 personas and 3 features',
                                'Saved artifact:', 'Successfully wrote file:',
                                'Artifact manager initialized',
                                'TF-IDF', 'Silhouette Score', 'Cluster distribution',
                                'Analyzing Cluster', 'Optimal K:', 'Creating TF-IDF',
                                'Loading data from', 'Final dataset:',
                                'Starting round', 'Initializing structured simulation',
                                'Structured simulation completed',
                                'Architecture Benefits:'  # Remove architecture benefits from main.log
                            ]

                            # First check excludes
                            if any(pattern in message for pattern in exclude_patterns):
                                return False

                            # Then check includes
                            return any(pattern in message for pattern in include_patterns)

                    file_handler.addFilter(MainLogFilter())

                elif 'pipeline.log' in log_file:
                    # Pipeline log: Technical details and processing steps
                    class PipelineLogFilter(logging.Filter):
                        def filter(self, record):
                            message = record.getMessage()

                            # Include all technical implementation details
                            include_patterns = [
                                'Pain points identified from clustering',
                                'Starting feature formulation',
                                'Successfully generated and validated',
                                'Saving 5 personas and 3 features',
                                'Saved artifact:', 'Successfully wrote file:',
                                'Artifact manager initialized',
                                'Analyzing Cluster', 'Generated', 'Personas:',
                                'TF-IDF', 'Silhouette Score', 'Cluster distribution',
                                'Optimal K:', 'Creating TF-IDF', 'Loading data from',
                                'Final dataset:', 'Starting round',
                                'Initializing structured simulation',
                                'Structured simulation completed',
                                'Configuration loaded and validated successfully'
                            ]

                            # Exclude high-level user flow messages
                            exclude_patterns = [
                                'STARTING ENHANCED CLUSTER-DRIVEN PIPELINE',
                                'PIPELINE COMPLETED SUCCESSFULLY',
                                'Configuration validated successfully',
                                'Starting Complete Pipeline Execution',
                                'Phase 1:', 'Phase 2:', 'Phase 3:', 'Phase 4:', 'Phase 5:',
                                'Total Execution Time:',
                                'Architecture Benefits:',
                                'Generated Files:'
                            ]

                            # First check excludes
                            if any(pattern in message for pattern in exclude_patterns):
                                return False

                            # Then check includes
                            return any(pattern in message for pattern in include_patterns)

                    file_handler.addFilter(PipelineLogFilter())

                root_logger.addHandler(file_handler)

        except Exception as e:
            print(f"Failed to set up file logging: {e}")

    return logging.getLogger(__name__)