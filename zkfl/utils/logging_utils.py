"""
Logging utilities for ZKFL framework.

This module provides enhanced logging functionality with support for
structured logging, performance tracking, and federated learning context.
"""

import logging
import time
import json
import os
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from pathlib import Path

from ..core.exceptions import ConfigurationError


class LoggingUtils:
    """Enhanced logging utilities for federated learning."""

    _loggers: Dict[str, logging.Logger] = {}
    _performance_stack: List[Dict[str, Any]] = []

    @classmethod
    def setup_logger(
        cls,
        name: str,
        level: Union[str, int] = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        enable_structured: bool = False,
    ) -> logging.Logger:
        """
        Set up a logger with enhanced configuration.

        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for logging
            format_string: Custom format string
            enable_structured: Enable structured JSON logging

        Returns:
            Configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Default format
        if format_string is None:
            if enable_structured:
                format_string = "%(message)s"
            else:
                format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add structured logging capability
        if enable_structured:
            logger = cls._wrap_structured_logger(logger)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def _wrap_structured_logger(cls, logger: logging.Logger) -> logging.Logger:
        """Wrap logger to support structured logging."""
        original_log = logger._log

        def structured_log(level, msg, args, **kwargs):
            if isinstance(msg, dict):
                # Convert dict to JSON string
                msg = json.dumps(msg, default=str, indent=2)
            original_log(level, msg, args, **kwargs)

        logger._log = structured_log
        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get existing logger or create default one."""
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]

    @classmethod
    def log_federated_round(
        cls,
        logger: logging.Logger,
        round_num: int,
        num_clients: int,
        metrics: Dict[str, Any],
        duration: Optional[float] = None,
    ):
        """
        Log federated learning round information.

        Args:
            logger: Logger instance
            round_num: Current round number
            num_clients: Number of participating clients
            metrics: Round metrics
            duration: Round duration in seconds
        """
        log_data = {
            "event": "federated_round",
            "round": round_num,
            "num_clients": num_clients,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        if duration is not None:
            log_data["duration_seconds"] = duration

        logger.info(log_data)

    @classmethod
    def log_client_update(
        cls,
        logger: logging.Logger,
        client_id: str,
        round_num: int,
        local_epochs: int,
        local_metrics: Dict[str, Any],
        privacy_budget: Optional[float] = None,
    ):
        """
        Log client update information.

        Args:
            logger: Logger instance
            client_id: Client identifier
            round_num: Current round number
            local_epochs: Number of local training epochs
            local_metrics: Client's local training metrics
            privacy_budget: Remaining privacy budget
        """
        log_data = {
            "event": "client_update",
            "client_id": client_id,
            "round": round_num,
            "local_epochs": local_epochs,
            "timestamp": datetime.now().isoformat(),
            "local_metrics": local_metrics,
        }

        if privacy_budget is not None:
            log_data["privacy_budget"] = privacy_budget

        logger.info(log_data)

    @classmethod
    def log_aggregation(
        cls,
        logger: logging.Logger,
        round_num: int,
        aggregation_method: str,
        num_updates: int,
        aggregated_metrics: Dict[str, Any],
        validation_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model aggregation information.

        Args:
            logger: Logger instance
            round_num: Current round number
            aggregation_method: Aggregation algorithm used
            num_updates: Number of client updates aggregated
            aggregated_metrics: Metrics after aggregation
            validation_metrics: Optional validation metrics
        """
        log_data = {
            "event": "model_aggregation",
            "round": round_num,
            "aggregation_method": aggregation_method,
            "num_updates": num_updates,
            "timestamp": datetime.now().isoformat(),
            "aggregated_metrics": aggregated_metrics,
        }

        if validation_metrics:
            log_data["validation_metrics"] = validation_metrics

        logger.info(log_data)

    @classmethod
    def log_privacy_analysis(
        cls,
        logger: logging.Logger,
        client_id: str,
        epsilon: float,
        delta: float,
        mechanism: str,
        noise_scale: float,
    ):
        """
        Log differential privacy analysis.

        Args:
            logger: Logger instance
            client_id: Client identifier
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta
            mechanism: Privacy mechanism used
            noise_scale: Scale of added noise
        """
        log_data = {
            "event": "privacy_analysis",
            "client_id": client_id,
            "epsilon": epsilon,
            "delta": delta,
            "mechanism": mechanism,
            "noise_scale": noise_scale,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(log_data)

    @classmethod
    def log_security_event(
        cls,
        logger: logging.Logger,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        client_id: Optional[str] = None,
    ):
        """
        Log security-related events.

        Args:
            logger: Logger instance
            event_type: Type of security event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details
            client_id: Optional client identifier
        """
        log_data = {
            "event": "security_event",
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }

        if client_id:
            log_data["client_id"] = client_id

        # Use appropriate log level based on severity
        level_map = {
            "LOW": logging.INFO,
            "MEDIUM": logging.WARNING,
            "HIGH": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level_map.get(severity, logging.WARNING)
        logger.log(level, log_data)

    @classmethod
    def start_performance_timer(
        cls, operation_name: str, context: Optional[Dict[str, Any]] = None
    ):
        """
        Start performance timing for an operation.

        Args:
            operation_name: Name of the operation
            context: Additional context information
        """
        timer_data = {
            "operation": operation_name,
            "start_time": time.time(),
            "context": context or {},
        }
        cls._performance_stack.append(timer_data)

    @classmethod
    def end_performance_timer(cls, logger: logging.Logger) -> Optional[float]:
        """
        End performance timing and log the result.

        Args:
            logger: Logger instance

        Returns:
            Duration in seconds, or None if no timer was active
        """
        if not cls._performance_stack:
            logger.warning("No active performance timer to end")
            return None

        timer_data = cls._performance_stack.pop()
        end_time = time.time()
        duration = end_time - timer_data["start_time"]

        log_data = {
            "event": "performance_timing",
            "operation": timer_data["operation"],
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "context": timer_data["context"],
        }

        logger.info(log_data)
        return duration

    @classmethod
    def log_exception(
        cls,
        logger: logging.Logger,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True,
    ):
        """
        Log exception with context.

        Args:
            logger: Logger instance
            exception: Exception that occurred
            context: Additional context information
            include_traceback: Whether to include full traceback
        """
        import traceback

        log_data = {
            "event": "exception",
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }

        if include_traceback:
            log_data["traceback"] = traceback.format_exc()

        logger.error(log_data)

    @classmethod
    def create_audit_logger(cls, audit_file: str) -> logging.Logger:
        """
        Create dedicated audit logger for compliance tracking.

        Args:
            audit_file: Path to audit log file

        Returns:
            Audit logger instance
        """
        audit_logger = cls.setup_logger(
            name="zkfl_audit",
            level=logging.INFO,
            log_file=audit_file,
            format_string="%(asctime)s - AUDIT - %(message)s",
            enable_structured=True,
        )

        return audit_logger

    @classmethod
    def log_audit_event(
        cls,
        audit_logger: logging.Logger,
        event_type: str,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log audit event for compliance.

        Args:
            audit_logger: Audit logger instance
            event_type: Type of audit event
            actor: Who performed the action
            action: What action was performed
            resource: What resource was affected
            outcome: Outcome of the action
            details: Additional details
        """
        audit_data = {
            "audit_event": event_type,
            "timestamp": datetime.now().isoformat(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "details": details or {},
        }

        audit_logger.info(audit_data)

    @classmethod
    def setup_federated_logging(
        cls, base_dir: str, client_id: Optional[str] = None, enable_audit: bool = True
    ) -> Dict[str, logging.Logger]:
        """
        Set up comprehensive logging for federated learning.

        Args:
            base_dir: Base directory for log files
            client_id: Client identifier (for client-side logging)
            enable_audit: Whether to enable audit logging

        Returns:
            Dictionary of configured loggers
        """
        log_dir = Path(base_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        loggers = {}

        # Main application logger
        app_log_file = log_dir / f"zkfl{'_' + client_id if client_id else ''}.log"
        loggers["main"] = cls.setup_logger(
            name="zkfl",
            level=logging.INFO,
            log_file=str(app_log_file),
            enable_structured=True,
        )

        # Performance logger
        perf_log_file = (
            log_dir / f"performance{'_' + client_id if client_id else ''}.log"
        )
        loggers["performance"] = cls.setup_logger(
            name="zkfl_performance",
            level=logging.INFO,
            log_file=str(perf_log_file),
            enable_structured=True,
        )

        # Security logger
        security_log_file = (
            log_dir / f"security{'_' + client_id if client_id else ''}.log"
        )
        loggers["security"] = cls.setup_logger(
            name="zkfl_security",
            level=logging.WARNING,
            log_file=str(security_log_file),
            enable_structured=True,
        )

        # Audit logger (if enabled)
        if enable_audit:
            audit_log_file = (
                log_dir / f"audit{'_' + client_id if client_id else ''}.log"
            )
            loggers["audit"] = cls.create_audit_logger(str(audit_log_file))

        return loggers

    @classmethod
    def cleanup_old_logs(cls, log_dir: str, days_to_keep: int = 30):
        """
        Clean up old log files.

        Args:
            log_dir: Directory containing log files
            days_to_keep: Number of days worth of logs to retain
        """
        import glob
        from datetime import timedelta

        log_path = Path(log_dir)
        if not log_path.exists():
            return

        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        for log_file in log_path.glob("*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Error deleting log file {log_file}: {e}")

    @classmethod
    def configure_log_rotation(
        cls, log_file: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
    ):
        """
        Configure log rotation for a log file.

        Args:
            log_file: Path to log file
            max_bytes: Maximum size before rotation
            backup_count: Number of backup files to keep
        """
        from logging.handlers import RotatingFileHandler

        # This would be used when setting up file handlers
        # Implementation depends on the specific logging setup
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )

        return handler
