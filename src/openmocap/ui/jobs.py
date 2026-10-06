"""Cancelable Qt workers for the application service API."""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class PipelineJob(QThread):
    """Keep processing outside the GUI thread and cancellation cooperative."""

    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    elapsed = Signal(float)

    def __init__(self, task: Callable[..., Any], parent: Any = None) -> None:
        super().__init__(parent)
        self.task = task
        self.cancel_event = threading.Event()
        self.started_at = 0.0

    def run(self) -> None:
        self.started_at = time.monotonic()
        try:
            result = self.task(self.progress.emit, self.cancel_event)
            self.succeeded.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.elapsed.emit(time.monotonic() - self.started_at)

    def cancel(self) -> None:
        self.cancel_event.set()
