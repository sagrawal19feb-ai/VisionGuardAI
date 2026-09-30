"""One bounded, thread-safe publication point for every dashboard consumer.

The worker owns inference. The desktop and local web server only read its latest
result; neither consumes a queue that can steal another interface's frames.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    result: object | None
    jpeg: bytes | None
    updated_at: float


class MonitorState:
    def __init__(self):
        self._condition = threading.Condition()
        self._sequence = 0
        self._result = None
        self._jpeg = None
        self._updated_at = 0.0
        self._event_number = 0
        self._events = deque(maxlen=200)

    def publish(self, result, incident=None):
        # Encode once, off both UI threads. Do not publish an unmatched image
        # and assessment if JPEG encoding fails.
        ok, image = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        with self._condition:
            self._result = result
            self._jpeg = image.tobytes() if ok else None
            self._updated_at = time.monotonic()
            self._sequence += 1
            if incident is not None:
                self._event_number += 1
                self._events.append((self._event_number, incident))
            self._condition.notify_all()

    def clear(self):
        """Discard frames from a previous camera session before restart."""
        with self._condition:
            self._result = None
            self._jpeg = None
            self._updated_at = 0.0
            self._sequence += 1
            self._condition.notify_all()

    def read(self):
        with self._condition:
            return Snapshot(self._sequence, self._result, self._jpeg, self._updated_at)

    def wait_after(self, sequence, timeout=5):
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > sequence, timeout=timeout)
            return Snapshot(self._sequence, self._result, self._jpeg, self._updated_at)

    def events_after(self, event_number):
        with self._condition:
            return [(n, event) for n, event in self._events if n > event_number]
