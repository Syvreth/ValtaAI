from __future__ import annotations

import time


class JobStopped(Exception):
    """The running job was stopped from the control center."""


class JobControl:
    def __init__(self) -> None:
        self.stop = False
        self.paused = False

    def checkpoint(self) -> None:
        while self.paused and not self.stop:
            time.sleep(0.1)
        if self.stop:
            raise JobStopped()

    def request_pause(self) -> None:
        self.paused = True

    def request_resume(self) -> None:
        self.paused = False

    def request_stop(self) -> None:
        self.stop = True
        self.paused = False
