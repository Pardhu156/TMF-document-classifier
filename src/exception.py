"""Custom exceptions that retain useful source-location context."""

from __future__ import annotations

import sys
from types import TracebackType


class CustomException(Exception):
    """Wrap an exception with the filename and line where it occurred."""

    def __init__(
        self,
        error_message: object,
        error_detail: tuple[object, object, TracebackType | None] | None = None,
    ) -> None:
        exception_type, exception_value, traceback = error_detail or sys.exc_info()

        if traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename
            line_no = traceback.tb_lineno
        else:
            filename = "<unknown>"
            line_no = "<unknown>"

        original_message = exception_value if exception_value is not None else error_message
        self.error_message = (
            f"Error occurred in Python script [{filename}]\n"
            f"line number [{line_no}]\n"
            f"error message [{original_message}]"
        )
        super().__init__(self.error_message)
