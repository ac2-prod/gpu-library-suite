"""Scheduler identity validation and filesystem-token derivation."""

import re
from typing import Optional, Tuple


class SchedulerIdentityError(ValueError):
    pass


NQSV_JOB_ID_RE = re.compile(
    r"^(?:[0-9]+:)?([0-9]+)(?:\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)?$"
)
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,121}$")


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def parse_pbs_job_token(raw_job_id: str) -> str:
    """Return the numeric NQSV request token without changing the raw ID."""

    if (
        not isinstance(raw_job_id, str)
        or raw_job_id == ""
        or _contains_control(raw_job_id)
    ):
        raise SchedulerIdentityError("invalid NQSV scheduler job ID")
    match = NQSV_JOB_ID_RE.fullmatch(raw_job_id)
    if match is None:
        raise SchedulerIdentityError("invalid NQSV scheduler job ID")
    token = match.group(1)
    if not token.isascii() or not token.isdigit() or len(token) > 122:
        raise SchedulerIdentityError("invalid NQSV scheduler job token")
    return token


def validate_scheduler_identity(
    scheduler: Optional[str], scheduler_job_id: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Validate a nullable scheduler/raw-job-ID pair without sanitizing it."""

    if scheduler is None and scheduler_job_id is None:
        return None, None
    if scheduler is None or scheduler_job_id is None:
        raise SchedulerIdentityError(
            "scheduler and scheduler job ID must both be null or both be strings"
        )
    if (
        not isinstance(scheduler, str)
        or scheduler == ""
        or "/" in scheduler
        or "\\" in scheduler
        or _contains_control(scheduler)
    ):
        raise SchedulerIdentityError("invalid scheduler name")
    if (
        not isinstance(scheduler_job_id, str)
        or scheduler_job_id == ""
        or "/" in scheduler_job_id
        or "\\" in scheduler_job_id
        or _contains_control(scheduler_job_id)
    ):
        raise SchedulerIdentityError("invalid scheduler job ID")
    if scheduler == "NQSV":
        parse_pbs_job_token(scheduler_job_id)
    return scheduler, scheduler_job_id


def scheduler_job_path_token(
    scheduler: Optional[str], scheduler_job_id: Optional[str],
) -> Optional[str]:
    """Derive a safe component only for filesystem and log paths."""

    validate_scheduler_identity(scheduler, scheduler_job_id)
    if scheduler is None:
        return None
    if scheduler == "NQSV":
        return parse_pbs_job_token(scheduler_job_id)
    if (
        SAFE_COMPONENT_RE.fullmatch(scheduler_job_id) is None
        or scheduler_job_id.startswith(".")
        or ".." in scheduler_job_id
    ):
        raise SchedulerIdentityError(
            "scheduler job ID has no safe filesystem token"
        )
    return scheduler_job_id
