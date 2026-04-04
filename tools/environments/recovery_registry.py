"""Static recovery registry for backend-owned background checkpoints."""

from __future__ import annotations

from tools.environments.daytona import DaytonaEnvironment
from tools.environments.docker import DockerEnvironment
from tools.environments.local import LocalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment
from tools.environments.modal import ModalEnvironment
from tools.environments.singularity import SingularityEnvironment
from tools.environments.ssh import SSHEnvironment
from tools.environments.vercel_sandbox import VercelSandboxEnvironment


ENVIRONMENT_CLASS_BY_BACKGROUND_BACKEND = {
    "local": LocalEnvironment,
    "docker": DockerEnvironment,
    "singularity": SingularityEnvironment,
    "ssh": SSHEnvironment,
    "modal": ModalEnvironment,
    "managed_modal": ManagedModalEnvironment,
    "daytona": DaytonaEnvironment,
    "vercel_sandbox": VercelSandboxEnvironment,
}


def get_environment_class_for_background_backend(backend: str):
    """Return the environment class that owns checkpoint parsing/recovery."""
    return ENVIRONMENT_CLASS_BY_BACKGROUND_BACKEND.get(backend)
