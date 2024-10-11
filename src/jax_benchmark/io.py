import logging
import os
import socket
from importlib.metadata import version

import cpuinfo


def enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def get_machine_info() -> dict:
    info = {
        "host_name": socket.gethostname(),
        "scheduled_cpus": sorted(os.sched_getaffinity(0)),
        "jax_version": version("jax"),
    }
    info.update(cpuinfo.get_cpu_info())
    return info


def mute_warnings() -> None:
    logging.getLogger("jax").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
