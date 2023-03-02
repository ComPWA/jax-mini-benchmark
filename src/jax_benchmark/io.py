import logging
import os


def enable_x64() -> None:
    from jax.config import config

    config.update("jax_enable_x64", True)


def mute_warnings() -> None:
    logging.getLogger("jax").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
