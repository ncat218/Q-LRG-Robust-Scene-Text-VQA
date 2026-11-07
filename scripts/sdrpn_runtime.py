"""Environment shared by the SD-RPN import preflight and selector worker."""

import os


def selector_environment(settings):
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    libexpat = settings.get("sdrpn_libexpat")
    if libexpat:
        prior = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = f"{libexpat}:{prior}" if prior else libexpat
    return env
