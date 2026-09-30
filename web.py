"""Headless loopback dashboard: python web.py (no Tkinter display required)."""

import argparse
import logging

from config import Config
from visionguard.core.service import MonitorService
from visionguard.interfaces.web.server import LocalWebServer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=Config.WEB_PORT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    service = MonitorService(Config())
    try:
        service.start()
        server = LocalWebServer(service, args.port)
        print(
            f"Open {server.url} in a browser on this computer. Ctrl+C to stop.",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()


if __name__ == "__main__":
    main()
