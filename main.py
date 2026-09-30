"""Desktop monitor and optional private browser dashboard. Run: python main.py"""

import argparse
import logging
import tkinter as tk

from config import Config
from visionguard.core.service import MonitorService
from visionguard.interfaces.desktop import MainWindow
from visionguard.interfaces.web.server import LocalWebServer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=Config.WEB_PORT)
    parser.add_argument("--no-web", action="store_true", help="Desktop UI only")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    root = tk.Tk()
    service = None
    server = None
    window = None
    try:
        service = MonitorService(Config())
        if not args.no_web:
            try:
                server = LocalWebServer(service, args.port)
                server.start()
            except OSError:
                logging.exception("Local dashboard port unavailable; desktop continues")
                server = None
        window = MainWindow(
            root,
            service,
            web_url=server.url if server else None,
            stop_web=server.close if server else None,
        )
        service.start()  # A missing webcam never prevents the UI from opening.
        root.mainloop()
    finally:
        if window is not None:
            window.close()
        else:
            if server:
                server.close()
            if service:
                service.close()
            root.destroy()


if __name__ == "__main__":
    main()
