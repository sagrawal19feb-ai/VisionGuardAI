"""Desktop fullscreen is a view of the existing camera, not a second capture."""

import unittest
from types import SimpleNamespace
from unittest import mock

from visionguard.interfaces.desktop import MainWindow


class FullscreenTests(unittest.TestCase):
    def test_open_and_close_without_starting_another_camera(self):
        view = MainWindow.__new__(MainWindow)
        view.root = mock.Mock()
        view._fullscreen_window = None
        view._fullscreen_video = None
        view.fullscreen_button = mock.Mock()
        view.service = SimpleNamespace(
            camera=SimpleNamespace(is_available=False, start=mock.Mock()),
            worker=SimpleNamespace(
                state=SimpleNamespace(read=lambda: SimpleNamespace(result=None))
            ),
        )
        view.label = mock.Mock(return_value=mock.Mock())
        view.button = mock.Mock(return_value=mock.Mock())
        window = mock.Mock()
        with (
            mock.patch(
                "visionguard.interfaces.desktop.tk.Toplevel", return_value=window
            ),
            mock.patch("visionguard.interfaces.desktop.tk.Frame"),
            mock.patch("visionguard.interfaces.desktop.tk.Label"),
        ):
            view.toggle_fullscreen()
        self.assertIs(view._fullscreen_window, window)
        window.attributes.assert_called_once_with("-fullscreen", True)
        self.assertTrue(window.bind.called)
        view.service.camera.start.assert_not_called()
        view.exit_fullscreen()
        self.assertIsNone(view._fullscreen_window)
        window.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
