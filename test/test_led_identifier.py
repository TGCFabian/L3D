import sys

sys.path.append("./")
from lib.led_identifier import LedFinder
from mock_camera import MockCamera


def close(x, y):
    return abs(x - y) < 1.0


def test_init():
    LedFinder()


def test_basic_image_loading():
    led_finder = LedFinder()

    mock_camera = MockCamera()

    led_results = led_finder.find_led(mock_camera.read())
    u, v = led_results.get_center()
    assert close(u, 257)
    assert close(v, 177)


def test_none_found():

    led_finder = LedFinder()

    mock_camera = MockCamera()

    for frame_id in [7, 15, 23]:  # None of these should be visible from any views
        frame = mock_camera.read_frame(frame_id)
        led_results = led_finder.find_led(frame)
        assert led_results is None
