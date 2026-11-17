import numpy as np

from assembly_assistant.pipeline import virtual_projection


def test_warped_sprite_is_cached_for_static_projection(monkeypatch):
    virtual_projection._WARPED_SPRITE_CACHE.clear()
    calls = {"count": 0}

    def fake_warp_perspective(sprite, matrix, size, flags=None, borderMode=None):
        calls["count"] += 1
        w, h = size
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[:, :, :3] = sprite[0, 0, :3]
        out[:, :, 3] = sprite[0, 0, 3]
        return out

    monkeypatch.setattr(virtual_projection.cv2, "warpPerspective", fake_warp_perspective)
    sprite = np.zeros((8, 8, 4), dtype=np.uint8)
    sprite[:, :, :] = [10, 20, 30, 255]
    quad = np.array([[2, 2], [20, 2], [20, 20], [2, 20]], dtype=np.float32)
    frame_a = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_b = np.zeros((32, 32, 3), dtype=np.uint8)

    assert virtual_projection._warp_rgba_onto_frame(frame_a, sprite, quad, alpha=0.5)
    assert virtual_projection._warp_rgba_onto_frame(frame_b, sprite, quad, alpha=0.5)
    assert calls["count"] == 1


def test_corner_instance_composite_is_cached(monkeypatch):
    virtual_projection._WARPED_SPRITE_CACHE.clear()
    virtual_projection._COMPOSITE_SPRITE_CACHE.clear()
    calls = {"count": 0}

    def fake_warp_perspective(sprite, matrix, size, flags=None, borderMode=None):
        calls["count"] += 1
        w, h = size
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[:, :, :3] = sprite[0, 0, :3]
        out[:, :, 3] = sprite[0, 0, 3]
        return out

    monkeypatch.setattr(virtual_projection.cv2, "warpPerspective", fake_warp_perspective)
    sprite = np.zeros((8, 8, 4), dtype=np.uint8)
    sprite[:, :, :] = [10, 20, 30, 255]
    quads = [
        np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32),
        np.array([[20, 2], [26, 2], [26, 8], [20, 8]], dtype=np.float32),
        np.array([[20, 20], [26, 20], [26, 26], [20, 26]], dtype=np.float32),
        np.array([[2, 20], [8, 20], [8, 26], [2, 26]], dtype=np.float32),
    ]

    first = virtual_projection._cached_composite_sprite(sprite, quads, frame_size=(32, 32))
    second = virtual_projection._cached_composite_sprite(sprite, quads, frame_size=(32, 32))

    assert first is second
    assert calls["count"] == 4
