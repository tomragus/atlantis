"""GET /test-images (filename list) and the /test-image-files static mount
that serves the sample images the frontend's picker grid offers.

Runs against the real test-images/ directory (no fake state needed) since
these routes just list/serve files off disk.
"""


def test_list_test_images_returns_sorted_png_filenames(client):
    resp = client.get("/test-images")
    assert resp.status_code == 200
    names = resp.json()
    assert names == sorted(names)
    assert all(name.endswith(".png") for name in names)
    assert len(names) > 0


def test_test_image_file_is_servable(client):
    names = client.get("/test-images").json()
    resp = client.get(f"/test-image-files/{names[0]}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
