import io
import os

import pytest
from conftest import FakePredictor, image_bytes, make_settings

from model_api.app import create_api
from webapp import create_app
from webapp.validation import UploadRejected, clean_filename, validate_upload


@pytest.fixture
def client_and_predictor(tmp_path):
    predictor = FakePredictor()
    app = create_app(make_settings(tmp_path), predictor)
    return app.test_client(), predictor


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_valid_upload_redirects_to_result_and_is_logged(client_and_predictor, jpeg):
    client, _ = client_and_predictor
    response = upload(client, jpeg, "photo.jpg")
    assert response.status_code == 303
    page = client.get(response.headers["Location"])
    assert b"AI-generated" in page.data and b"Do not treat this verdict as proof" in page.data
    history = client.get("/history")
    assert b"photo.jpg" in history.data and b"VisionGuard:test" in history.data and b"91.0%" in history.data


def test_wrong_type_rejected(client_and_predictor):
    client, predictor = client_and_predictor
    response = upload(client, b"plain text", "notes.txt")
    assert response.status_code == 415 and b"Unsupported file type" in response.data
    assert predictor.calls == 0


def test_undecodable_image_rejected(client_and_predictor):
    client, predictor = client_and_predictor
    response = upload(client, b"not really a jpeg", "fake.jpg")
    assert response.status_code == 422 and b"could not read" in response.data
    assert predictor.calls == 0


def test_truncated_image_rejected(client_and_predictor):
    client, _ = client_and_predictor
    whole = image_bytes(size=(600, 600))
    assert upload(client, whole[: len(whole) // 2], "cut.jpg").status_code == 422


def test_oversized_upload_rejected_with_friendly_page(client_and_predictor):
    client, predictor = client_and_predictor
    response = upload(client, b"x" * (2 * 1024 * 1024), "huge.jpg")
    assert response.status_code == 413 and b"too large" in response.data
    assert predictor.calls == 0


def test_rejections_are_logged(client_and_predictor):
    client, _ = client_and_predictor
    upload(client, b"plain text", "notes.txt")
    page = client.get("/history?status=rejected")
    assert b"wrong_type" in page.data and b"notes.txt" in page.data


def test_inference_failure_shows_friendly_page_and_logs(tmp_path, jpeg):
    client = create_app(make_settings(tmp_path), FakePredictor(fail=True)).test_client()
    response = upload(client, jpeg, "a.jpg")
    assert response.status_code == 502 and b"unavailable" in response.data
    assert b"inference_failed" in client.get("/history?status=error").data


def test_filename_is_sanitized(client_and_predictor, jpeg):
    client, _ = client_and_predictor
    assert upload(client, jpeg, "../../etc/evil name.jpg").status_code == 303
    history = client.get("/history").data
    assert b"etc_evil_name.jpg" in history and b"../" not in history
    assert clean_filename("../../") == "upload"


def test_upload_is_not_written_to_disk(tmp_path, jpeg, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app(make_settings(tmp_path), FakePredictor()).test_client()
    upload(client, jpeg, "photo.jpg")
    assert sorted(os.listdir(tmp_path)) == ["test.db"]


def test_health_reports_database_and_debug_off(client_and_predictor):
    client, _ = client_and_predictor
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["debug"] is False


def test_magic_bytes_beat_extension():
    with pytest.raises(UploadRejected) as caught:
        validate_upload("gif_in_disguise.png", image_bytes("GIF"), 10**6, 10**7)
    assert caught.value.code == "wrong_type"


@pytest.fixture
def api(tmp_path):
    return create_api(make_settings(tmp_path), FakePredictor(), api_key="secret").test_client()


def test_api_requires_key(api, jpeg):
    response = api.post("/v1/predict", data={"image": (io.BytesIO(jpeg), "a.jpg")})
    assert response.status_code == 401


def test_api_returns_verdict(api, jpeg):
    response = api.post("/v1/predict", data={"image": (io.BytesIO(jpeg), "a.jpg")}, headers={"X-API-Key": "secret"})
    body = response.get_json()
    assert response.status_code == 200 and body["label"] == "ai_generated" and body["model_version"] == "VisionGuard:test"


def test_api_rejects_bad_image_as_json(api):
    response = api.post("/v1/predict", data={"image": (io.BytesIO(b"junk"), "a.jpg")}, headers={"X-API-Key": "secret"})
    assert response.status_code == 422 and response.get_json()["error"] == "undecodable"


def test_api_health_is_public(api):
    assert api.get("/health").get_json()["status"] == "ok"
