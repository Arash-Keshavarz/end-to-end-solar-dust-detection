import base64


def test_predict_missing_image_returns_400():
    from app import create_app

    class FakePipeline:
        def __init__(self, filename, model_path=None):
            self.filename = filename
            self.model_path = model_path

        def predict(self):
            return [{"image": "Clean"}]

    app = create_app(pipeline_cls=FakePipeline)
    client = app.test_client()

    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_happy_path_returns_label(monkeypatch, tmp_path):
    # Ensure the server writes into a temp file location
    monkeypatch.setenv("PREDICT_FILENAME", str(tmp_path / "in.jpg"))

    # Avoid loading real torch weights by swapping the PredictionPipeline class
    class FakePipeline:
        def __init__(self, filename, model_path=None):
            self.filename = filename
            self.model_path = model_path

        def predict(self):
            return [{"image": "Clean"}]

    from app import create_app

    flask_app = create_app(pipeline_cls=FakePipeline)
    client = flask_app.test_client()

    image_bytes = b"not a real jpeg, just bytes"
    payload = {"image": base64.b64encode(image_bytes).decode("utf-8")}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    assert resp.get_json() == [{"image": "Clean"}]
