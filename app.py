import base64
import binascii
import os
from typing import Optional

from flask import Flask, jsonify, render_template, request
try:
    from flask_cors import CORS
except ModuleNotFoundError:  # pragma: no cover
    CORS = None  # type: ignore[misc,assignment]

from solar_dust_detection import logger
# Define environment variables for UTF-8 output
os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")


def _parse_cors_origins(value: str) -> Optional[list[str]]:
    value = (value or "").strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def create_app(pipeline_cls=None) -> Flask:
    flask_app = Flask(__name__)

    # CORS: set CORS_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"
    cors_origins = _parse_cors_origins(os.getenv("CORS_ORIGINS", ""))
    if CORS is not None:
        if cors_origins:
            CORS(flask_app, resources={r"/*": {"origins": cors_origins}})
        else:
            # Default for local dev; lock down in production with CORS_ORIGINS
            CORS(flask_app)
    else:
        logger.warning("flask-cors is not installed; continuing without CORS support.")

    predict_filename = os.getenv("PREDICT_FILENAME", "inputImage.jpg")
    max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))  # 5MB default

    # MODEL_PATH can point to artifacts/training/model.pt or model/model.pt
    if pipeline_cls is None:
        # Import lazily so unit tests can run without torch installed.
        from solar_dust_detection.pipeline.prediction import PredictionPipeline as pipeline_cls

    classifier = pipeline_cls(filename=predict_filename, model_path=os.getenv("MODEL_PATH"))

    @flask_app.get("/")
    def home():
        return render_template("index.html")

    @flask_app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @flask_app.post("/predict")
    def predict_route():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image")
        if not isinstance(image_b64, str) or not image_b64.strip():
            return jsonify({"error": "Missing required field 'image' (base64 string)."}), 400

        # Accept either raw base64 or data URL form.
        if "," in image_b64 and image_b64.strip().lower().startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_b64, validate=False)
        except (binascii.Error, ValueError):
            return jsonify({"error": "Invalid base64 in field 'image'."}), 400

        if len(image_bytes) > max_image_bytes:
            return (
                jsonify({"error": f"Image too large. Max is {max_image_bytes} bytes."}),
                413,
            )

        try:
            with open(predict_filename, "wb") as f:
                f.write(image_bytes)
            result = classifier.predict()
            return jsonify(result)
        except Exception as e:
            logger.exception("Prediction failed", exc_info=e)
            return jsonify({"error": "Prediction failed. Check server logs."}), 500

    return flask_app


if __name__ == "__main__":
    app = create_app()
    # Use 0.0.0.0 for Docker/AWS, or 127.0.0.1 for local
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))