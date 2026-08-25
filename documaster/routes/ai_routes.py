"""Flask routes for optional DocuMaster AI preview/API use.

SLAIHub should use the PyQt GUI directly. These routes reuse the same service
layer and keep explicit upload limits.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.document_ai_service import DocumentAIService
from ..services.document_extractor import DEFAULT_MAX_FILE_SIZE_BYTES


def create_ai_blueprint(service: DocumentAIService | None = None) -> Blueprint:
    ai_bp = Blueprint("documaster_ai", __name__, url_prefix="/api/ai")
    ai_service = service or DocumentAIService()

    def _file_or_error():
        if "file" not in request.files:
            return None, (jsonify({"status": "error", "error": "No file uploaded"}), 400)
        return request.files["file"], None

    def _json_response(payload):
        status_code = 200 if payload.get("status") in {"success", "partial"} else 400
        return jsonify(payload), status_code

    @ai_bp.get("/health")
    def ai_health():
        return jsonify(ai_service.health())

    @ai_bp.post("/runtime/initialize")
    def initialize_runtime():
        return jsonify(ai_service.initialize_runtime())

    @ai_bp.post("/analyze")
    def analyze():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="analyze", file=file, options=request.form.to_dict()))

    @ai_bp.post("/summarize")
    def summarize():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="summarize", file=file, options=request.form.to_dict()))

    @ai_bp.post("/explain")
    def explain():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="explain", file=file, options=request.form.to_dict()))

    @ai_bp.post("/ask")
    def ask():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(
            ai_service.run_upload(task="ask", file=file, question=request.form.get("question", ""), options=request.form.to_dict())
        )

    @ai_bp.post("/rewrite-suggestions")
    def rewrite_suggestions():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="rewrite_suggestions", file=file, options=request.form.to_dict()))

    @ai_bp.post("/quality-check")
    def quality_check():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="quality_check", file=file, options=request.form.to_dict()))

    @ai_bp.post("/key-points")
    def key_points():
        file, error = _file_or_error()
        if error:
            return error
        return _json_response(ai_service.run_upload(task="key_points", file=file, options=request.form.to_dict()))

    @ai_bp.app_errorhandler(413)
    def payload_too_large(_error):
        limit_mb = DEFAULT_MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return jsonify({"status": "error", "error": f"File is too large. Maximum upload size is {limit_mb:.0f} MB."}), 413

    return ai_bp
