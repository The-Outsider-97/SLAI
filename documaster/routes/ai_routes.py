"""Flask routes for DocMaster AI services.

These routes are optional for local/cloud preview. The SLAIHub PyQt GUI calls the
same DocumentAIService directly and does not need to expose uploaded documents to
a browser frontend.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.document_ai_service import DocumentAIService


def create_ai_blueprint(service: DocumentAIService | None = None) -> Blueprint:
    ai_bp = Blueprint("docmaster_ai", __name__, url_prefix="/api/ai")
    ai_service = service or DocumentAIService()

    def _file_or_error():
        if "file" not in request.files:
            return None, (jsonify({"status": "error", "error": "No file uploaded"}), 400)
        return request.files["file"], None

    def _json_response(payload):
        status_code = 200 if payload.get("status") == "success" else 400
        return jsonify(payload), status_code

    @ai_bp.get("/health")
    def ai_health():
        return jsonify(ai_service.health())

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

    return ai_bp
