"""Authenticated endpoints; the shared server additionally enforces cookie, Host, Origin and CSRF marker."""
from __future__ import annotations

from typing import Any

from ..tools.errors import BadRequest
from ..tools.server import Request, Response, Router, created
from .vault import Vault

router = Router()


def _token(req: Request) -> str:
    return req.headers.get("x-vault-session", "")


@router.route("GET", "/api/status")
def status(app: Vault, req: Request) -> dict[str, Any]:
    return app.status()


@router.route("POST", "/api/setup")
def setup(app: Vault, req: Request) -> dict[str, Any]:
    data = req.json()
    return app.setup(data.get("password"), data.get("confirmation"))


@router.route("POST", "/api/unlock")
def unlock(app: Vault, req: Request) -> dict[str, Any]:
    return app.unlock(req.json().get("password"))


@router.route("POST", "/api/lock")
def lock(app: Vault, req: Request) -> dict[str, bool]:
    return app.lock(_token(req))


@router.route("POST", "/api/reauth")
def reauth(app: Vault, req: Request) -> dict[str, bool]:
    return app.reauthenticate(_token(req), req.json().get("password"))


@router.route("GET", "/api/items")
def items(app: Vault, req: Request) -> dict[str, Any]:
    return app.snapshot(_token(req))


@router.route("POST", "/api/folders")
def folders(app: Vault, req: Request) -> Any:
    data = req.json()
    return created(app.folder(_token(req), data.get("name"), data.get("parent")))


@router.route("POST", "/api/uploads")
def upload_begin(app: Vault, req: Request) -> Any:
    data = req.json()
    return created(app.start_upload(_token(req), data.get("name"), data.get("parent"),
                                    data.get("mime", "application/octet-stream"), data.get("size")))


@router.route("POST", "/api/uploads/{id}/chunks")
def upload_chunk(app: Vault, req: Request) -> dict[str, int]:
    data = req.json()
    return app.upload_chunk(_token(req), req.params["id"], data.get("index"), data.get("data"))


@router.route("POST", "/api/uploads/{id}/finish")
def upload_finish(app: Vault, req: Request) -> Any:
    return created(app.finish_upload(_token(req), req.params["id"]))


@router.route("POST", "/api/uploads/{id}/cancel")
def upload_cancel(app: Vault, req: Request) -> dict[str, bool]:
    return app.cancel_upload(_token(req), req.params["id"])


@router.route("GET", "/api/files/{id}/chunks/{index}")
def file_chunk(app: Vault, req: Request) -> Response:
    try:
        index = int(req.params["index"])
    except ValueError as exc:
        raise BadRequest("Invalid chunk index") from exc
    body = app.read_chunk(_token(req), req.params["id"], index)
    return Response(200, body, "application/octet-stream", {"Cache-Control": "no-store"})


@router.route("POST", "/api/notes")
def note_create(app: Vault, req: Request) -> Any:
    data = req.json()
    return created(app.create_note(_token(req), data.get("name"), data.get("content", ""), data.get("parent")))


@router.route("GET", "/api/notes/{id}")
def note_read(app: Vault, req: Request) -> Any:
    return app.note_text(_token(req), req.params["id"])


@router.route("PUT", "/api/notes/{id}")
def note_edit(app: Vault, req: Request) -> Any:
    return app.update_note(_token(req), req.params["id"], req.json().get("content"))


@router.route("PATCH", "/api/items/{id}")
def item_edit(app: Vault, req: Request) -> Any:
    return app.edit(_token(req), req.params["id"], req.json())


@router.route("DELETE", "/api/items/{id}")
def item_trash(app: Vault, req: Request) -> Any:
    return app.trash(_token(req), req.params["id"])


@router.route("POST", "/api/items/{id}/restore")
def item_restore(app: Vault, req: Request) -> Any:
    return app.trash(_token(req), req.params["id"], restore=True)


@router.route("POST", "/api/items/{id}/purge")
def item_purge(app: Vault, req: Request) -> Any:
    return app.purge(_token(req), req.params["id"], req.json().get("password"))


@router.route("POST", "/api/password/change")
def password_change(app: Vault, req: Request) -> Any:
    data = req.json()
    return app.change_password(_token(req), data.get("oldPassword"), data.get("newPassword"))
