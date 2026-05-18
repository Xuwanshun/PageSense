"""
Conversation history endpoints.

A conversation is a named session — it groups all the questions a user
asked and the answers they received. Conversations persist in PostgreSQL
so they survive container restarts, deploys, and new logins.

Endpoints:
  GET  /conversations                    list all conversations (newest first)
  GET  /conversations/{id}/messages      full message history for one conversation
  DELETE /conversations/{id}             delete a conversation and all its messages
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import delete, select

from api.dependencies import get_current_user
from db.models import conversations, messages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    request: Request,
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """
    Return all conversations for the logged-in user, newest first.

    Each item includes a preview of the last message so the sidebar
    can show something useful without loading all messages.
    """
    engine = request.app.state.db_engine

    with engine.connect() as conn:
        rows = conn.execute(
            select(conversations)
            .where(conversations.c.user_id == user["id"])
            .order_by(conversations.c.created_at.desc())
        ).fetchall()

        result = []
        for row in rows:
            # Fetch just the last message for the preview
            last = conn.execute(
                select(messages)
                .where(messages.c.conversation_id == row.id)
                .order_by(messages.c.created_at.desc())
                .limit(1)
            ).first()

            result.append(
                {
                    "id": row.id,
                    "title": row.title or "Untitled",
                    "created_at": row.created_at.isoformat(),
                    "last_message": last.content[:120] if last else None,
                    "last_role": last.role if last else None,
                }
            )

    return JSONResponse({"conversations": result})


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """
    Return the full message history for one conversation.

    Only returns the conversation if it belongs to the current user —
    prevents users from reading each other's chat histories.
    """
    engine = request.app.state.db_engine

    with engine.connect() as conn:
        convo = conn.execute(
            select(conversations).where(
                conversations.c.id == conversation_id,
                conversations.c.user_id == user["id"],
            )
        ).first()

        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        rows = conn.execute(
            select(messages).where(messages.c.conversation_id == conversation_id).order_by(messages.c.created_at.asc())
        ).fetchall()

    return JSONResponse(
        {
            "id": convo.id,
            "title": convo.title or "Untitled",
            "created_at": convo.created_at.isoformat(),
            "messages": [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "sources": row.sources,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ],
        }
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """Delete a conversation and all its messages."""
    engine = request.app.state.db_engine

    with engine.begin() as conn:
        convo = conn.execute(
            select(conversations).where(
                conversations.c.id == conversation_id,
                conversations.c.user_id == user["id"],
            )
        ).first()

        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conn.execute(delete(messages).where(messages.c.conversation_id == conversation_id))
        conn.execute(delete(conversations).where(conversations.c.id == conversation_id))

    logger.info("Deleted conversation %s for user %s", conversation_id, user["id"])
    return JSONResponse({"deleted": conversation_id})
