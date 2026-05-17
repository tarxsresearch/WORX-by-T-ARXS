"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                     WORCX \u2014 Production Backend v2.2                            \u2551
\u2551        End-to-End Encrypted Social Chat with Realtime Reactions                \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

CHANGELOG v2.2 (complete implementation):
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
  ADD  \u2014 POST /messages                       send DM
  ADD  \u2014 GET  /messages/conversation/{uid}    fetch DM history (paginated)
  ADD  \u2014 PATCH /messages/{id}                 edit message
  ADD  \u2014 DELETE /messages/{id}                soft-delete message
  ADD  \u2014 POST /groups                         create group
  ADD  \u2014 GET  /groups                         list current user's groups
  ADD  \u2014 GET  /groups/{id}                    group detail + members
  ADD  \u2014 PATCH /groups/{id}                   edit group name/avatar
  ADD  \u2014 DELETE /groups/{id}                  delete group (owner only)
  ADD  \u2014 POST /groups/{id}/members            add member
  ADD  \u2014 DELETE /groups/{id}/members/{uid}    remove member
  ADD  \u2014 POST /groups/{id}/messages           send group message
  ADD  \u2014 GET  /groups/{id}/messages           fetch group history (paginated)
  ADD  \u2014 POST /users/{id}/follow              send follow request
  ADD  \u2014 DELETE /users/{id}/follow            unfollow
  ADD  \u2014 PATCH /follows/{id}                  approve / reject follow
  ADD  \u2014 GET  /users/{id}/followers           list followers
  ADD  \u2014 GET  /users/{id}/following           list following
  ADD  \u2014 GET  /users/search                   search users by username
  ADD  \u2014 GET  /users/{id}                     public user profile
  ADD  \u2014 PATCH /auth/me                       update own profile
  ADD  \u2014 POST /rooms                          create ephemeral room
  ADD  \u2014 GET  /rooms/{token}                  room detail
  ADD  \u2014 POST /rooms/{token}/join             join room
  ADD  \u2014 POST /rooms/{token}/messages         send room message
  ADD  \u2014 GET  /rooms/{token}/messages         room history
  FIX  \u2014 SlowAPI Request param properly wired (was request=None)
  FIX  \u2014 Request imported from fastapi
  ADD  \u2014 requirements.txt generated alongside this file
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501

QUICK START:
  pip install -r requirements.txt
  export SESSION_SECRET="your-stable-secret-here"
  python main.py
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
"""

import os
import json
import secrets
import asyncio
import sqlite3
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Set

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect,
    Header, status, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey,
    Boolean, UniqueConstraint, Index, or_, and_,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import bcrypt
from jose import jwt, JWTError

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  LOGGING & CONFIG
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("SESSION_SECRET")
if not SECRET_KEY:
    logger.critical(
        "SESSION_SECRET env var is not set. "
        "Set it to a stable secret or all tokens will be invalidated on restart.\n"
        "Example: export SESSION_SECRET=$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
    )
    sys.exit(1)

E2EE_BACKEND_URL  = os.environ.get("E2EE_BACKEND", "https://e2ee-api.worcx.io")
ALGORITHM         = "HS256"
TOKEN_TTL_HOURS   = 24 * 7
DB_PATH           = os.environ.get("CHAT_DB", "chat.db")
MAX_GROUP_MEMBERS = 20
MAX_EMOJI_LEN     = 32
MAX_MSG_LEN       = 64_000   # ciphertext byte cap
PORT              = int(os.environ.get("PORT", "5000"))
ALLOWED_KINDS     = {"text", "image", "video", "audio", "file"}
ALLOWED_ORIGINS   = [
    "http://localhost:3000",
    "http://localhost:5173",
    os.environ.get("FRONTEND_URL", "https://worcx.io"),
]
RATE_LIMIT_AUTH   = 100
RATE_LIMIT_MSG    = 50
PAGE_SIZE         = 50

_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  DATABASE MODELS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id               = Column(Integer, primary_key=True)
    username         = Column(String(64), unique=True, nullable=False, index=True)
    password_hash    = Column(String(256), nullable=False)
    public_key       = Column(Text, nullable=True)
    avatar           = Column(Text, nullable=True)
    display_name     = Column(String(120), nullable=True)
    location         = Column(String(120), nullable=True)
    is_private       = Column(Boolean, default=False)
    invite_token     = Column(String(64), unique=True, nullable=True, index=True)
    e2ee_api_key     = Column(String(128), unique=True, nullable=True, index=True)
    e2ee_key_backup  = Column(Text, nullable=True)
    e2ee_key_expires = Column(DateTime, nullable=True)
    e2ee_last_rotate = Column(DateTime, nullable=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Group(Base):
    __tablename__ = "groups"
    id         = Column(Integer, primary_key=True)
    name       = Column(String(120), nullable=False)
    avatar     = Column(Text, nullable=True)
    owner_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    members    = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")
    messages   = relationship("Message", primaryjoin="Group.id == foreign(Message.group_id)", lazy="dynamic")


class GroupMember(Base):
    __tablename__ = "group_members"
    id        = Column(Integer, primary_key=True)
    group_id  = Column(Integer, ForeignKey("groups.id"), nullable=False, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    group     = relationship("Group", back_populates="members")
    __table_args__ = (
        UniqueConstraint("group_id", "user_id", name="uniq_group_member"),
        Index("ix_group_user", "group_id", "user_id"),
    )


class Follow(Base):
    __tablename__ = "follows"
    id          = Column(Integer, primary_key=True)
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    followee_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status      = Column(String(16), nullable=False, default="pending")  # pending/accepted/rejected
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint("follower_id", "followee_id", name="uniq_follow"),
        Index("ix_follow_status", "status"),
    )


class Message(Base):
    __tablename__ = "messages"
    id            = Column(Integer, primary_key=True)
    sender_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipient_id  = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    group_id      = Column(Integer, ForeignKey("groups.id"), nullable=True, index=True)
    client_msg_id = Column(String(64), nullable=True, index=True)
    kind          = Column(String(16), nullable=False, default="text")
    ciphertext    = Column(Text, nullable=False)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    edited_at     = Column(DateTime, nullable=True)
    deleted       = Column(Boolean, default=False)
    delivered     = Column(Boolean, default=False)
    likes         = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")


class Reaction(Base):
    __tablename__ = "reactions"
    id            = Column(Integer, primary_key=True)
    message_id    = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    client_msg_id = Column(String(64), nullable=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    emoji         = Column(String(32), nullable=False)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    message       = relationship("Message", back_populates="likes")
    __table_args__ = (
        UniqueConstraint("message_id", "user_id", "emoji", name="uniq_reaction_per_user_emoji"),
        Index("ix_reaction_client_msg_id", "client_msg_id"),
    )


class Room(Base):
    __tablename__ = "rooms"
    id          = Column(Integer, primary_key=True)
    token       = Column(String(64), unique=True, nullable=False, index=True)
    name        = Column(String(120), nullable=True)
    owner_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    max_members = Column(Integer, default=10, nullable=False)
    incognito   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at  = Column(DateTime, nullable=True)
    members     = relationship("RoomMember", back_populates="room", cascade="all, delete-orphan")
    messages    = relationship("RoomMessage", back_populates="room", cascade="all, delete-orphan")


class RoomMember(Base):
    __tablename__ = "room_members"
    id        = Column(Integer, primary_key=True)
    room_id   = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=True)
    nickname  = Column(String(64), nullable=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    room      = relationship("Room", back_populates="members")
    __table_args__ = (
        UniqueConstraint("room_id", "nickname", name="uniq_room_nickname"),
        Index("ix_room_member_user", "room_id", "user_id"),
    )


class RoomMessage(Base):
    __tablename__ = "room_messages"
    id              = Column(Integer, primary_key=True)
    room_id         = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    sender_nickname = Column(String(64), nullable=False)
    content         = Column(Text, nullable=True)    # deprecated \u2014 legacy plaintext
    ciphertext      = Column(Text, nullable=True)    # E2EE ciphertext
    kind            = Column(String(16), default="text", nullable=False)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    room            = relationship("Room", back_populates="messages")


Base.metadata.create_all(engine)


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#  DB MIGRATION  (idempotent)
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _migrate_db() -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        def cols(table: str) -> set:
            cur.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cur.fetchall()}

        u = cols("users")
        user_cols = [
            ("e2ee_api_key",    "ALTER TABLE users ADD COLUMN e2ee_api_key VARCHAR(128)"),
            ("e2ee_key_backup", "ALTER TABLE users ADD COLUMN e2ee_key_backup TEXT"),
            ("e2ee_key_expires","ALTER TABLE users ADD COLUMN e2ee_key_expires DATETIME"),
            ("e2ee_last_rotate","ALTER TABLE users ADD COLUMN e2ee_last_rotate DATETIME"),
            ("location",        "ALTER TABLE users ADD COLUMN location VARCHAR(120)"),
            ("updated_at",      "ALTER TABLE users ADD COLUMN updated_at DATETIME"),
            ("invite_token",    "ALTER TABLE users ADD COLUMN invite_token VARCHAR(64)"),
        ]
        for col, sql in user_cols:
            if col not in u:
                cur.execute(sql)
                logger.info(f"\u2713 Migrated: users.{col}")

        for idx_sql in [
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_e2ee_api_key ON users(e2ee_api_key)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_invite_token  ON users(invite_token)",
        ]:
            try:
                cur.execute(idx_sql)
            except sqlite3.OperationalError:
                pass

        rm = cols("room_messages")
        if "ciphertext" not in rm:
            cur.execute("ALTER TABLE room_messages ADD COLUMN ciphertext TEXT")
            logger.info("\u2713 Migrated: room_messages.ciphertext")

        conn.commit()
        conn.close()
        logger.info("\u2713 DB migration complete")
    except Exception as e:
        logger.warning(f"\u26a0 Migration issue (non-fatal): {e}")


_migrate_db()


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  WEBSOCKET HUB  (defined before routes)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

class Hub:
    def __init__(self) -> None:
        self.connections: Dict[int, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, user_id: int, ws: WebSocket) -> None:
        async with self.lock:
            self.connections.setdefault(user_id, set()).add(ws)

    async def disconnect(self, user_id: int, ws: WebSocket) -> None:
        async with self.lock:
            if user_id in self.connections:
                self.connections[user_id].discard(ws)
                if not self.connections[user_id]:
                    del self.connections[user_id]

    async def deliver(self, user_id: int, msg: dict) -> None:
        async with self.lock:
            sockets = list(self.connections.get(user_id, set()))
        dead: List[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(user_id, ws)

    def is_online(self, user_id: int) -> bool:
        return user_id in self.connections


hub = Hub()


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  AUTH & CRYPTO
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def hash_password(pw: str) -> str:
    if len(pw.encode("utf-8")) > 72:
        logger.warning("\u26a0 Password >72 bytes will be truncated by bcrypt")
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception as e:
        logger.warning(f"\u26a0 Password verify error: {e}")
        return False


def make_token(user_id: int) -> str:
    return jwt.encode(
        {
            "sub": str(user_id),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> int:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(data["sub"])
    except (JWTError, KeyError, ValueError) as e:
        logger.warning(f"\u26a0 Token decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def generate_e2ee_api_key() -> str:
    return f"e2ee_{secrets.token_urlsafe(96)}"


def generate_invite_token() -> str:
    return secrets.token_urlsafe(48)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid Authorization format")
    uid  = decode_token(parts[1])
    user = db.get(User, uid)
    if not user:
        raise HTTPException(401, "User not found")
    return user


def _norm(u: str) -> str:
    return (u or "").strip().lower()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  RATE LIMITING  (optional \u2014 graceful fallback if slowapi not installed)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    _RATE_OK = True
except ImportError:
    _RATE_OK = False
    logger.warning("\u26a0 slowapi not installed \u2014 rate limiting disabled. pip install slowapi")

    class _NoopLimiter:                             # type: ignore
        def limit(self, *a, **kw):
            return lambda fn: fn
    limiter = _NoopLimiter()                        # type: ignore


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
#  SCHEMAS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

# \u2500\u2500 Auth \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class RegisterIn(BaseModel):
    username:     str           = Field(min_length=3, max_length=64)
    password:     str           = Field(min_length=6, max_length=256)
    public_key:   Optional[str] = None
    display_name: Optional[str] = None
    avatar:       Optional[str] = None

    @field_validator("username")
    @classmethod
    def _u(cls, v: str) -> str:
        return _norm(v)


class LoginIn(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _u(cls, v: str) -> str:
        return _norm(v)


class TokenOut(BaseModel):
    access_token: str
    token_type:   str           = "bearer"
    user_id:      int
    username:     str
    e2ee_api_key: Optional[str] = None


class UserOut(BaseModel):
    id:                  int
    username:            str
    display_name:        Optional[str] = None
    avatar:              Optional[str] = None
    location:            Optional[str] = None
    public_key:          Optional[str] = None
    is_private:          bool          = False
    e2ee_api_key_exists: bool          = False


class ProfileUpdateIn(BaseModel):
    display_name: Optional[str] = Field(None, max_length=120)
    avatar:       Optional[str] = None
    location:     Optional[str] = Field(None, max_length=120)
    public_key:   Optional[str] = None
    is_private:   Optional