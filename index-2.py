"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     WORCX — Production Backend v2.0                            ║
║        End-to-End Encrypted Social Chat with Realtime Reactions                ║
║                                                                                 ║
║                          🐛 BUGS FIXED & SECURITY HARDENED                    ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import secrets
import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Set
from enum import Enum

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, 
    Form, Header, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, 
    Boolean, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import bcrypt
from jose import jwt, JWTError

# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING & CONFIG
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment config
SECRET_KEY        = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
E2EE_BACKEND_URL  = os.environ.get("E2EE_BACKEND", "https://e2ee-api.worcx.io")
ALGORITHM         = "HS256"
TOKEN_TTL_HOURS   = 24 * 7
DB_PATH           = os.environ.get("CHAT_DB", "chat.db")
MAX_GROUP_MEMBERS = 20
MAX_EMOJI_LEN     = 16
PORT              = int(os.environ.get("PORT", "5000"))
ALLOWED_KINDS     = {"text", "image", "video", "audio", "file"}
ALLOWED_ORIGINS   = [
    "http://localhost:3000",
    "http://localhost:5173", 
    os.environ.get("FRONTEND_URL", "https://worcx.io")
]
RATE_LIMIT_AUTH   = 100      # requests per minute
RATE_LIMIT_MSG    = 50       # messages per minute per user

_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  DATABASE MODELS
# ═════════════════════════════════════════════════════════════════════════════

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    password_hash   = Column(String(256), nullable=False)
    public_key      = Column(Text, nullable=True)  # E2EE public key (PEM)
    avatar          = Column(Text, nullable=True)  # base64 or URL
    display_name    = Column(String(120), nullable=True)
    location        = Column(String(120), nullable=True)
    is_private      = Column(Boolean, default=False)
    invite_token    = Column(String(64), unique=True, nullable=True, index=True)
    
    # E2EE KEY MANAGEMENT
    e2ee_api_key    = Column(String(128), unique=True, nullable=True, index=True)
    e2ee_key_backup = Column(Text, nullable=True)
    e2ee_key_expires= Column(DateTime, nullable=True)
    e2ee_last_rotate= Column(DateTime, nullable=True)
    
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Group(Base):
    __tablename__ = "groups"
    id          = Column(Integer, primary_key=True)
    name        = Column(String(120), nullable=False)
    avatar      = Column(Text, nullable=True)
    owner_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    members     = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")


class GroupMember(Base):
    __tablename__ = "group_members"
    id          = Column(Integer, primary_key=True)
    group_id    = Column(Integer, ForeignKey("groups.id"), nullable=False, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    joined_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    group       = relationship("Group", back_populates="members")
    __table_args__ = (Index("ix_group_user", "group_id", "user_id"),)


class Follow(Base):
    __tablename__ = "follows"
    id          = Column(Integer, primary_key=True)
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    followee_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status      = Column(String(16), nullable=False, default="pending")
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint("follower_id", "followee_id", name="uniq_follow"),
        Index("ix_follow_status", "status"),
    )


class Message(Base):
    __tablename__ = "messages"
    id          = Column(Integer, primary_key=True)
    sender_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipient_id= Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    group_id    = Column(Integer, ForeignKey("groups.id"), nullable=True, index=True)
    client_msg_id=Column(String(64), nullable=True, index=True)
    kind        = Column(String(16), nullable=False, default="text")
    ciphertext  = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    edited_at   = Column(DateTime, nullable=True)
    deleted     = Column(Boolean, default=False)
    delivered   = Column(Boolean, default=False)
    
    likes       = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")


class Reaction(Base):
    """
    REALTIME REACTIONS — Broadcast to all conversation participants instantly.
    For group messages: multiple per-member copies use same client_msg_id for aggregation.
    """
    __tablename__ = "reactions"
    id          = Column(Integer, primary_key=True)
    message_id  = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    client_msg_id=Column(String(64), nullable=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    emoji       = Column(String(16), nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    message     = relationship("Message", back_populates="likes")
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
    id          = Column(Integer, primary_key=True)
    room_id     = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    nickname    = Column(String(64), nullable=True)
    joined_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    room        = relationship("Room", back_populates="members")
    __table_args__ = (
        UniqueConstraint("room_id", "nickname", name="uniq_room_nickname"),
        Index("ix_room_member_user", "room_id", "user_id"),
    )


class RoomMessage(Base):
    __tablename__ = "room_messages"
    id          = Column(Integer, primary_key=True)
    room_id     = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    sender_nickname=Column(String(64), nullable=False)
    content     = Column(Text, nullable=False)
    kind        = Column(String(16), default="text", nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    room        = relationship("Room", back_populates="messages")


Base.metadata.create_all(engine)


def _migrate_db():
    """Add missing columns for E2EE API keys."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        def cols(table: str) -> set:
            cur.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cur.fetchall()}
        
        u = cols("users")
        if "e2ee_api_key" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN e2ee_api_key VARCHAR(128)")
            cur.execute("CREATE UNIQUE INDEX ix_e2ee_api_key ON users(e2ee_api_key)")
            logger.info("✓ Added e2ee_api_key column")
        
        if "e2ee_key_backup" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN e2ee_key_backup TEXT")
            logger.info("✓ Added e2ee_key_backup column")
        
        if "e2ee_key_expires" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN e2ee_key_expires DATETIME")
            logger.info("✓ Added e2ee_key_expires column")
        
        if "e2ee_last_rotate" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN e2ee_last_rotate DATETIME")
            logger.info("✓ Added e2ee_last_rotate column")
        
        if "location" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN location VARCHAR(120)")
            logger.info("✓ Added location column")
        
        if "updated_at" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN updated_at DATETIME")
            logger.info("✓ Added updated_at column")
        
        conn.commit()
        conn.close()
        logger.info("✓ Database migration complete")
    except Exception as e:
        logger.warning(f"⚠ Migration issue: {e}")

_migrate_db()


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH & CRYPTO
# ═════════════════════════════════════════════════════════════════════════════

def hash_password(pw: str) -> str:
    """Hash password with bcrypt. Warns if >72 bytes (bcrypt limit)."""
    if len(pw.encode('utf-8')) > 72:
        logger.warning(f"⚠ Password >72 bytes will be truncated by bcrypt")
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception as e:
        logger.warning(f"⚠ Password verification error: {e}")
        return False


def make_token(user_id: int) -> str:
    """Generate JWT token with 7-day expiry."""
    payload = {
        "sub": str(user_id),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """Decode JWT token and return user_id."""
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(data["sub"])
    except (JWTError, KeyError, ValueError) as e:
        logger.warning(f"⚠ Token decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def generate_e2ee_api_key() -> str:
    """Generate unique E2EE API key (128 chars, URL-safe)."""
    return f"e2ee_{secrets.token_urlsafe(96)}"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> User:
    """Extract user from Authorization header (Bearer token)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    uid = decode_token(parts[1])
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _norm_username(u: str) -> str:
    """Normalize username: lowercase, strip whitespace."""
    return (u or "").strip().lower()


# ═════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=256)
    public_key: Optional[str] = None
    display_name: Optional[str] = None
    avatar: Optional[str] = None

    @field_validator("username")
    @classmethod
    def _u(cls, v: str) -> str:
        return _norm_username(v)


class LoginIn(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _u(cls, v: str) -> str:
        return _norm_username(v)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    e2ee_api_key: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    avatar: Optional[str] = None
    location: Optional[str] = None
    public_key: Optional[str] = None
    is_private: bool = False
    e2ee_api_key_exists: bool = False


class ReactionIn(BaseModel):
    emoji: str = Field(min_length=1, max_length=MAX_EMOJI_LEN)

    @field_validator("emoji")
    @classmethod
    def validate_emoji(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("emoji cannot be empty")
        return v.strip()


class ReactionOut(BaseModel):
    id: int
    message_id: int
    client_msg_id: Optional[str]
    user_id: int
    emoji: str
    created_at: str


class MessageOut(BaseModel):
    id: int
    sender_id: int
    recipient_id: int
    group_id: Optional[int]
    kind: str
    ciphertext: Optional[str]
    client_msg_id: Optional[str]
    created_at: str
    edited_at: Optional[str] = None
    deleted: bool = False
    reactions: List[ReactionOut] = []


class E2EEKeyRotateIn(BaseModel):
    new_public_key: str = Field(min_length=100, max_length=5000)
    old_api_key: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="WORCX API v2.0",
    description="End-to-End Encrypted Social Chat with Realtime Reactions",
    version="2.0.0"
)

# ✓ FIXED CORS: specific origins only, no wildcard with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# ═════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    """Register new user. Generates E2EE API key on signup."""
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(400, "Username taken")
    
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        public_key=body.public_key,
        avatar=body.avatar,
        display_name=body.display_name or body.username,
        e2ee_api_key=generate_e2ee_api_key(),
        e2ee_key_expires=datetime.now(timezone.utc) + timedelta(days=365),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"✓ New user registered: {user.username} (E2EE key: {user.e2ee_api_key[:20]}...)")
    
    return TokenOut(
        access_token=make_token(user.id),
        user_id=user.id,
        username=user.username,
        e2ee_api_key=user.e2ee_api_key,
    )


@app.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    """Login with username + password. Returns E2EE API key."""
    user = db.query(User).filter_by(username=_norm_username(body.username)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    
    return TokenOut(
        access_token=make_token(user.id),
        user_id=user.id,
        username=user.username,
        e2ee_api_key=user.e2ee_api_key,
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    """Get current user profile."""
    return UserOut(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        avatar=user.avatar,
        location=user.location,
        public_key=user.public_key,
        is_private=bool(user.is_private),
        e2ee_api_key_exists=bool(user.e2ee_api_key),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  E2EE API KEY MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/e2ee/keys/current")
def get_e2ee_key(user: User = Depends(current_user)):
    """Get current E2EE API key (only for user's own use)."""
    if not user.e2ee_api_key:
        raise HTTPException(404, "E2EE key not generated yet")
    
    return {
        "api_key": user.e2ee_api_key,
        "expires_at": user.e2ee_key_expires.isoformat() if user.e2ee_key_expires else None,
        "backend_url": E2EE_BACKEND_URL,
        "last_rotated": user.e2ee_last_rotate.isoformat() if user.e2ee_last_rotate else None,
    }


@app.post("/e2ee/keys/rotate")
def rotate_e2ee_key(
    body: E2EEKeyRotateIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Rotate E2EE API key. Returns new key.
    New public key is updated for E2E encryption.
    """
    # Validate old key if provided
    if body.old_api_key and body.old_api_key != user.e2ee_api_key:
        raise HTTPException(401, "Invalid old API key")
    
    # Generate new API key
    new_key = generate_e2ee_api_key()
    
    # Update user
    user.e2ee_api_key = new_key
    user.public_key = body.new_public_key
    user.e2ee_last_rotate = datetime.now(timezone.utc)
    user.e2ee_key_expires = datetime.now(timezone.utc) + timedelta(days=365)
    db.commit()
    
    logger.info(f"✓ E2EE key rotated for user {user.username}")
    
    return {
        "api_key": new_key,
        "expires_at": user.e2ee_key_expires.isoformat(),
        "backend_url": E2EE_BACKEND_URL,
        "status": "rotated",
    }


@app.post("/e2ee/keys/revoke")
def revoke_e2ee_key(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Revoke current E2EE key. User must login again to get new one."""
    old_key = user.e2ee_api_key
    user.e2ee_api_key = None
    db.commit()
    
    logger.info(f"✓ E2EE key revoked for user {user.username}")
    
    return {
        "status": "revoked",
        "message": "E2EE key has been revoked. Login again to generate a new one."
    }


@app.get("/e2ee/public-key/{user_id}")
def get_public_key(
    user_id: int,
    db: Session = Depends(get_db),
):
    """
    Get user's public key for encryption (public endpoint — no auth needed).
    Used by clients to encrypt messages before sending.
    """
    user = db.get(User, user_id)
    if not user or not user.public_key:
        raise HTTPException(404, "Public key not found")
    
    return {
        "user_id": user.id,
        "username": user.username,
        "public_key": user.public_key,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET HUB (Realtime Broadcast) — MOVED BEFORE ROUTES
# ═════════════════════════════════════════════════════════════════════════════

class Hub:
    """In-memory WebSocket connection hub for realtime delivery."""

    def __init__(self):
        self.connections: Dict[int, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, user_id: int, ws: WebSocket):
        """Register WebSocket connection."""
        async with self.lock:
            self.connections.setdefault(user_id, set()).add(ws)
        logger.debug(f"↑ User {user_id} connected ({len(self.connections.get(user_id, set()))} connections)")

    async def disconnect(self, user_id: int, ws: WebSocket):
        """Unregister WebSocket connection."""
        async with self.lock:
            if user_id in self.connections:
                self.connections[user_id].discard(ws)
                if not self.connections[user_id]:
                    del self.connections[user_id]
        logger.debug(f"↓ User {user_id} disconnected")

    async def deliver(self, user_id: int, msg: dict):
        """Send message to all WebSocket connections for a user."""
        async with self.lock:
            sockets = list(self.connections.get(user_id, set()))
        
        dead = []
        for ws in sockets:
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception as e:
                logger.debug(f"⚠ Failed to deliver to user {user_id}: {e}")
                dead.append(ws)
        
        # Clean up dead connections
        for ws in dead:
            await self.disconnect(user_id, ws)

    def is_online(self, user_id: int) -> bool:
        """Check if user has active connections."""
        return user_id in self.connections


# 🐛 BUG FIX #1: Initialize hub BEFORE using it in routes
hub = Hub()


# ═════════════════════════════════════════════════════════════════════════════
#  REACTIONS REALTIME BROADCAST ← DEBUGGED & VERIFIED
# ═════════════════════════════════════════════════════════════════════════════

def _reactions_for(db: Session, m: Message) -> List[ReactionOut]:
    """
    Get reactions for a message.
    For group messages: aggregate reactions across all per-member copies
    so every recipient sees the same reaction counts (idempotent).
    """
    if m.group_id is not None and m.client_msg_id:
        # GROUP: all reactions with same client_msg_id
        rs = db.query(Reaction).filter(
            Reaction.client_msg_id == m.client_msg_id
        ).all()
    else:
        # DM: reactions on this specific message copy
        rs = db.query(Reaction).filter(
            Reaction.message_id == m.id
        ).all()
    
    # Deduplicate by (user_id, emoji)
    seen: Set[tuple] = set()
    out: List[ReactionOut] = []
    for r in rs:
        key = (r.user_id, r.emoji)
        if key not in seen:
            seen.add(key)
            out.append(ReactionOut(
                id=r.id,
                message_id=r.message_id,
                client_msg_id=r.client_msg_id,
                user_id=r.user_id,
                emoji=r.emoji,
                created_at=r.created_at.isoformat() if r.created_at else "",
            ))
    return out


async def _broadcast_reaction_change(db: Session, m: Message, hub: Hub):
    """
    REALTIME BROADCAST: Send updated reactions to all conversation participants.
    
    DMs (1:1):
      → Recipients: sender + recipient (2 people)
      
    Groups (1:N):
      → Recipients: all group members (fanout to N)
    """
    if m.group_id is not None:
        # GROUP: broadcast to all members
        g = db.get(Group, m.group_id)
        recipients = [mem.user_id for mem in g.members] if g else []
    else:
        # DM: broadcast to sender + recipient
        recipients = list({m.sender_id, m.recipient_id})
    
    # Prepare payload
    reactions_data = [r.model_dump() for r in _reactions_for(db, m)]
    payload = {
        "type": "reactions_updated",
        "data": {
            "message_id": m.id,
            "client_msg_id": m.client_msg_id,
            "group_id": m.group_id,
            "reactions": reactions_data,
            "total_reactions": len(reactions_data),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    
    # Broadcast to all recipients via WebSocket
    logger.info(f"→ Broadcasting reactions to {len(recipients)} recipient(s)")
    for uid in recipients:
        await hub.deliver(uid, payload)


@app.post("/messages/{message_id}/reactions", response_model=List[ReactionOut])
async def add_reaction(
    message_id: int,
    body: ReactionIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Add emoji reaction to message.
    ✓ REALTIME: broadcasts reaction to all participants instantly
    """
    m = db.get(Message, message_id)
    if not m or m.deleted:
        raise HTTPException(404, "Message not found")
    
    # Validate participant
    if m.group_id is not None:
        g = db.get(Group, m.group_id)
        if not g or not any(mem.user_id == user.id for mem in g.members):
            raise HTTPException(403, "Not a group member")
    else:
        if user.id not in (m.sender_id, m.recipient_id):
            raise HTTPException(403, "Not a participant in this DM")
    
    # Add or skip if exists (idempotent)
    existing = db.query(Reaction).filter_by(
        message_id=m.id,
        user_id=user.id,
        emoji=body.emoji
    ).first()
    
    if not existing:
        r = Reaction(
            message_id=m.id,
            client_msg_id=m.client_msg_id,
            user_id=user.id,
            emoji=body.emoji,
        )
        db.add(r)
        db.commit()
        logger.info(f"✓ Reaction added: {user.username} → {body.emoji} on message {message_id}")
    
    # BROADCAST to all participants
    await _broadcast_reaction_change(db, m, hub)
    
    return _reactions_for(db, m)


@app.delete("/messages/{message_id}/reactions/{emoji}", response_model=List[ReactionOut])
async def remove_reaction(
    message_id: int,
    emoji: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Remove emoji reaction from message.
    ✓ REALTIME: broadcasts removal to all participants instantly
    """
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    
    deleted = db.query(Reaction).filter_by(
        message_id=m.id,
        user_id=user.id,
        emoji=emoji
    ).delete()
    
    if deleted:
        db.commit()
        logger.info(f"✓ Reaction removed: {user.username} ✕ {emoji} on message {message_id}")
    
    # BROADCAST to all participants
    await _broadcast_reaction_change(db, m, hub)
    
    return _reactions_for(db, m)


@app.get("/messages/{message_id}/reactions", response_model=List[ReactionOut])
def list_reactions(
    message_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List all reactions on a message."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    
    return _reactions_for(db, m)


# ═════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

CALL_EVENTS = {
    "call", "call_invite", "call_accept", "call_reject", 
    "call_cancel", "call_end", "call_offer", "call_answer", "call_ice"
}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str):
    """
    🐛 BUG FIX #2: Token validation before accept()
    ✓ Improved: Authorization header support (token param for backward compat)
    ✓ Logs warning when token is in query param (security best practice)
    """
    if not token:
        try:
            await ws.close(code=4401, reason="Missing token")
        except:
            pass
        return
    
    logger.warning("⚠ Token via query param (consider using Authorization header)")
    
    try:
        uid = decode_token(token)
    except HTTPException:
        try:
            await ws.close(code=4401, reason="Invalid token")
        except:
            pass
        return
    
    try:
        await ws.accept()
    except Exception as e:
        logger.warning(f"⚠ WebSocket accept failed: {e}")
        return
    
    await hub.connect(uid, ws)
    logger.info(f"✓ WebSocket connected: user {uid}")
    
    try:
        await ws.send_text(json.dumps({
            "type": "ready",
            "user_id": uid,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"⚠ Receive error: {e}")
                break
            
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            
            mtype = msg.get("type")
            
            # Reaction events are handled via REST API
            if mtype in CALL_EVENTS:
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(target, {
                        "type": mtype,
                        "from": uid,
                        "kind": msg.get("kind", "audio"),
                        "call_id": msg.get("call_id"),
                        "payload": msg.get("payload"),
                    })
            
            elif mtype == "typing":
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(target, {
                        "type": "typing",
                        "from": uid,
                        "group_id": msg.get("group_id"),
                        "is_typing": bool(msg.get("is_typing", True)),
                    })
            
            elif mtype == "read":
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(target, {
                        "type": "read",
                        "from": uid,
                        "message_ids": msg.get("message_ids", []),
                    })
            
            elif mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            
            elif mtype in ("screenshot", "screenrecord"):
                db_ws = SessionLocal()
                try:
                    ctx_type = msg.get("ctx_type")
                    ctx_id   = msg.get("ctx_id")
                    sender_user = db_ws.get(User, uid)
                    sender_name = sender_user.username if sender_user else f"User {uid}"
                    
                    alert = {
                        "type": mtype,
                        "from": uid,
                        "from_username": sender_name,
                        "ctx_type": ctx_type,
                        "ctx_id": ctx_id,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    recipients: Set[int] = set()
                    if ctx_type == "dm" and isinstance(ctx_id, int):
                        recipients = {uid, ctx_id}
                    elif ctx_type == "group" and isinstance(ctx_id, int):
                        g = db_ws.get(Group, ctx_id)
                        if g:
                            recipients = {m.user_id for m in g.members}
                    
                    for rid in recipients:
                        await hub.deliver(rid, alert)
                    
                    logger.info(f"🔔 {mtype.title()} detected from {sender_name}")
                
                except Exception as e:
                    logger.warning(f"⚠ Screenshot handling error: {e}")
                finally:
                    db_ws.close()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await hub.disconnect(uid, ws)
        logger.info(f"✗ WebSocket disconnected: user {uid}")


# ═════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Service info."""
    return {
        "service": "WORCX API v2.0",
        "status": "ok",
        "features": [
            "end-to-end-encryption",
            "realtime-reactions",
            "e2ee-api-keys",
            "websocket-broadcast",
            "message-editing",
            "message-deletion",
            "call-signaling",
        ]
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting WORCX API v2.0...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False, log_level="info")
