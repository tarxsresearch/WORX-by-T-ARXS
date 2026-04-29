"""
End-to-End Encrypted Chat Backend
Features:
- Username/password auth (bcrypt + JWT). Login accepts JSON or OAuth2 form data.
- Profile avatars (any base64 image)
- E2E encryption: clients store private keys, server stores only public keys
  and ciphertext payloads. Server NEVER sees plaintext.
- Direct messages (1:1) and group chats (max 20 members)
- Message kinds: text, image, video, audio, file (all encrypted client-side)
- Edit and delete messages
- Emoji reactions (multiple distinct emojis per user per message)
- Real-time delivery via WebSocket (typing, read, reactions, edits, deletes)
- WebRTC call signaling (audio + video) with ringing/accept/reject/end events
- Anonymous ephemeral rooms (incognito, auto-expiring)
"""

import os
import json
import secrets
import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Set

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Form, Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import bcrypt
from jose import jwt, JWTError

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Config ----------
SECRET_KEY = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24 * 7
DB_PATH = os.environ.get("CHAT_DB", "chat.db")
MAX_GROUP_MEMBERS = 20
MAX_EMOJI_LEN = 16
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED_KINDS = {"text", "image", "video", "audio", "file"}

_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# ---------- DB ----------
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    public_key = Column(Text, nullable=True)
    avatar = Column(Text, nullable=True)
    display_name = Column(String(120), nullable=True)
    is_private = Column(Boolean, default=False, nullable=False)
    invite_token = Column(String(64), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Group(Base):
    __tablename__ = "groups"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    avatar = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    members = relationship(
        "GroupMember", back_populates="group", cascade="all, delete-orphan"
    )


class GroupMember(Base):
    __tablename__ = "group_members"
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    group = relationship("Group", back_populates="members")


# Follow / connection system
# status: "pending" | "accepted" | "rejected"
class Follow(Base):
    __tablename__ = "follows"
    id = Column(Integer, primary_key=True)
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    followee_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(16), nullable=False, default="pending")  # pending/accepted/rejected
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint("follower_id", "followee_id", name="uniq_follow"),
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipient_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=True, index=True)
    client_msg_id = Column(String(64), nullable=True, index=True)
    kind = Column(String(16), nullable=False, default="text")
    ciphertext = Column(Text, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    edited_at = Column(DateTime, nullable=True)
    deleted = Column(Boolean, default=False, nullable=False)
    delivered = Column(Boolean, default=False)


class Reaction(Base):
    __tablename__ = "reactions"
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    client_msg_id = Column(String(64), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    emoji = Column(String(16), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint(
            "message_id", "user_id", "emoji", name="uniq_reaction_per_user_emoji"
        ),
    )


class Room(Base):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable = anon
    max_members = Column(Integer, default=10, nullable=False)
    incognito = Column(Boolean, default=True, nullable=False)  # no history stored
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)  # optional TTL
    members = relationship("RoomMember", back_populates="room", cascade="all, delete-orphan")
    messages = relationship("RoomMessage", back_populates="room", cascade="all, delete-orphan")


class RoomMember(Base):
    __tablename__ = "room_members"
    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)   # nullable = anon
    nickname = Column(String(64), nullable=True)                        # anon display name
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    room = relationship("Room", back_populates="members")
    __table_args__ = (UniqueConstraint("room_id", "user_id", name="uniq_room_member"),)


class RoomMessage(Base):
    __tablename__ = "room_messages"
    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False, index=True)
    sender_nickname = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    kind = Column(String(16), default="text", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    room = relationship("Room", back_populates="messages")


Base.metadata.create_all(engine)


# ---------- Tiny SQLite migration: add new columns if missing ----------
def _ensure_columns():
    """Add any missing columns without dropping existing data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()

            def cols(table: str) -> set:
                cur.execute(f"PRAGMA table_info({table})")
                return {row[1] for row in cur.fetchall()}

            u = cols("users")
            if "avatar" not in u:
                cur.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
            if "display_name" not in u:
                cur.execute("ALTER TABLE users ADD COLUMN display_name VARCHAR(120)")
            if "is_private" not in u:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN is_private BOOLEAN NOT NULL DEFAULT 0"
                )
            if "invite_token" not in u:
                cur.execute("ALTER TABLE users ADD COLUMN invite_token VARCHAR(64)")

            g = cols("groups")
            if "avatar" not in g:
                cur.execute("ALTER TABLE groups ADD COLUMN avatar TEXT")

            m = cols("messages")
            if "edited_at" not in m:
                cur.execute("ALTER TABLE messages ADD COLUMN edited_at DATETIME")
            if "deleted" not in m:
                cur.execute(
                    "ALTER TABLE messages ADD COLUMN deleted BOOLEAN NOT NULL DEFAULT 0"
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(f"_ensure_columns migration issue: {exc}")


_ensure_columns()


# ---------- Auth helpers ----------
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=True)


def hash_password(pw: str) -> str:
    """Hash password using bcrypt with salt."""
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def make_token(user_id: int) -> str:
    """Generate JWT token with 7-day expiry."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """Decode JWT token and return user_id."""
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(data["sub"])
    except (JWTError, KeyError, ValueError) as e:
        logger.warning(f"Token decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def get_db() -> Session:
    """Dependency to provide DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    token: str = Depends(oauth2), db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from token."""
    uid = decode_token(token)
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _norm_username(u: str) -> str:
    """Normalize username to lowercase and strip whitespace."""
    return (u or "").strip().lower()


# ---------- Schemas ----------
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


class PublicKeyIn(BaseModel):
    public_key: str


class ProfileIn(BaseModel):
    display_name: Optional[str] = None
    avatar: Optional[str] = None
    public_key: Optional[str] = None
    is_private: Optional[bool] = None


class UserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    avatar: Optional[str] = None
    public_key: Optional[str] = None
    is_private: bool = False


class FollowOut(BaseModel):
    id: int
    follower_id: int
    followee_id: int
    status: str  # pending / accepted / rejected
    created_at: str


class InviteLinkOut(BaseModel):
    token: str
    url: str  # full shareable URL hint (client should build the real one)


class GroupCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    member_ids: List[int] = []
    avatar: Optional[str] = None


class GroupUpdateIn(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None


class GroupOut(BaseModel):
    id: int
    name: str
    avatar: Optional[str] = None
    owner_id: int
    member_ids: List[int]


class AddMemberIn(BaseModel):
    user_id: int


class RecipientPayload(BaseModel):
    recipient_id: int
    ciphertext: str


class SendDMIn(BaseModel):
    recipient_id: int
    ciphertext: str
    kind: str = "text"
    client_msg_id: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        if v not in ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {sorted(ALLOWED_KINDS)}")
        return v


class SendGroupIn(BaseModel):
    group_id: int
    payloads: List[RecipientPayload]
    kind: str = "text"
    client_msg_id: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        if v not in ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {sorted(ALLOWED_KINDS)}")
        return v


class EditMessageIn(BaseModel):
    payloads: List[RecipientPayload]


class ChangePasswordIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=6, max_length=256)


class ReactionIn(BaseModel):
    """FIXED: Was missing from original code."""
    emoji: str = Field(min_length=1, max_length=MAX_EMOJI_LEN)

    @field_validator("emoji")
    @classmethod
    def validate_emoji(cls, v: str) -> str:
        # Basic emoji validation: ensure it's not empty and within length
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


# ---------- App ----------
app = FastAPI(title="Encrypted Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Routes: Auth ----------
@app.post("/auth/register", response_model=TokenOut)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    """Register a new user."""
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(400, "Username taken")
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        public_key=body.public_key,
        avatar=body.avatar,
        display_name=body.display_name or body.username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenOut(
        access_token=make_token(user.id), user_id=user.id, username=user.username
    )


@app.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    """Login with username and password (JSON)."""
    user = db.query(User).filter_by(username=body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    return TokenOut(
        access_token=make_token(user.id), user_id=user.id, username=user.username
    )


@app.post("/auth/token", response_model=TokenOut)
def login_form(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """Login with OAuth2 form data."""
    uname = _norm_username(username)
    user = db.query(User).filter_by(username=uname).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    return TokenOut(
        access_token=make_token(user.id), user_id=user.id, username=user.username
    )


def _user_out(u: User) -> UserOut:
    """Convert User model to UserOut response."""
    return UserOut(
        id=u.id,
        username=u.username,
        display_name=u.display_name,
        avatar=u.avatar,
        public_key=u.public_key,
        is_private=bool(u.is_private),
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    """Get current user profile."""
    return _user_out(user)


@app.put("/auth/public-key", response_model=UserOut)
def set_public_key(
    body: PublicKeyIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Update user's public key."""
    user.public_key = body.public_key
    db.commit()
    return _user_out(user)


@app.put("/auth/profile", response_model=UserOut)
def update_profile(
    body: ProfileIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Update user profile (display name, avatar, public key, privacy)."""
    if body.display_name is not None:
        user.display_name = body.display_name.strip() or user.username
    if body.avatar is not None:
        user.avatar = body.avatar
    if body.public_key is not None:
        user.public_key = body.public_key
    if body.is_private is not None:
        user.is_private = body.is_private
    db.commit()
    return _user_out(user)


@app.put("/auth/password")
def change_password(
    body: ChangePasswordIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Change user password."""
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(401, "Old password incorrect")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"ok": True}


# ---------- Follow helpers ----------
def _is_connected(db: Session, a: int, b: int) -> bool:
    """True if a follows b (accepted) OR b follows a (accepted)."""
    return db.query(Follow).filter(
        Follow.status == "accepted",
    ).filter(
        ((Follow.follower_id == a) & (Follow.followee_id == b)) |
        ((Follow.follower_id == b) & (Follow.followee_id == a))
    ).first() is not None


def _follow_status(db: Session, follower: int, followee: int) -> Optional[str]:
    """Return follow status from follower → followee, or None."""
    f = db.query(Follow).filter_by(follower_id=follower, followee_id=followee).first()
    return f.status if f else None


def _follow_out(f: Follow) -> FollowOut:
    """Convert Follow model to FollowOut response."""
    return FollowOut(
        id=f.id,
        follower_id=f.follower_id,
        followee_id=f.followee_id,
        status=f.status,
        created_at=f.created_at.isoformat() if f.created_at else "",
    )


# ---------- Routes: Users ----------
@app.get("/users", response_model=List[UserOut])
def list_users(
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    """Search users by username."""
    query = db.query(User)
    if q:
        query = query.filter(User.username.ilike(f"%{q.strip().lower()}%"))
    all_u = query.limit(200).all()
    # Private accounts ARE visible in search but profile is locked
    # (frontend shows lock icon + Follow button instead of Open Chat)
    return [_user_out(u) for u in all_u if u.id != user.id][:100]


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    """Get user profile by ID."""
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    # Profile always visible — privacy only gates messaging
    return _user_out(u)


# ---------- Routes: Follow System ----------
@app.post("/users/{user_id}/follow", response_model=FollowOut)
def send_follow_request(
    user_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Send a follow request (or auto-follow if public)."""
    if user_id == user.id:
        raise HTTPException(400, "Cannot follow yourself")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "User not found")
    existing = db.query(Follow).filter_by(
        follower_id=user.id, followee_id=user_id
    ).first()
    if existing:
        if existing.status == "accepted":
            raise HTTPException(400, "Already following")
        if existing.status == "pending":
            raise HTTPException(400, "Follow request already sent")
        # rejected — allow re-request
        existing.status = "pending" if target.is_private else "accepted"
        db.commit()
        db.refresh(existing)
        return _follow_out(existing)
    status = "pending" if target.is_private else "accepted"
    f = Follow(follower_id=user.id, followee_id=user_id, status=status)
    db.add(f)
    db.commit()
    db.refresh(f)
    return _follow_out(f)


@app.delete("/users/{user_id}/follow")
def unfollow(
    user_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Unfollow a user."""
    db.query(Follow).filter_by(follower_id=user.id, followee_id=user_id).delete()
    db.commit()
    return {"ok": True}


@app.get("/follow/requests", response_model=List[FollowOut])
def list_follow_requests(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get pending follow requests sent to current user."""
    reqs = db.query(Follow).filter_by(followee_id=user.id, status="pending").all()
    return [_follow_out(f) for f in reqs]


@app.get("/follow/following", response_model=List[UserOut])
def list_following(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get users that current user follows."""
    follows = db.query(Follow).filter_by(follower_id=user.id, status="accepted").all()
    ids = [f.followee_id for f in follows]
    users = db.query(User).filter(User.id.in_(ids)).all() if ids else []
    return [_user_out(u) for u in users]


@app.get("/follow/followers", response_model=List[UserOut])
def list_followers(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get users who follow current user."""
    follows = db.query(Follow).filter_by(followee_id=user.id, status="accepted").all()
    ids = [f.follower_id for f in follows]
    users = db.query(User).filter(User.id.in_(ids)).all() if ids else []
    return [_user_out(u) for u in users]


@app.post("/follow/requests/{follow_id}/accept", response_model=FollowOut)
def accept_follow(
    follow_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Accept a follow request."""
    f = db.get(Follow, follow_id)
    if not f or f.followee_id != user.id:
        raise HTTPException(404, "Request not found")
    f.status = "accepted"
    db.commit()
    db.refresh(f)
    return _follow_out(f)


@app.post("/follow/requests/{follow_id}/reject", response_model=FollowOut)
def reject_follow(
    follow_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Reject a follow request."""
    f = db.get(Follow, follow_id)
    if not f or f.followee_id != user.id:
        raise HTTPException(404, "Request not found")
    f.status = "rejected"
    db.commit()
    db.refresh(f)
    return _follow_out(f)


# ---------- Routes: Invite Links ----------
@app.post("/invite/generate", response_model=InviteLinkOut)
def generate_invite(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Generate (or return existing) invite token for the current user."""
    if not user.invite_token:
        user.invite_token = secrets.token_urlsafe(24)
        db.commit()
    return InviteLinkOut(
        token=user.invite_token,
        url=f"/invite/{user.invite_token}",
    )


@app.post("/invite/{token}/use", response_model=FollowOut)
def use_invite(
    token: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    Using an invite link auto-connects the caller with the link owner.
    - If owner is public  → immediately accepted (both directions)
    - If owner is private → sends a follow request (pending)
    """
    owner = db.query(User).filter_by(invite_token=token).first()
    if not owner:
        raise HTTPException(404, "Invite link not found or expired")
    if owner.id == user.id:
        raise HTTPException(400, "Cannot use your own invite link")

    def _get_or_create(follower_id: int, followee_id: int, status: str) -> Follow:
        existing = db.query(Follow).filter_by(
            follower_id=follower_id, followee_id=followee_id
        ).first()
        if existing:
            if existing.status != "accepted":
                existing.status = status
            return existing
        f = Follow(follower_id=follower_id, followee_id=followee_id, status=status)
        db.add(f)
        return f

    status = "accepted" if not owner.is_private else "pending"
    # user → owner
    f1 = _get_or_create(user.id, owner.id, status)
    # owner → user (mutual connection for public accounts)
    if not owner.is_private:
        _get_or_create(owner.id, user.id, "accepted")

    db.commit()
    db.refresh(f1)
    return _follow_out(f1)


@app.get("/invite/{token}/preview")
def preview_invite(token: str, db: Session = Depends(get_db)):
    """Public endpoint — returns basic info about the invite owner (no auth needed)."""
    owner = db.query(User).filter_by(invite_token=token).first()
    if not owner:
        raise HTTPException(404, "Invite link not found")
    return {
        "user_id": owner.id,
        "username": owner.username,
        "display_name": owner.display_name or owner.username,
        "avatar": owner.avatar,
        "is_private": bool(owner.is_private),
    }


# ---------- Routes: Groups ----------
def _group_to_out(g: Group) -> GroupOut:
    """Convert Group model to GroupOut response."""
    return GroupOut(
        id=g.id,
        name=g.name,
        avatar=g.avatar,
        owner_id=g.owner_id,
        member_ids=[m.user_id for m in g.members],
    )


@app.post("/groups", response_model=GroupOut)
def create_group(
    body: GroupCreateIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Create a new group (creator is auto-included)."""
    member_ids = set(body.member_ids) | {user.id}
    if len(member_ids) > MAX_GROUP_MEMBERS:
        raise HTTPException(400, f"Group exceeds max of {MAX_GROUP_MEMBERS} members")
    users = db.query(User).filter(User.id.in_(member_ids)).all()
    if len(users) != len(member_ids):
        raise HTTPException(400, "One or more member IDs are invalid")
    g = Group(name=body.name, owner_id=user.id, avatar=body.avatar)
    db.add(g)
    db.flush()
    for uid in member_ids:
        db.add(GroupMember(group_id=g.id, user_id=uid))
    db.commit()
    db.refresh(g)
    return _group_to_out(g)


@app.get("/groups", response_model=List[GroupOut])
def list_my_groups(
    user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """List all groups current user is a member of."""
    gids = [m.group_id for m in db.query(GroupMember).filter_by(user_id=user.id).all()]
    groups = db.query(Group).filter(Group.id.in_(gids)).all() if gids else []
    return [_group_to_out(g) for g in groups]


@app.get("/groups/{group_id}", response_model=GroupOut)
def get_group(
    group_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get group details (must be a member)."""
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Not found")
    if not any(m.user_id == user.id for m in g.members):
        raise HTTPException(403, "Not a member")
    return _group_to_out(g)


@app.patch("/groups/{group_id}", response_model=GroupOut)
def update_group(
    group_id: int,
    body: GroupUpdateIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Update group (owner only)."""
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Not found")
    if g.owner_id != user.id:
        raise HTTPException(403, "Only owner can edit group")
    if body.name is not None:
        g.name = body.name
    if body.avatar is not None:
        g.avatar = body.avatar
    db.commit()
    db.refresh(g)
    return _group_to_out(g)


@app.post("/groups/{group_id}/members", response_model=GroupOut)
def add_member(
    group_id: int,
    body: AddMemberIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Add member to group (owner only)."""
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Not found")
    if g.owner_id != user.id:
        raise HTTPException(403, "Only owner can add members")
    if len(g.members) >= MAX_GROUP_MEMBERS:
        raise HTTPException(
            400, f"Group already at max of {MAX_GROUP_MEMBERS} members"
        )
    if any(m.user_id == body.user_id for m in g.members):
        raise HTTPException(400, "Already a member")
    if not db.get(User, body.user_id):
        raise HTTPException(404, "User not found")
    db.add(GroupMember(group_id=g.id, user_id=body.user_id))
    db.commit()
    db.refresh(g)
    return _group_to_out(g)


@app.delete("/groups/{group_id}/members/{user_id}", response_model=GroupOut)
def remove_member(
    group_id: int,
    user_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Remove member from group (owner only, or self)."""
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Not found")
    if user.id != g.owner_id and user.id != user_id:
        raise HTTPException(403, "Only owner can remove others")
    m = db.query(GroupMember).filter_by(group_id=group_id, user_id=user_id).first()
    if not m:
        raise HTTPException(404, "Member not found")
    db.delete(m)
    db.commit()
    db.refresh(g)
    return _group_to_out(g)


# ---------- Reactions helpers ----------
def _reactions_for(db: Session, m: Message) -> List[ReactionOut]:
    """Get reactions for a message (aggregated across group copies)."""
    # For group messages with a client_msg_id, aggregate reactions across all
    # per-member copies so every recipient sees the same reaction counts.
    if m.group_id is not None and m.client_msg_id:
        rs = (
            db.query(Reaction)
            .filter(Reaction.client_msg_id == m.client_msg_id)
            .all()
        )
    else:
        rs = db.query(Reaction).filter(Reaction.message_id == m.id).all()
    seen: Set[tuple] = set()
    out: List[ReactionOut] = []
    for r in rs:
        key = (r.user_id, r.emoji)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ReactionOut(
                id=r.id,
                message_id=r.message_id,
                client_msg_id=r.client_msg_id,
                user_id=r.user_id,
                emoji=r.emoji,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )
    return out


def _msg_out(db: Session, m: Message) -> MessageOut:
    """Convert Message model to MessageOut response."""
    return MessageOut(
        id=m.id,
        sender_id=m.sender_id,
        recipient_id=m.recipient_id,
        group_id=m.group_id,
        kind=m.kind,
        ciphertext=None if m.deleted else m.ciphertext,
        client_msg_id=m.client_msg_id,
        created_at=m.created_at.isoformat() if m.created_at else "",
        edited_at=m.edited_at.isoformat() if m.edited_at else None,
        deleted=bool(m.deleted),
        reactions=_reactions_for(db, m),
    )


# ---------- Routes: Messaging ----------
@app.post("/messages/dm", response_model=MessageOut)
async def send_dm(
    body: SendDMIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Send a direct message."""
    recipient = db.get(User, body.recipient_id)
    if not recipient:
        raise HTTPException(404, "Recipient not found")
    # Private accounts: must be mutually connected to DM
    if recipient.is_private and not _is_connected(db, user.id, body.recipient_id):
        raise HTTPException(
            403, "You must be connected to message this private account"
        )
    m = Message(
        sender_id=user.id,
        recipient_id=body.recipient_id,
        group_id=None,
        kind=body.kind,
        ciphertext=body.ciphertext,
        client_msg_id=body.client_msg_id,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    out = _msg_out(db, m)
    payload = {"type": "message", "data": out.model_dump()}
    await hub.deliver(body.recipient_id, payload)
    if body.recipient_id != user.id:
        await hub.deliver(user.id, payload)
    return out


@app.post("/messages/group", response_model=List[MessageOut])
async def send_group(
    body: SendGroupIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Send a message to a group."""
    g = db.get(Group, body.group_id)
    if not g:
        raise HTTPException(404, "Group not found")
    member_ids = {m.user_id for m in g.members}
    if user.id not in member_ids:
        raise HTTPException(403, "Not a member")
    payload_ids = {p.recipient_id for p in body.payloads}
    if not payload_ids.issubset(member_ids):
        raise HTTPException(400, "Payloads include non-members")
    rows: List[Message] = []
    for p in body.payloads:
        m = Message(
            sender_id=user.id,
            recipient_id=p.recipient_id,
            group_id=g.id,
            kind=body.kind,
            ciphertext=p.ciphertext,
            client_msg_id=body.client_msg_id,
        )
        db.add(m)
        rows.append(m)
    db.commit()
    for m in rows:
        db.refresh(m)
    outs = [_msg_out(db, m) for m in rows]
    notified: Set[int] = set()
    for o in outs:
        await hub.deliver(
            o.recipient_id, {"type": "message", "data": o.model_dump()}
        )
        notified.add(o.recipient_id)
    # deliver one copy back to sender so their other devices see it
    if user.id not in notified and outs:
        await hub.deliver(user.id, {"type": "message", "data": outs[0].model_dump()})
    return outs


@app.get("/messages/dm/{user_id}", response_model=List[MessageOut])
def get_dm_history(
    user_id: int,
    limit: int = 200,
    offset: int = 0,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get DM history with a specific user."""
    limit = min(limit, 500)  # cap to prevent abuse
    q = (
        db.query(Message)
        .filter(
            Message.group_id.is_(None),
            ((Message.sender_id == user.id) & (Message.recipient_id == user_id))
            | ((Message.sender_id == user_id) & (Message.recipient_id == user.id)),
        )
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_msg_out(db, m) for m in q]


@app.get("/messages/group/{group_id}", response_model=List[MessageOut])
def get_group_history(
    group_id: int,
    limit: int = 500,
    offset: int = 0,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Get group message history (must be member)."""
    limit = min(limit, 1000)  # cap to prevent abuse
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Group not found")
    if not any(m.user_id == user.id for m in g.members):
        raise HTTPException(403, "Not a member")
    q = (
        db.query(Message)
        .filter(Message.group_id == group_id, Message.recipient_id == user.id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_msg_out(db, m) for m in q]


def _fanout_targets_for_message(db: Session, m: Message) -> List[Message]:
    """Find all message copies for edit/delete operations."""
    if m.group_id is not None and m.client_msg_id:
        return (
            db.query(Message)
            .filter(
                Message.group_id == m.group_id,
                Message.client_msg_id == m.client_msg_id,
            )
            .all()
        )
    if m.group_id is None and m.client_msg_id:
        return (
            db.query(Message)
            .filter(
                Message.group_id.is_(None),
                Message.client_msg_id == m.client_msg_id,
                Message.sender_id == m.sender_id,
            )
            .all()
        )
    return [m]


@app.patch("/messages/{message_id}", response_model=List[MessageOut])
async def edit_message(
    message_id: int,
    body: EditMessageIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Edit message (sender only)."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    if m.sender_id != user.id:
        raise HTTPException(403, "Only sender can edit")
    if m.deleted:
        raise HTTPException(400, "Cannot edit a deleted message")
    # verify user is still a participant (group member or DM party)
    if m.group_id is not None:
        g = db.get(Group, m.group_id)
        if not g or not any(mem.user_id == user.id for mem in g.members):
            raise HTTPException(403, "Not a member of this group")

    targets = _fanout_targets_for_message(db, m)
    by_recipient = {t.recipient_id: t for t in targets}
    now = datetime.now(timezone.utc)
    updated: List[Message] = []
    for p in body.payloads:
        t = by_recipient.get(p.recipient_id)
        if not t:
            continue
        t.ciphertext = p.ciphertext
        t.edited_at = now
        updated.append(t)
    db.commit()
    for t in updated:
        db.refresh(t)

    outs = [_msg_out(db, t) for t in updated]
    for o in outs:
        await hub.deliver(
            o.recipient_id, {"type": "message_edited", "data": o.model_dump()}
        )
    return outs


@app.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Delete message (sender only)."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    if m.sender_id != user.id:
        raise HTTPException(403, "Only sender can delete")
    targets = _fanout_targets_for_message(db, m)
    for t in targets:
        t.deleted = True
        t.ciphertext = ""
    db.commit()
    for t in targets:
        await hub.deliver(
            t.recipient_id,
            {
                "type": "message_deleted",
                "data": {
                    "id": t.id,
                    "client_msg_id": t.client_msg_id,
                    "group_id": t.group_id,
                    "recipient_id": t.recipient_id,
                    "sender_id": t.sender_id,
                },
            },
        )
    return {"ok": True, "deleted": len(targets)}


# ---------- Reactions ----------
async def _broadcast_reaction_change(db: Session, m: Message):
    """Broadcast reaction changes to all conversation participants."""
    if m.group_id is not None:
        g = db.get(Group, m.group_id)
        recipients = [mem.user_id for mem in g.members] if g else []
    else:
        recipients = list({m.sender_id, m.recipient_id})
    rxs = [r.model_dump() for r in _reactions_for(db, m)]
    payload = {
        "type": "reactions",
        "data": {
            "message_id": m.id,
            "client_msg_id": m.client_msg_id,
            "group_id": m.group_id,
            "reactions": rxs,
        },
    }
    for uid in recipients:
        await hub.deliver(uid, payload)


@app.post("/messages/{message_id}/reactions", response_model=List[ReactionOut])
async def add_reaction(
    message_id: int,
    body: ReactionIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Add emoji reaction to message."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    if m.group_id is not None:
        g = db.get(Group, m.group_id)
        if not g or not any(mem.user_id == user.id for mem in g.members):
            raise HTTPException(403, "Not a member")
    else:
        if user.id not in (m.sender_id, m.recipient_id):
            raise HTTPException(403, "Not a participant")
    existing = (
        db.query(Reaction)
        .filter_by(message_id=m.id, user_id=user.id, emoji=body.emoji)
        .first()
    )
    if not existing:
        r = Reaction(
            message_id=m.id,
            client_msg_id=m.client_msg_id,
            user_id=user.id,
            emoji=body.emoji,
        )
        db.add(r)
        db.commit()
    await _broadcast_reaction_change(db, m)
    return _reactions_for(db, m)


@app.delete("/messages/{message_id}/reactions/{emoji}", response_model=List[ReactionOut])
async def remove_reaction(
    message_id: int,
    emoji: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Remove emoji reaction from message."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    db.query(Reaction).filter_by(
        message_id=m.id, user_id=user.id, emoji=emoji
    ).delete()
    db.commit()
    await _broadcast_reaction_change(db, m)
    return _reactions_for(db, m)


@app.get("/messages/{message_id}/reactions", response_model=List[ReactionOut])
def list_reactions(
    message_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List reactions on a message."""
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    return _reactions_for(db, m)


# ---------- WebSocket Hub (realtime + call signaling) ----------
class Hub:
    """In-memory connection hub for authenticated users."""

    def __init__(self):
        self.connections: Dict[int, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, user_id: int, ws: WebSocket):
        """Register a WebSocket connection for a user."""
        async with self.lock:
            self.connections.setdefault(user_id, set()).add(ws)

    async def disconnect(self, user_id: int, ws: WebSocket):
        """Unregister a WebSocket connection."""
        async with self.lock:
            if user_id in self.connections:
                self.connections[user_id].discard(ws)
                if not self.connections[user_id]:
                    del self.connections[user_id]

    async def deliver(self, user_id: int, msg: dict):
        """Send a message to all WebSocket connections for a user."""
        async with self.lock:
            sockets = list(self.connections.get(user_id, set()))
        dead = []
        for ws in sockets:
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket: {e}")
                dead.append(ws)
        for ws in dead:
            await self.disconnect(user_id, ws)

    def is_online(self, user_id: int) -> bool:
        """Check if user has any active connections."""
        return user_id in self.connections


hub = Hub()


CALL_EVENTS = {
    "call",
    "call_invite", "call_accept", "call_reject", "call_cancel", "call_end",
    "call_offer", "call_answer", "call_ice",
}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str):
    """WebSocket endpoint for authenticated users (messaging, typing, calls, etc.)."""
    try:
        uid = decode_token(token)
    except HTTPException:
        try:
            await ws.close(code=4401)
        except Exception:
            pass
        return
    try:
        await ws.accept()
    except Exception as e:
        logger.warning(f"WebSocket accept failed: {e}")
        return

    await hub.connect(uid, ws)
    try:
        await ws.send_text(json.dumps({"type": "ready", "user_id": uid}))
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WebSocket receive error: {e}")
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype in CALL_EVENTS:
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(
                        target,
                        {
                            "type": mtype,
                            "from": uid,
                            "kind": msg.get("kind", "audio"),
                            "call_id": msg.get("call_id"),
                            "payload": msg.get("payload"),
                        },
                    )
            elif mtype == "typing":
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(
                        target,
                        {
                            "type": "typing",
                            "from": uid,
                            "group_id": msg.get("group_id"),
                            "is_typing": bool(msg.get("is_typing", True)),
                        },
                    )
            elif mtype == "read":
                target = msg.get("to")
                if isinstance(target, int):
                    await hub.deliver(
                        target,
                        {
                            "type": "read",
                            "from": uid,
                            "message_ids": msg.get("message_ids", []),
                        },
                    )
            elif mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

            # ── Screenshot / Screen-record detection ──
            # Client fires this when it detects a capture event.
            # Server fans out an alert to everyone in the same conversation.
            elif mtype in ("screenshot", "screenrecord"):
                db_ws = SessionLocal()
                try:
                    ctx_type = msg.get("ctx_type")   # "dm" | "group" | "room"
                    ctx_id   = msg.get("ctx_id")     # user_id / group_id / room_token
                    sender_user = db_ws.get(User, uid)
                    sender_name = sender_user.username if sender_user else f"User {uid}"
                    alert = {
                        "type": mtype,          # "screenshot" or "screenrecord"
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
                    elif ctx_type == "room" and isinstance(ctx_id, str):
                        # broadcast via room hub
                        await room_hub.broadcast(ctx_id, alert)
                        recipients = set()
                    for rid in recipients:
                        await hub.deliver(rid, alert)
                except Exception as e:
                    logger.warning(f"Screenshot/screenrecord handling error: {e}")
                finally:
                    db_ws.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await hub.disconnect(uid, ws)


# ═══════════════════════════════════════════════════════
#  ROOM HUB  — in-memory connections keyed by room token
# ═══════════════════════════════════════════════════════
class RoomHub:
    """In-memory connection hub for anonymous rooms."""

    def __init__(self):
        self.rooms: Dict[str, Dict[str, WebSocket]] = {}  # token → {nickname: ws}
        self.lock = asyncio.Lock()

    async def join(self, token: str, nickname: str, ws: WebSocket):
        """Add a user to a room."""
        async with self.lock:
            self.rooms.setdefault(token, {})[nickname] = ws

    async def leave(self, token: str, nickname: str):
        """Remove a user from a room."""
        async with self.lock:
            if token in self.rooms:
                self.rooms[token].pop(nickname, None)
                if not self.rooms[token]:
                    del self.rooms[token]

    async def broadcast(self, token: str, msg: dict):
        """Broadcast message to all users in a room."""
        async with self.lock:
            sockets = dict(self.rooms.get(token, {}))
        dead = []
        for nick, ws in sockets.items():
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception as e:
                logger.debug(f"Failed to broadcast in room: {e}")
                dead.append(nick)
        for nick in dead:
            await self.leave(token, nick)

    def member_count(self, token: str) -> int:
        """Get number of members in a room."""
        return len(self.rooms.get(token, {}))

    def members(self, token: str) -> List[str]:
        """Get list of member nicknames in a room."""
        return list(self.rooms.get(token, {}).keys())


room_hub = RoomHub()


# ═══════════════════════════════════════════════════════
#  ROOMS — Schemas
# ═══════════════════════════════════════════════════════
class RoomCreateIn(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)


class RoomMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    kind: str = "text"


class RoomOut(BaseModel):
    token: str
    name: Optional[str]
    member_count: int
    max_members: int
    created_at: str


class RoomMessageOut(BaseModel):
    id: int
    sender_nickname: str
    content: str
    kind: str
    created_at: str


# ═══════════════════════════════════════════════════════
#  ROOMS — REST endpoints
# ═══════════════════════════════════════════════════════
@app.post("/rooms", response_model=RoomOut)
def create_room(
    body: RoomCreateIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Create a new ephemeral room."""
    token = secrets.token_urlsafe(16)
    room = Room(
        token=token,
        name=body.name or f"{user.username}'s room",
        owner_id=user.id,
        max_members=10,
        incognito=True,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db.add(room)
    db.commit()
    db.refresh(room)
    return RoomOut(
        token=room.token,
        name=room.name,
        member_count=room_hub.member_count(token),
        max_members=room.max_members,
        created_at=room.created_at.isoformat(),
    )


@app.get("/rooms/{token}", response_model=RoomOut)
def get_room(token: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Get room details."""
    room = db.query(Room).filter_by(token=token).first()
    if not room:
        raise HTTPException(404, "Room not found")
    _check_room_expired(room, db)
    return RoomOut(
        token=room.token,
        name=room.name,
        member_count=room_hub.member_count(token),
        max_members=room.max_members,
        created_at=room.created_at.isoformat(),
    )


@app.get("/rooms/{token}/preview")
def preview_room(token: str, db: Session = Depends(get_db)):
    """No-auth preview for join page."""
    room = db.query(Room).filter_by(token=token).first()
    if not room:
        raise HTTPException(404, "Room not found or deleted")
    _check_room_expired(room, db)
    return {
        "token": room.token,
        "name": room.name,
        "member_count": room_hub.member_count(token),
        "max_members": room.max_members,
        "is_full": room_hub.member_count(token) >= room.max_members,
    }


@app.delete("/rooms/{token}")
def delete_room(
    token: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Delete room (owner only)."""
    room = db.query(Room).filter_by(token=token).first()
    if not room:
        raise HTTPException(404, "Room not found")
    if room.owner_id != user.id:
        raise HTTPException(403, "Only the creator can delete the room")
    db.delete(room)
    db.commit()
    return {"ok": True}


def _check_room_expired(room: Room, db: Session):
    """Check if room has expired and delete it."""
    if room.expires_at and datetime.now(timezone.utc) > room.expires_at.replace(tzinfo=timezone.utc):
        db.delete(room)
        db.commit()
        raise HTTPException(404, "Room has expired and been deleted")


# ═══════════════════════════════════════════════════════
#  ROOM WebSocket — incognito real-time chat
#  ws://host/ws/room/{token}?nickname=Alice
# ═══════════════════════════════════════════════════════
@app.websocket("/ws/room/{token}")
async def room_ws(ws: WebSocket, token: str, nickname: str = "Anonymous"):
    """WebSocket endpoint for anonymous room chat."""
    db = SessionLocal()
    try:
        room = db.query(Room).filter_by(token=token).first()
        if not room:
            try:
                await ws.close(code=4404)
            except Exception:
                pass
            return
        _check_room_expired(room, db)

        nickname = (nickname or "Anonymous")[:32].strip() or "Anonymous"

        # Cap at max members
        if room_hub.member_count(token) >= room.max_members:
            try:
                await ws.close(code=4429)   # too many
            except Exception:
                pass
            return

        try:
            await ws.accept()
        except Exception as e:
            logger.warning(f"Room WebSocket accept failed: {e}")
            return

        await room_hub.join(token, nickname, ws)

        # Announce join
        await room_hub.broadcast(token, {
            "type": "room_join",
            "nickname": nickname,
            "members": room_hub.members(token),
        })

        try:
            await ws.send_text(json.dumps({
                "type": "room_ready",
                "token": token,
                "nickname": nickname,
                "members": room_hub.members(token),
            }))

            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"Room WebSocket receive error: {e}")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                mtype = msg.get("type")

                if mtype == "room_message":
                    content = str(msg.get("content", "")).strip()[:4000]
                    kind = msg.get("kind", "text")
                    if not content:
                        continue
                    now = datetime.now(timezone.utc)
                    # Incognito: store only if room.incognito is False
                    # Default is True so messages are NOT persisted
                    out = {
                        "type": "room_message",
                        "nickname": nickname,
                        "content": content,
                        "kind": kind,
                        "at": now.isoformat(),
                    }
                    await room_hub.broadcast(token, out)

                elif mtype == "room_typing":
                    await room_hub.broadcast(token, {
                        "type": "room_typing",
                        "nickname": nickname,
                        "is_typing": bool(msg.get("is_typing", True)),
                    })

                elif mtype in ("screenshot", "screenrecord"):
                    await room_hub.broadcast(token, {
                        "type": mtype,
                        "from_username": nickname,
                        "ctx_type": "room",
                        "ctx_id": token,
                        "at": datetime.now(timezone.utc).isoformat(),
                    })

                elif mtype == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Room WebSocket error: {e}")
        finally:
            await room_hub.leave(token, nickname)
            await room_hub.broadcast(token, {
                "type": "room_leave",
                "nickname": nickname,
                "members": room_hub.members(token),
            })
            # Auto-delete room if empty and incognito
            if room.incognito and room_hub.member_count(token) == 0:
                try:
                    r = db.query(Room).filter_by(token=token).first()
                    if r:
                        db.delete(r)
                        db.commit()
                except Exception as e:
                    logger.warning(f"Room auto-delete error: {e}")
    finally:
        db.close()


# ---------- Health ----------
@app.get("/")
def root():
    """Health check and service info."""
    return {
        "service": "encrypted-chat-api",
        "status": "ok",
        "max_group_members": MAX_GROUP_MEMBERS,
        "message_kinds": sorted(ALLOWED_KINDS),
    }


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=PORT, reload=False)
