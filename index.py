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
"""

import os
import json
import secrets
import asyncio
import sqlite3
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

# ---------- Config ----------
SECRET_KEY = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24 * 7
DB_PATH = os.environ.get("CHAT_DB", "chat.db")
MAX_GROUP_MEMBERS = 20
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


Base.metadata.create_all(engine)


# ---------- Tiny SQLite migration: add new columns if missing ----------
def _ensure_columns():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        def cols(table):
            cur.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cur.fetchall()}

        u = cols("users")
        if "avatar" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
        if "display_name" not in u:
            cur.execute("ALTER TABLE users ADD COLUMN display_name VARCHAR(120)")

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


_ensure_columns()


# ---------- Auth helpers ----------
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=True)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def make_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(data["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(
    token: str = Depends(oauth2), db: Session = Depends(get_db)
) -> User:
    uid = decode_token(token)
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _norm_username(u: str) -> str:
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
    def _u(cls, v):
        return _norm_username(v)


class LoginIn(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _u(cls, v):
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


class UserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    avatar: Optional[str] = None
    public_key: Optional[str] = None


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


class SendGroupIn(BaseModel):
    group_id: int
    payloads: List[RecipientPayload]
    kind: str = "text"
    client_msg_id: Optional[str] = None


class EditMessageIn(BaseModel):
    payloads: List[RecipientPayload]


class ReactionIn(BaseModel):
    emoji: str = Field(min_length=1, max_length=16)


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
    uname = _norm_username(username)
    user = db.query(User).filter_by(username=uname).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    return TokenOut(
        access_token=make_token(user.id), user_id=user.id, username=user.username
    )


def _user_out(u: User) -> UserOut:
    return UserOut(
        id=u.id,
        username=u.username,
        display_name=u.display_name,
        avatar=u.avatar,
        public_key=u.public_key,
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return _user_out(user)


@app.put("/auth/public-key", response_model=UserOut)
def set_public_key(
    body: PublicKeyIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    user.public_key = body.public_key
    db.commit()
    return _user_out(user)


@app.put("/auth/profile", response_model=UserOut)
def update_profile(
    body: ProfileIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if body.display_name is not None:
        user.display_name = body.display_name.strip() or user.username
    if body.avatar is not None:
        user.avatar = body.avatar
    if body.public_key is not None:
        user.public_key = body.public_key
    db.commit()
    return _user_out(user)


@app.put("/auth/password")
def change_password(
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True, min_length=6),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(old_password, user.password_hash):
        raise HTTPException(401, "Old password incorrect")
    user.password_hash = hash_password(new_password)
    db.commit()
    return {"ok": True}


# ---------- Routes: Users ----------
@app.get("/users", response_model=List[UserOut])
def list_users(
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    query = db.query(User)
    if q:
        query = query.filter(User.username.ilike(f"%{q.strip().lower()}%"))
    return [_user_out(u) for u in query.limit(100).all()]


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    return _user_out(u)


# ---------- Routes: Groups ----------
def _group_to_out(g: Group) -> GroupOut:
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
    gids = [m.group_id for m in db.query(GroupMember).filter_by(user_id=user.id).all()]
    groups = db.query(Group).filter(Group.id.in_(gids)).all()
    return [_group_to_out(g) for g in groups]


@app.get("/groups/{group_id}", response_model=GroupOut)
def get_group(
    group_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
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
    if m.client_msg_id and m.group_id is not None:
        rs = (
            db.query(Reaction)
            .filter(Reaction.client_msg_id == m.client_msg_id)
            .all()
        )
    else:
        rs = db.query(Reaction).filter(Reaction.message_id == m.id).all()
    seen = set()
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
def _validate_kind(kind: str):
    if kind not in ALLOWED_KINDS:
        raise HTTPException(
            400, f"kind must be one of {sorted(ALLOWED_KINDS)}"
        )


@app.post("/messages/dm", response_model=MessageOut)
async def send_dm(
    body: SendDMIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    _validate_kind(body.kind)
    if not db.get(User, body.recipient_id):
        raise HTTPException(404, "Recipient not found")
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
    _validate_kind(body.kind)
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
    for o in outs:
        await hub.deliver(
            o.recipient_id, {"type": "message", "data": o.model_dump()}
        )
    return outs


@app.get("/messages/dm/{user_id}", response_model=List[MessageOut])
def get_dm_history(
    user_id: int,
    limit: int = 200,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    q = (
        db.query(Message)
        .filter(
            Message.group_id.is_(None),
            ((Message.sender_id == user.id) & (Message.recipient_id == user_id))
            | ((Message.sender_id == user_id) & (Message.recipient_id == user.id)),
        )
        .order_by(Message.created_at.asc())
        .limit(limit)
        .all()
    )
    return [_msg_out(db, m) for m in q]


@app.get("/messages/group/{group_id}", response_model=List[MessageOut])
def get_group_history(
    group_id: int,
    limit: int = 500,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    g = db.get(Group, group_id)
    if not g:
        raise HTTPException(404, "Group not found")
    if not any(m.user_id == user.id for m in g.members):
        raise HTTPException(403, "Not a member")
    q = (
        db.query(Message)
        .filter(Message.group_id == group_id, Message.recipient_id == user.id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .all()
    )
    return [_msg_out(db, m) for m in q]


def _fanout_targets_for_message(db: Session, m: Message) -> List[Message]:
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
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    if m.sender_id != user.id:
        raise HTTPException(403, "Only sender can edit")
    if m.deleted:
        raise HTTPException(400, "Cannot edit a deleted message")

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
    m = db.get(Message, message_id)
    if not m:
        raise HTTPException(404, "Message not found")
    return _reactions_for(db, m)


# ---------- WebSocket Hub (realtime + call signaling) ----------
class Hub:
    def __init__(self):
        self.connections: Dict[int, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, user_id: int, ws: WebSocket):
        async with self.lock:
            self.connections.setdefault(user_id, set()).add(ws)

    async def disconnect(self, user_id: int, ws: WebSocket):
        async with self.lock:
            if user_id in self.connections:
                self.connections[user_id].discard(ws)
                if not self.connections[user_id]:
                    del self.connections[user_id]

    async def deliver(self, user_id: int, msg: dict):
        async with self.lock:
            sockets = list(self.connections.get(user_id, set()))
        dead = []
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


CALL_EVENTS = {
    "call",
    "call_invite", "call_accept", "call_reject", "call_cancel", "call_end",
    "call_offer", "call_answer", "call_ice",
}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str):
    try:
        uid = decode_token(token)
    except HTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    await hub.connect(uid, ws)
    try:
        await ws.send_text(json.dumps({"type": "ready", "user_id": uid}))
        while True:
            raw = await ws.receive_text()
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
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(uid, ws)


# ---------- Health ----------
@app.get("/")
def root():
    return {
        "service": "encrypted-chat-api",
        "status": "ok",
        "max_group_members": MAX_GROUP_MEMBERS,
        "message_kinds": sorted(ALLOWED_KINDS),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=PORT, reload=False)