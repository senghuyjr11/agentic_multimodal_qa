"""
auth.py - User authentication system
"""
import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kyojurojr11")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# User storage file
USERS_FILE = "users.json"


# === Models ===

class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


# === Helper Functions ===

def load_users() -> dict:
    """Load users from JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict):
    """Save users to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:  # ← ADD THIS
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:  # ← ADD THIS
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# === Endpoint ===
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    """
    Register a new user.

    - username: unique username (3-20 characters)
    - password: minimum 6 characters
    """

    print(f"[DEBUG] Username: {user.username}")
    print(f"[DEBUG] Password length: {len(user.password)}")
    print(f"[DEBUG] Password: {repr(user.password)}")

    # Load existing users
    users = load_users()

    # Validation
    if len(user.username) < 3 or len(user.username) > 20:
        raise HTTPException(
            status_code=400,
            detail="Username must be between 3 and 20 characters"
        )

    if len(user.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters"
        )

    # Check if username already exists
    if user.username in users:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    # Create new user
    users[user.username] = {
        "username": user.username,
        "hashed_password": hash_password(user.password),
        "created_at": datetime.now().isoformat()
    }

    # Save to file
    save_users(users)

    return {
        "message": "User registered successfully",
        "username": user.username
    }


@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """
    Login and get access token.

    - username: your username
    - password: your password

    Returns JWT bearer token valid for 24 hours.
    """
    # Load users
    users = load_users()

    # Check if user exists
    if user.username not in users:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Get stored user data
    stored_user = users[user.username]

    # Verify password
    if not verify_password(user.password, stored_user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }