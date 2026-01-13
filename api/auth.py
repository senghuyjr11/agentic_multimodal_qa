"""
auth.py - Authentication endpoints

Provides:
- User registration
- User login
- Current user info
- Simple JSON-based user storage
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from pydantic import BaseModel
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()

# Router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Users file
USERS_FILE = "users.json"


# ===== Models =====

class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str


class UserInfo(BaseModel):
    username: str
    email: Optional[str]
    created_at: str


# ===== Helper Functions =====

def load_users():
    """Load users from JSON file"""
    if not os.path.exists(USERS_FILE):
        return {}

    with open(USERS_FILE, 'r') as f:
        return json.load(f)


def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# ===== Dependency =====

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Get current authenticated user from JWT token

    Returns:
        username (str)
    """
    token = credentials.credentials
    payload = decode_token(token)
    username = payload.get("sub")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify user still exists
    users = load_users()
    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return username


# ===== Endpoints =====

@router.post("/register", response_model=Token)
async def register(user: UserRegister):
    """
    Register a new user

    Returns JWT access token
    """
    users = load_users()

    # Check if username exists
    if user.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Validate username
    if len(user.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )

    # Validate password
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )

    # Create user
    users[user.username] = {
        "username": user.username,
        "password_hash": get_password_hash(user.password),
        "email": user.email,
        "created_at": datetime.utcnow().isoformat()
    }

    save_users(users)

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }


@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """
    Login user

    Returns JWT access token
    """
    users = load_users()

    # Check if user exists
    if user.username not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    user_data = users[user.username]

    # Verify password
    if not verify_password(user.password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }


@router.get("/me", response_model=UserInfo)
async def get_me(username: str = Depends(get_current_user)):
    """
    Get current user info

    Requires authentication
    """
    users = load_users()
    user_data = users[username]

    return {
        "username": user_data["username"],
        "email": user_data.get("email"),
        "created_at": user_data["created_at"]
    }


@router.get("/verify")
async def verify_token(username: str = Depends(get_current_user)):
    """
    Verify if token is valid

    Returns username if valid
    """
    return {"valid": True, "username": username}


# ===== Optional: List all users (admin only - implement your own auth) =====

@router.get("/users")
async def list_users():
    """
    List all registered users (for debugging)

    WARNING: Remove this endpoint in production or add proper authorization!
    """
    users = load_users()

    return {
        "count": len(users),
        "users": [
            {
                "username": u["username"],
                "email": u.get("email"),
                "created_at": u["created_at"]
            }
            for u in users.values()
        ]
    }


if __name__ == "__main__":
    # Test
    print("Testing auth system...")

    # Create test user
    users = {}
    users["test"] = {
        "username": "test",
        "password_hash": get_password_hash("test123"),
        "email": "test@example.com",
        "created_at": datetime.utcnow().isoformat()
    }

    save_users(users)
    print("✓ Created test user: test / test123")

    # Test token
    token = create_access_token(data={"sub": "test"})
    print(f"✓ Generated token: {token[:50]}...")

    # Verify token
    payload = decode_token(token)
    print(f"✓ Token valid for user: {payload['sub']}")