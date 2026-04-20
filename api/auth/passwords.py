from passlib.context import CryptContext

_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _ctx.hash(password, rounds=12)


def verify_password(plain: str, hashed: str) -> bool:
    return _ctx.verify(plain, hashed)
