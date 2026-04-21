# Secure Sign-In Feature Design

**Date:** 2026-04-19
**Status:** Approved

## Summary

Add multi-user authentication to the PDF RAG web app. Each user has a fully isolated document library. Users can register with email/password or sign in via Google/GitHub OAuth. The backend is FastAPI; auth state is persisted in SQLite (local) / PostgreSQL (AWS prod).

---

## 1. Architecture Overview

### New Components

| Path | Purpose |
|---|---|
| `api/routers/auth.py` | Auth endpoints (register, login, refresh, logout, OAuth) |
| `api/auth/tokens.py` | JWT creation and verification |
| `api/auth/passwords.py` | bcrypt hashing via passlib |
| `api/auth/oauth.py` | Google and GitHub OAuth clients via authlib |
| `api/dependencies.py` | `get_current_user` FastAPI dependency |
| `db/` | SQLAlchemy models and session management |
| `api/static/login.html` | Login/register UI page |
| `api/static/login.js` | Frontend auth logic |

### Data Isolation

Every user is assigned a UUID at registration. All processed artifacts and vector store data are scoped under:

```
data/processed/{user_id}/
data/embedded/{user_id}/
```

All document and query endpoints filter exclusively by the authenticated user's `user_id` extracted from the JWT.

### Auth Flow

1. User registers (email+password) or signs in via Google/GitHub OAuth.
2. Server issues a short-lived JWT access token (15 min) in the response body + a long-lived refresh token (7 days) in an `httponly` cookie.
3. Frontend stores access token in a module-level JS variable (not `localStorage` — prevents XSS theft). Sends `Authorization: Bearer <token>` on every API call.
4. `get_current_user` FastAPI dependency decodes and validates the JWT on every protected request.

---

## 2. Database & User Model

**Engine:** SQLite locally (`data/auth.db`), configurable via `DATABASE_URL` env var (swap to `postgresql://...` for RDS in prod). SQLAlchemy Core.

### `users` table

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | primary key |
| `email` | TEXT UNIQUE NOT NULL | lowercased on write |
| `hashed_password` | TEXT NULLABLE | null for OAuth-only accounts |
| `oauth_provider` | TEXT NULLABLE | `"google"` or `"github"` |
| `oauth_sub` | TEXT NULLABLE | provider's user ID |
| `created_at` | TIMESTAMP | server default now() |
| `is_active` | BOOLEAN | soft disable without deletion |

### `refresh_tokens` table

| Column | Type | Notes |
|---|---|---|
| `token_hash` | TEXT PRIMARY KEY | SHA-256 of the raw token |
| `user_id` | UUID FK → users.id | |
| `expires_at` | TIMESTAMP | |

### New `Settings` fields

```
DATABASE_URL = "sqlite:///data/auth.db"   # local default
JWT_SECRET_KEY = "<required, no default>" # app fails to start if missing
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7
GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
HTTPS_ONLY = false  # set true in prod to enable Secure cookie flag
```

---

## 3. API Endpoints

### Auth router (`/auth`, unprotected)

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/register` | email + password → creates user, returns access token + sets refresh cookie |
| `POST` | `/auth/login` | email + password → returns access token + sets refresh cookie |
| `POST` | `/auth/refresh` | reads `httponly` refresh cookie → returns new access token |
| `POST` | `/auth/logout` | revokes refresh token from DB, clears cookie |
| `GET` | `/auth/oauth/google` | redirects to Google OAuth consent screen |
| `GET` | `/auth/oauth/google/callback` | exchanges code → creates/finds user, sets tokens, redirects to `/` |
| `GET` | `/auth/oauth/github` | redirects to GitHub OAuth consent screen |
| `GET` | `/auth/oauth/github/callback` | exchanges code → creates/finds user, sets tokens, redirects to `/` |

### Protected routes (all existing endpoints)

All existing `/documents/*` and `/query` endpoints receive `user: User = Depends(get_current_user)` and use `user.id` to scope all data access.

### Frontend routing

- `GET /login` → serves `login.html`
- `GET /` → app checks token in memory; if missing, calls `/auth/refresh`; if that fails, redirects to `/login`

---

## 4. Frontend Auth Flow

### Login page (`login.html` / `login.js`)

Single page with two tabs (Sign In / Register) plus Google and GitHub OAuth buttons. On success, server returns `{ access_token, expires_in }` in the JSON body and sets the refresh token as an `httponly` cookie.

### Token lifecycle

1. On app load (`app.js`): check if access token exists in memory. If not, call `POST /auth/refresh`. If refresh succeeds, store the new access token. If it fails, redirect to `/login`.
2. All `fetch()` calls use a shared `authedFetch(url, options)` wrapper that injects `Authorization: Bearer <token>`.
3. On any `401` response: attempt one silent token refresh, retry the request. If refresh fails, redirect to `/login`.
4. Logout button: calls `POST /auth/logout`, clears in-memory token, redirects to `/login`.

### OAuth callback

After provider redirect, the server sets the refresh cookie and redirects to `/#token=<access_token>`. `app.js` reads the token from the URL fragment, stores it in memory, then immediately calls `history.replaceState` to clear the fragment from the URL.

---

## 5. Security

| Concern | Approach |
|---|---|
| Password hashing | bcrypt via passlib, cost factor 12, minimum 8 characters |
| JWT signing | HS256 with `JWT_SECRET_KEY` (required env var — app refuses to start if unset) |
| Refresh token storage | SHA-256 hash stored in DB; raw token never persisted |
| OAuth CSRF | `state` nonce generated per redirect, stored in session cookie, validated in callback |
| Cookie flags | `HttpOnly=true`, `SameSite=Lax`, `Secure=true` when `HTTPS_ONLY=true` |
| XSS token protection | Access token in JS memory only, never `localStorage`/`sessionStorage` |
| Rate limiting | 10 requests/minute per IP on `/auth/register` and `/auth/login` (in-memory counter) |
| Email verification | Not in v1 — open self-registration, no verification step |

---

## 6. Out of Scope (v1)

- Email verification on registration
- Password reset / forgot password flow
- Admin dashboard for user management
- Per-document sharing between users
- Multi-instance rate limiting (resets on ECS container restart)
