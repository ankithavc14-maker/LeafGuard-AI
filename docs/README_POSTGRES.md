# LeafGuard77 - PostgreSQL Complete Version

This package is the full LeafGuard77 project with the PostgreSQL database layer enabled.

## Setup

1. Create a PostgreSQL database named `leafguard`.
2. Create `.env` in `LeafGuard-AI` using `.env.example` as a template.
3. Set:

`DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/leafguard`

4. Install dependencies:

`python -m pip install -r requirements.txt`

5. Run:

`python -m uvicorn app:app`

Open `http://127.0.0.1:8000`.

The application creates the `users` and `predictions` tables automatically on startup.

Demo account for local development:
- Email: `demo@leafguard.ai`
- Password: `LeafGuard123!`

Do not commit `.env` or real secrets to GitHub.
