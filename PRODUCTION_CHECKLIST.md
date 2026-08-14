# LeafGuard77 Production Hardening

Implemented in this build:
- SECRET_KEY from environment; production refuses the demo fallback/short secrets.
- Security response headers and HSTS in production.
- Per-process rate limiting for login/register and prediction endpoints. Use Redis/shared limiting when running multiple replicas.
- Upload size/type validation.
- .env excluded from Git.
- Render start configuration included.
- Health endpoint exposes deployment-safe service metadata.
- Demo account is created only outside production.

Still recommended before a real multi-user agricultural service:
1. Move SQLite to managed PostgreSQL and persist uploaded images in object storage.
2. Use Redis-backed distributed rate limiting.
3. Add centralized logging/monitoring and alerting.
4. Independently evaluate the model on representative real-world images; do not claim dataset accuracy as field accuracy.
5. Have treatment guidance reviewed against local agricultural extension/regulatory guidance.
6. Put the service behind HTTPS and a managed reverse proxy/WAF (Render provides HTTPS for its web services).
