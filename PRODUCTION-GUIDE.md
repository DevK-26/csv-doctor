# CSV Doctor — Production Guide (the full working model)

Goal: CSV Doctor live on the public internet as Khushi's OWN product —
own domain, own cloud VM, real backend, CI/CD shipping from GitHub —
built to the same production standard as codescriet.dev, whose stack
was verified live on 2026-11-11 (see §1). Everything here is either
already built in this repo or an explicit step with an owner.

## 1. Reference teardown: codescriet.dev (verified 2026-11-11)

Evidence from the site's own page text + live HTTP/DNS checks:

| Layer | codescriet.dev (observed) | Evidence |
|---|---|---|
| Edge / DNS / TLS | Cloudflare in front (CDN, WAF, HTTPS, HTTP/3) | `Server: cloudflare`, CF-Ray, `alt-svc: h3`, HSTS preload |
| Security headers | Strict CSP, X-Frame-Options DENY, nosniff, Referrer-Policy | response headers |
| Analytics | Cloudflare Web Analytics (privacy-friendly, no cookies) | CSP allows `static.cloudflareinsights.com` |
| Hosting | **Oracle Cloud VM** | site text: "deployed on Oracle Cloud" |
| Web server | **Caddy** (reverse proxy, auto-HTTPS) | site text: "served by Caddy" |
| Shipping | GitHub organisation → deploy pipeline | site text: "shipped through the codescriet GitHub organisation" |
| App stack | React + TypeScript + Node + Prisma + PostgreSQL | site text (web/product pillar) |
| Products | main site + `code.` playground subdomain, events, team, recruitment (Join Us) | page structure |

**The pattern in one line:** Cloudflare → Oracle VM → Caddy → app +
database, shipped from GitHub. CSV Doctor's deploy automation
(built 2026-11-08) already implements this exact pattern.

## 2. CSV Doctor target architecture

```
User ──► Cloudflare (DNS · CDN · WAF · TLS · Web Analytics)
            │  proxied (orange-cloud) A record
            ▼
      Oracle Cloud VM — Ubuntu 24.04, Always Free ARM 4 OCPU / 24 GB
            │  Caddy — only public door (80/443), auto-HTTPS,
            │  /api/* → engine, everything else → web
            ├──► web    (Next.js, internal :3000)
            ├──► engine (FastAPI, internal :7860)
            │       parse → profile → Health Score → anomalies
            │       (ensemble) → drift → dedupe → Ask-Your-Data
            ├──► worker (Celery + Redis — Phase C: big files, async
            │       jobs, live progress over SSE)
            └──► data   (uploads volume, DuckDB for ~1M-row scans;
                        SQLite/Postgres added only when user
                        accounts land — YAGNI until then)
      GitHub Actions ──SSH──► VM: git pull + docker compose up --build
                        (tests must pass before deploy runs)
```

Differences from the club site, on purpose: the engine is Python/FastAPI
(the ML is the product), and Celery/Redis exist because profiling a 1M-row
CSV cannot block a web request the way a normal CRUD page can.

## 3. Gap table — what exists vs what's missing

| Piece | Status |
|---|---|
| Docker Compose stack (Caddy + web + engine + Redis) | ✅ Built 2026-11-08 |
| VM setup script (Docker, firewall, fail2ban, swap) | ✅ Built |
| Deploy script + GitHub Actions (test → SSH deploy) | ✅ Built |
| Engine: profile + Health Score | ✅ Working (Day 1) |
| Engine: anomaly ensemble | ✅ Built by Claude Code 2026-11-11 (1 test fix pending) |
| Engine: benchmark, drift, dedupe, Ask-Your-Data | ⏳ Sprint Phases A2–B |
| GitHub repo push (DevK-26/csv-doctor) | ❌ Repo exists, still empty — Khushi |
| Oracle Cloud account + VM | ❌ Blocked on Shubham signup (account in his name, Khushi as OCI admin) |
| Domain | ❌ Not bought yet — Khushi (payment step) |
| Cloudflare DNS + proxy in front of the VM | ❌ 15-min setup once domain + VM exist — guided |
| Uptime monitor + backups | ❌ Configured at first deploy — guided |

## 4. Build order (each step: owner + done-test)

1. **[Khushi] GitHub push** — repo stops being empty; Actions runs the
   test job on every push from now on. Done-test: files visible at
   github.com/DevK-26/csv-doctor.
2. **[Khushi + Shubham] Oracle account** — his signup (tenancy
   `sy323333`, Mumbai), card Online+International ON first; then he adds
   Khushi as an OCI admin user. Done-test: Khushi can sign in to the
   console herself.
3. **[Guided] VM creation** — Ubuntu 24.04, A1.Flex 4 OCPU/24 GB,
   50 GB boot, public IP, SSH key; open ingress 80/443. Then
   `deploy/oracle/setup-vm.sh` → log out/in → `deploy/scripts/deploy.sh`.
   Done-test: `HEALTH OK` and the site answers on the VM's public IP.
4. **[Khushi] Domain** — buy on **Cloudflare Registrar** (at-cost
   pricing, free WHOIS privacy, DNS already in the right place).
   First choice `csvdoctor.dev` (.dev forces HTTPS — fine, Caddy
   issues certs automatically); fallbacks `.app`, `.io`. Budget
   roughly ₹900–1,300/year (registrar shows the exact price before
   payment — that number wins over any estimate here).
5. **[Guided] Cloudflare wiring** — add site to Cloudflare, A record
   `csvdoctor.dev → <VM public IP>` with proxy ON, `www` CNAME to
   root, SSL/TLS mode **Full (strict)**, set `DOMAIN=csvdoctor.dev`
   in the VM `.env`, re-run deploy. Done-test: `https://csvdoctor.dev`
   loads with a valid padlock and the response carries Cloudflare
   headers — the same fingerprint as codescriet.dev.
6. **[Guided] GitHub secrets** — ORACLE_HOST / ORACLE_USER /
   ORACLE_SSH_KEY (dedicated deploy key on the VM). Done-test: a push
   to `main` deploys itself; nobody SSHes by hand for releases again.
7. **[Guided] Ops basics (30 min, once)** — UptimeRobot free monitor
   on `/api/health` (email alert if the site dies), Cloudflare Web
   Analytics beacon in the web app, log rotation check, and a weekly
   backup script (uploads volume + any DB dump, timestamped, kept on
   the VM; copy to object storage when accounts land).
8. **[Claude Code] Product phases continue** — A2 benchmark → A3 drift
   → B dedupe/Ask-Your-Data → C async scale — each merged only with
   tests green, each auto-deployed by step 6.

## 5. Expert rules baked into this setup

- **One public door.** Only Caddy faces the internet; web/engine/Redis
  ports are internal to the Compose network. Same reason codescriet
  sits behind Cloudflare *and* Caddy — two cheap layers, each doing
  one job.
- **Tests gate deploys.** Actions runs pytest first; a red suite means
  the VM never sees the code.
- **Secrets live in exactly two places:** the VM's `.env` (never in
  Git) and GitHub Actions secrets. The Claude OAuth token stays on the
  build machine, never on the VM.
- **Uploads are hostile input.** Size cap at Caddy, parse in the
  engine with row/column limits, files deleted after analysis unless
  the user explicitly saves them.
- **Measure, don't claim.** Uptime %, p95 latency, rows/sec and the
  anomaly benchmark table in the README all come from monitors and
  benchmark scripts — the same evidence standard as the engine itself.

## 6. Running cost

| Item | Cost |
|---|---|
| Oracle Always Free VM (4 OCPU / 24 GB, Mumbai) | ₹0 — never expires |
| Cloudflare (DNS/CDN/WAF/analytics, free tier) | ₹0 |
| UptimeRobot free tier | ₹0 |
| Domain (~1 year) | ~₹900–1,300 — the only recurring cost |

## 7. Why this shape wins interviews

It is the club platform's production pattern — Cloudflare → Oracle →
Caddy → GitHub-shipped apps — rebuilt under her own brand, with a
harder backend: a real ML engine, async workers, benchmarks and a
test-gated pipeline. "I run my own product on the same architecture
as our college's platform, which I also helped build" is a story
with two live URLs as proof.
