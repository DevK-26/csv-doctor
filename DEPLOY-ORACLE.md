# CSV Doctor on Oracle VM — Full Automation Guide

Confirmed target (2026-11-08): **CSV Doctor → Oracle Cloud VM, fully automated.**

## What is already built and verified

In the repo (new files, ready for you to commit):

| File | What it does |
|---|---|
| `docker-compose.yml` | Whole stack in one command: Caddy + Web (Next.js) + Engine (FastAPI) + Redis |
| `deploy/caddy/Caddyfile` | Only public door. `/api/*` → engine, everything else → web. Auto-HTTPS when a domain is set |
| `apps/web/Dockerfile` + `tsconfig.json` + `next.config.mjs` | Web app can now actually build in Docker (these were missing in Day 1 skeleton) |
| `deploy/oracle/setup-vm.sh` | One-time VM setup: updates, Docker, firewall, fail2ban, 2GB swap |
| `deploy/scripts/deploy.sh` | Deploy/update in one command + automatic health check (`/api/health`) |
| `.github/workflows/deploy.yml` | Push to `main` → engine tests run → SSH into VM → auto-deploy |
| `.env.example` | Only setting needed: `DOMAIN` (blank = IP-only mode) |

Verified here: engine runs on the sample data (sales.csv → 183 rows, health score 98), all scripts pass bash syntax check, both YAML files parse.
**Not verified here:** a live `docker compose up` — Docker is not installed in my sandbox VM, so the first live test is the health check built into `deploy.sh` on your Oracle VM. If it fails, it prints the logs automatically.

Honest note on Claude: the deployment automation does **not** need Claude — it is scripted above. Your Claude subscription is useful for the *next build work* (features). When you want that, the login route is: I open claude.ai in my browser, you log in yourself through the secure card / takeover (I never see your password), and we work from there. Don't share your password in chat.

---

## Step 0 — Code must be on GitHub first (blocker if not done)

Your Day 1 steps were never confirmed: public repo `csv-doctor` under **DevK-26**, first commit/push by you, `pytest` = 3 passed.

If the repo doesn't exist yet, do Day 1 first (DAY1.md), then add these new deploy files and commit them yourself:

```
git add docker-compose.yml deploy apps/web/Dockerfile apps/web/tsconfig.json apps/web/next.config.mjs .github .env.example DEPLOY-ORACLE.md
git commit -m "Deploy: Oracle VM automation (Docker Compose, Caddy, GitHub Actions)"
git push
```

## Step 1 — Oracle Cloud account

1. Go to **cloud.oracle.com** → Sign Up for Oracle Cloud.
2. Email you control + a debit/credit card for identity verification (Always Free tier — a small temporary hold, not a charge for free resources).
3. **Home region: Mumbai (ap-mumbai-1)** — closest to Meerut, lowest latency. Region cannot be changed later.

If you already have an account, skip to Step 2 and tell me — I can do Steps 2–5 with you logged in via browser takeover.

## Step 2 — Create the VM (exact settings)

Console → ☰ → Compute → Instances → **Create instance**

- Name: `csv-doctor`
- Image: **Canonical Ubuntu 24.04**
- Shape: Change shape → **Ampere** → `VM.Standard.A1.Flex` → **4 OCPU, 24 GB RAM** (Always Free eligible)
  - If it says "out of capacity": try 2 OCPU / 12 GB, or a different Availability Domain. This is common for the free ARM shape — retry, don't pay.
- Boot volume: 50 GB
- Networking: create/select the default VCN + public subnet, **assign a public IPv4**
- SSH keys: **Generate a key pair for me** → download BOTH keys and keep them safe, OR paste your own public key. You will need the private key to log in and for GitHub Actions.

Create → wait until state = **RUNNING** → copy the **Public IP**.

## Step 3 — Open the doors (most people get stuck here)

Two firewalls, both must allow 80/443:

1. **Cloud side:** Instance details → Subnet → Default Security List → Add Ingress Rules:
   - Source `0.0.0.0/0`, TCP, Destination port **80**
   - Source `0.0.0.0/0`, TCP, Destination port **443**
   - (Port 22/SSH is already there by default)
2. **OS side:** handled automatically by `setup-vm.sh` in Step 4.

## Step 4 — Set up + deploy (3 commands)

From your laptop (replace IP):

```bash
ssh -i <your-key> ubuntu@<VM-PUBLIC-IP>
git clone https://github.com/DevK-26/csv-doctor.git ~/csv-doctor
cd ~/csv-doctor && bash deploy/oracle/setup-vm.sh
# log out and back in once (Docker group), then:
cp .env.example .env
bash deploy/scripts/deploy.sh
```

Success looks like: `HEALTH OK: {"ok": true, ...}` and the site loading at `http://<VM-PUBLIC-IP>` with the API at `http://<VM-PUBLIC-IP>/api/health`.

## Step 5 — Turn on push-to-deploy

1. On the VM, create a dedicated deploy key: `ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""`, add its `.pub` to `~/.ssh/authorized_keys`.
2. In GitHub repo → Settings → Secrets and variables → Actions, add:
   - `ORACLE_HOST` = VM public IP
   - `ORACLE_USER` = `ubuntu`
   - `ORACLE_SSH_KEY` = the private key from step 1 (the non-.pub file's contents)
3. Push anything to `main` → watch the **Actions** tab: tests → SSH → deploy → health check.

## Step 6 (later) — Domain + HTTPS

Buy/point a domain's A record to the VM IP, set `DOMAIN=your-domain.com` in `.env` on the VM, re-run `deploy.sh`. Caddy gets the HTTPS certificate automatically — no certbot, no manual renewal.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Site times out, SSH works | Step 3 cloud Security List — 80/443 not open |
| `docker: permission denied` | You didn't log out/in after setup-vm.sh |
| Actions deploy fails at SSH | Wrong key in `ORACLE_SSH_KEY` secret, or IP changed |
| Health check fails | deploy.sh already prints the last 80 log lines — send me that screenshot/output |
