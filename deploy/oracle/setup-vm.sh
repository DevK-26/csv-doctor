#!/usr/bin/env bash
# CSV Doctor — Oracle Cloud VM one-time setup
# Run ONCE on a fresh Ubuntu 24.04 / 22.04 ARM (Ampere A1) VM, as ubuntu user:
#   bash setup-vm.sh
# Idempotent: safe to run again.
set -euo pipefail

echo "==> 1/6 System update + basics"
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y curl git ufw fail2ban unattended-upgrades

echo "==> 2/6 Swap (2GB — protects the build on small boots)"
if ! sudo swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==> 3/6 Docker (official convenience script) + Compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker

echo "==> 4/6 Firewall — SSH + HTTP/HTTPS only"
# Oracle also has a cloud-side Security List — ports 80/443 must be opened
# there too (see DEPLOY-ORACLE.md Step 3). This is the OS-side firewall.
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw --force enable
# Oracle Ubuntu images ship iptables rules that block non-SSH traffic:
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT || true

echo "==> 5/6 App folder"
mkdir -p ~/csv-doctor
sudo chown -R "$USER":"$USER" ~/csv-doctor

echo "==> 6/6 Done"
echo "Log out and back in once (so the docker group applies), then run deploy.sh"
docker --version || true
docker compose version || true
