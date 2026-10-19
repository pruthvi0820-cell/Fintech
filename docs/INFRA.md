# FinTray infrastructure as code (Terraform + Ansible)

The same always-on server as [CLOUD.md](CLOUD.md), built from code instead of clicks:

| Layer | Tool | Files | What it does |
|---|---|---|---|
| Infrastructure | Terraform | `infra/terraform/` | Ubuntu 24.04 server, static IP, firewall (SSH from your IP only), encrypted private S3 backup bucket, IAM role so the server can reach only that bucket |
| Configuration | Ansible | `infra/ansible/playbook.yml` | Packages, FinTray code, Python environment, private `.env`, services that restart on failure, nightly S3 backup timer |
| Backups | Python + systemd | `scripts/backup_s3.py`, `deploy/fintray-backup.*` | Consistent copies of the journal, notes and library databases, gzipped, to S3 every night |
| Checks | GitHub Actions | `.github/workflows/checks.yml` | `terraform fmt/validate/test`, `ansible-lint`, `ruff`, `pytest` on every push |

## Safety design

- **No public app port.** The firewall allows only SSH, only from `admin_cidr`; Terraform refuses `0.0.0.0/0`.
  FinTray listens on 127.0.0.1 and is reached over Tailscale (CLOUD.md step 5).
- **No AWS keys on the server.** The server's IAM role allows `ListBucket`, `GetObject` and `PutObject` on the
  backup bucket only.
- **Backups can't leak or vanish quietly:** all public access blocked, encrypted at rest, HTTPS only, versioned
  (old versions kept 30 days), and `terraform destroy` will not delete a bucket that has backups in it.
- **Server hardening:** IMDSv2 required, encrypted disk; a newer Ubuntu image never silently replaces the server.

## Run it (about 20 minutes, free-tier eligible)

You need: an AWS account, the AWS CLI logged in (`aws configure`), Terraform ≥ 1.6, Ansible, and an SSH key.

```bash
# 1. Infrastructure
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars      # set admin_cidr (your IP/32) and your SSH public key path
terraform init
terraform plan                                    # read what will be created
terraform apply
terraform output -raw ansible_inventory > ../ansible/inventory.ini

# 2. Configuration
cd ../ansible
ansible-playbook playbook.yml                     # safe to re-run: changes only what is missing

# 3. Check
ssh ubuntu@$(cd ../terraform && terraform output -raw static_ip)
systemctl --user status fintray-app fintray-backup.timer
systemctl --user start fintray-backup.service && journalctl --user -u fintray-backup -n 20
```

Then continue with CLOUD.md from step 5 (Tailscale, `.env` keys, broker static IP).

**Tear down** (stops all charges except stored backups): `terraform destroy`. It stops at the bucket if it holds
backups: download what you need, empty it in the S3 console, and run `destroy` again.

## Tests without an AWS account

```bash
cd infra/terraform && terraform init -backend=false && terraform test     # mock AWS provider
cd infra/ansible && ansible-lint playbook.yml
pytest tests/test_backup.py tests/test_deploy.py
```

The Terraform tests check that SSH is open to your address only, that the server requires IMDSv2 and has an
encrypted disk, that the bucket is private, encrypted and versioned, and that unsafe inputs (`0.0.0.0/0`, a
malformed IP, an oversized disk) are refused.

**Renaming the Git branch?** Change it in both `deploy/setup_server.sh` (`BRANCH=`) and the playbook
(`repo_branch:`). A test fails if they differ.
