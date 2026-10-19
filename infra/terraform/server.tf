# One small Ubuntu server with a static IP. Only SSH from your own address is allowed in;
# FinTray itself listens on 127.0.0.1 and is reached over Tailscale (docs/CLOUD.md), never a public port.

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_key_pair" "admin" {
  key_name_prefix = "${var.project}-"
  public_key      = file(pathexpand(var.ssh_public_key_path))
}

resource "aws_security_group" "server" {
  name_prefix = "${var.project}-"
  description = "FinTray: SSH from the admin address only"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "ssh_admin" {
  security_group_id = aws_security_group.server.id
  description       = "SSH from the admin address"
  cidr_ipv4         = var.admin_cidr
  ip_protocol       = "tcp"
  from_port         = 22
  to_port           = 22
}

resource "aws_vpc_security_group_egress_rule" "all_out" {
  security_group_id = aws_security_group.server.id
  description       = "Outbound: packages, GitHub, broker APIs, S3, Tailscale"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

# The server may read and write its own backup bucket, and nothing else in the account.
data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "backup_access" {
  statement {
    sid       = "ListOwnBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.backups.arn]
  }
  statement {
    sid       = "ReadWriteBackups"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.backups.arn}/*"]
  }
}

resource "aws_iam_role" "server" {
  name_prefix        = "${var.project}-server-"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy" "backup_access" {
  name_prefix = "${var.project}-backups-"
  role        = aws_iam_role.server.id
  policy      = data.aws_iam_policy_document.backup_access.json
}

resource "aws_iam_instance_profile" "server" {
  name_prefix = "${var.project}-"
  role        = aws_iam_role.server.name
}

resource "aws_instance" "server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.admin.key_name
  vpc_security_group_ids = [aws_security_group.server.id]
  iam_instance_profile   = aws_iam_instance_profile.server.name

  metadata_options {
    http_tokens   = "required" # IMDSv2 only: blocks a common credential-theft path
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  tags = {
    Name = var.project
  }

  lifecycle {
    ignore_changes = [ami] # a newer Ubuntu image must not silently replace the running server
  }
}

# Static public IP: brokers require a fixed address for API orders (docs/CLOUD.md).
resource "aws_eip" "server" {
  instance = aws_instance.server.id
  domain   = "vpc"
}
