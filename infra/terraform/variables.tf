variable "region" {
  description = "AWS region. Mumbai keeps latency to Indian brokers low."
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Name used for tags and resource names."
  type        = string
  default     = "fintray"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,20}$", var.project))
    error_message = "project must be 3-21 characters: lower-case letters, digits and hyphens, starting with a letter."
  }
}

variable "instance_type" {
  description = "Server size. t3.micro (1 GB) is free-tier eligible but tight for FinTray; t3.small (2 GB) is comfortable."
  type        = string
  default     = "t3.micro"
}

variable "root_volume_gb" {
  description = "Disk size in GB (free tier covers up to 30 GB of gp3)."
  type        = number
  default     = 20

  validation {
    condition     = var.root_volume_gb >= 10 && var.root_volume_gb <= 30
    error_message = "root_volume_gb must be between 10 and 30."
  }
}

variable "admin_cidr" {
  description = "The only address allowed to SSH in, e.g. \"203.0.113.7/32\" (your current public IP). Never 0.0.0.0/0."
  type        = string

  validation {
    condition     = can(cidrhost(var.admin_cidr, 0)) && var.admin_cidr != "0.0.0.0/0"
    error_message = "admin_cidr must be a valid CIDR such as 203.0.113.7/32, and not 0.0.0.0/0 (the whole internet)."
  }
}

variable "ssh_public_key_path" {
  description = "Path to your SSH public key, e.g. ~/.ssh/id_ed25519.pub."
  type        = string
}

variable "backup_retention_days" {
  description = "How long old versions of backup files are kept after being replaced."
  type        = number
  default     = 30

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "backup_retention_days must be between 1 and 365."
  }
}
