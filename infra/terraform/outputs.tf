output "static_ip" {
  description = "The server's fixed public IP. Register it with your broker for API orders."
  value       = aws_eip.server.public_ip
}

output "ssh_command" {
  description = "Log in to the server."
  value       = "ssh ubuntu@${aws_eip.server.public_ip}"
}

output "backup_bucket" {
  description = "S3 bucket for FinTray backups. Pass it to Ansible as backup_bucket."
  value       = aws_s3_bucket.backups.bucket
}

output "ansible_inventory" {
  description = "Paste into infra/ansible/inventory.ini."
  value       = <<-EOT
    [fintray]
    ${aws_eip.server.public_ip} ansible_user=ubuntu backup_bucket=${aws_s3_bucket.backups.bucket} backup_region=${var.region}
  EOT
}
