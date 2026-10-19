# Offline tests: a mock AWS provider, so no account or network is needed.
#   terraform test   (or: tofu test)

mock_provider "aws" {
  mock_data "aws_ami" {
    defaults = { id = "ami-0123456789abcdef0" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
}

variables {
  admin_cidr          = "203.0.113.7/32"
  ssh_public_key_path = "tests/test_key.pub"
}

run "only_ssh_from_the_admin_address" {
  command = plan

  assert {
    condition     = aws_vpc_security_group_ingress_rule.ssh_admin.cidr_ipv4 == "203.0.113.7/32"
    error_message = "SSH must be open to the admin address only."
  }
  assert {
    condition     = aws_vpc_security_group_ingress_rule.ssh_admin.from_port == 22 && aws_vpc_security_group_ingress_rule.ssh_admin.to_port == 22
    error_message = "Only port 22 may be opened."
  }
}

run "server_is_hardened" {
  command = plan

  assert {
    condition     = aws_instance.server.metadata_options[0].http_tokens == "required"
    error_message = "IMDSv2 must be required."
  }
  assert {
    condition     = aws_instance.server.root_block_device[0].encrypted == true
    error_message = "The disk must be encrypted."
  }
  assert {
    condition     = aws_instance.server.instance_type == "t3.micro"
    error_message = "Default size should stay free-tier eligible."
  }
}

run "backups_are_private_encrypted_and_versioned" {
  command = plan

  assert {
    condition = alltrue([
      aws_s3_bucket_public_access_block.backups.block_public_acls,
      aws_s3_bucket_public_access_block.backups.block_public_policy,
      aws_s3_bucket_public_access_block.backups.ignore_public_acls,
      aws_s3_bucket_public_access_block.backups.restrict_public_buckets,
    ])
    error_message = "All four public-access blocks must be on."
  }
  assert {
    condition     = aws_s3_bucket_versioning.backups.versioning_configuration[0].status == "Enabled"
    error_message = "Versioning must be on."
  }
  assert {
    condition     = one(aws_s3_bucket_server_side_encryption_configuration.backups.rule).apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    error_message = "Backups must be encrypted at rest."
  }
  assert {
    condition     = aws_s3_bucket.backups.force_destroy == false
    error_message = "Destroying the stack must not delete backups."
  }
}

run "refuses_ssh_from_the_whole_internet" {
  command = plan
  variables {
    admin_cidr = "0.0.0.0/0"
  }
  expect_failures = [var.admin_cidr]
}

run "refuses_a_bad_cidr" {
  command = plan
  variables {
    admin_cidr = "my-ip"
  }
  expect_failures = [var.admin_cidr]
}

run "refuses_an_oversized_disk" {
  command = plan
  variables {
    root_volume_gb = 100
  }
  expect_failures = [var.root_volume_gb]
}
