# Infrastructure for one environment of the platform: an R2 bucket for files, a Neon Postgres
# project for state, an R2-scoped API token, and the Modal secret that hands all of it to the
# control plane and the workers.
#
#   cd deploy/terraform
#   cp terraform.tfvars.example terraform.tfvars   # fill in the account id and API keys
#   terraform init
#   terraform apply
#
# State holds secrets, so it stays local and ignored by git. Move it to a remote backend before
# a second person needs to run this.

terraform {
  required_version = ">= 1.6"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
    neon = {
      source  = "kislerdm/neon"
      version = "~> 0.9"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

provider "neon" {
  api_key = var.neon_api_key
}

locals {
  name = "autoinference-${var.environment}"
}

# --- files: Cloudflare R2 -------------------------------------------------------------------------

resource "cloudflare_r2_bucket" "files" {
  account_id    = var.cloudflare_account_id
  name          = local.name
  location      = var.r2_location
  storage_class = "Standard"
}

# An API token limited to R2 object access. Its S3 credentials derive from it: the access key id
# is the token's id and the secret is the SHA-256 of the token value.
data "cloudflare_api_token_permission_groups_list" "r2_write" {
  name  = "Workers R2 Storage Write"
  scope = "com.cloudflare.api.account"
}

locals {
  r2_write_group = data.cloudflare_api_token_permission_groups_list.r2_write.result[0].id
}

resource "cloudflare_api_token" "r2" {
  name = "${local.name}-r2"
  policies = [{
    effect            = "allow"
    permission_groups = [{ id = local.r2_write_group }]
    resources = jsonencode({
      "com.cloudflare.api.account.${var.cloudflare_account_id}" = "*"
    })
  }]
}

locals {
  r2_endpoint          = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
  r2_access_key_id     = cloudflare_api_token.r2.id
  r2_secret_access_key = sha256(cloudflare_api_token.r2.value)
}

# --- state: Neon Postgres -------------------------------------------------------------------------

resource "neon_project" "db" {
  name                      = local.name
  region_id                 = var.neon_region
  pg_version                = 17
  history_retention_seconds = 21600
}

# --- hand everything to Modal -------------------------------------------------------------------
# Modal has no Terraform provider, so the secret is written with its CLI whenever a value changes.

resource "terraform_data" "modal_secret" {
  triggers_replace = [
    neon_project.db.connection_uri,
    local.r2_secret_access_key,
    cloudflare_r2_bucket.files.name,
    var.environment,
  ]

  provisioner "local-exec" {
    interpreter = ["python", "-c"]
    command     = <<-EOT
      import os, subprocess, sys
      env = {**os.environ, "PYTHONUTF8": "1"}
      values = {
        "AUTOINFERENCE_ENV": "${var.environment}",
        "EXECUTOR": "modal",
        "STORAGE": "r2",
        "DATABASE_URL": os.environ["TF_DATABASE_URL"],
        "R2_BUCKET": "${cloudflare_r2_bucket.files.name}",
        "R2_ENDPOINT": "${local.r2_endpoint}",
        "R2_ACCESS_KEY_ID": "${local.r2_access_key_id}",
        "R2_SECRET_ACCESS_KEY": os.environ["TF_R2_SECRET"],
      }
      args = [sys.executable, "-m", "modal", "secret", "create", "${var.modal_secret_name}", "--force"]
      args += [f"{k}={v}" for k, v in values.items()]
      sys.exit(subprocess.run(args, env=env).returncode)
    EOT
    environment = {
      TF_DATABASE_URL = neon_project.db.connection_uri
      TF_R2_SECRET    = local.r2_secret_access_key
    }
  }
}
