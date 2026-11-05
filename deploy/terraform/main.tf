# Infrastructure for one environment of the platform: an R2 bucket for files, a Neon Postgres
# project for state, and the Modal secret that hands all of it to the
# control plane and the workers.
#
#   cd deploy/terraform
#   cp terraform.tfvars.example terraform.tfvars   # fill in the account id and API keys
#   terraform init
#   terraform import neon_project.db <project-id>   # only when the project was made with `neon init`
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

# The S3 credentials for the bucket come from an R2 API token made in the dashboard
# (R2 > Manage API tokens, "Object Read & Write" scoped to this bucket). Cloudflare only lets
# tokens be minted through the API by users with token-provisioning rights, so that one step
# stays manual; everything else here is declared.
locals {
  r2_endpoint = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
}

# --- state: Neon Postgres -------------------------------------------------------------------------

resource "neon_project" "db" {
  name                      = local.name
  org_id                    = var.neon_org_id
  region_id                 = var.neon_region
  pg_version                = 18
  history_retention_seconds = 21600
}

# --- hand everything to Modal -------------------------------------------------------------------
# Modal has no Terraform provider, so the secret is written with its CLI whenever a value changes.

resource "terraform_data" "modal_secret" {
  triggers_replace = [
    neon_project.db.connection_uri,
    var.r2_secret_access_key,
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
        "R2_ACCESS_KEY_ID": "${var.r2_access_key_id}",
        "R2_SECRET_ACCESS_KEY": os.environ["TF_R2_SECRET"],
      }
      args = [sys.executable, "-m", "modal", "secret", "create", "${var.modal_secret_name}", "--force"]
      args += [f"{k}={v}" for k, v in values.items()]
      sys.exit(subprocess.run(args, env=env).returncode)
    EOT
    environment = {
      TF_DATABASE_URL = neon_project.db.connection_uri
      TF_R2_SECRET    = var.r2_secret_access_key
    }
  }
}
