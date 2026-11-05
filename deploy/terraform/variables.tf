variable "environment" {
  description = "dev, staging or prod; names every resource"
  type        = string
  default     = "dev"
}

variable "cloudflare_account_id" {
  description = "Cloudflare account id (dashboard → the account's overview page)"
  type        = string
}

variable "cloudflare_api_token" {
  description = "A Cloudflare API token with 'Workers R2 Storage: Edit' on the account; it creates the bucket"
  type        = string
  sensitive   = true
}

variable "r2_access_key_id" {
  description = "Access key id of an R2 API token (R2 > Manage API tokens) with object read and write on the bucket"
  type        = string
}

variable "r2_secret_access_key" {
  description = "Secret access key of that R2 API token"
  type        = string
  sensitive   = true
}

variable "neon_api_key" {
  description = "Neon API key (console → account settings → API keys)"
  type        = string
  sensitive   = true
}

variable "neon_org_id" {
  description = "Neon organization that owns the project (null for a personal account)"
  type        = string
  default     = null
}

variable "neon_region" {
  description = "Neon region; keep it next to Modal's default region"
  type        = string
  default     = "aws-us-east-1"
}

variable "r2_location" {
  description = "R2 location hint"
  type        = string
  default     = "ENAM"
}

variable "modal_secret_name" {
  description = "The Modal secret the control plane and workers read (deploy with AUTOINFERENCE_SECRET set to this)"
  type        = string
  default     = "autoinference-platform"
}
