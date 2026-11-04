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
  description = "A Cloudflare API token that can create R2 buckets and API tokens (template: 'Edit Cloudflare Workers' plus 'API Tokens: Edit', or a custom token with Workers R2 Storage Write and API Tokens Write)"
  type        = string
  sensitive   = true
}

variable "neon_api_key" {
  description = "Neon API key (console → account settings → API keys)"
  type        = string
  sensitive   = true
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
