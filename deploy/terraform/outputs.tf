output "r2_bucket" {
  value = cloudflare_r2_bucket.files.name
}

output "r2_endpoint" {
  value = local.r2_endpoint
}

output "database_url" {
  value     = neon_project.db.connection_uri
  sensitive = true
}

output "modal_secret" {
  value = var.modal_secret_name
}
