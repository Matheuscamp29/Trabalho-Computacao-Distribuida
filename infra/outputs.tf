output "api" {
  value = aws_instance.api.public_ip
}

output "workers" {
  value = aws_instance.worker[*].public_ip
}

output "workers_rede_interna" {
  value = aws_instance.worker[*].private_ip
}

output "testar" {
  value = "curl http://${aws_instance.api.public_ip}:8000/health"
}
