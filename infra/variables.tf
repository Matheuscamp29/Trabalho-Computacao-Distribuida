variable "perfil_aws" {
  type    = string
  default = "pessoal"
}

variable "regiao" {
  type    = string
  default = "us-east-1"
}

variable "meu_ip" {
  type = string
}

variable "chave_publica" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "tipo_worker" {
  type    = string
  default = "t3.medium"
}

variable "tipo_api" {
  type    = string
  default = "t3.small"
}

variable "quantidade_workers" {
  type    = number
  default = 2
}

variable "repositorio" {
  type    = string
  default = "https://github.com/Matheuscamp29/Trabalho-Computacao-Distribuida.git"
}

variable "branch" {
  type    = string
  default = "main"
}
