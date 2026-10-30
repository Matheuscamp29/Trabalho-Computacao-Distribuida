terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.regiao
  profile = var.perfil_aws
}

data "aws_ssm_parameter" "ubuntu" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

data "aws_vpc" "padrao" {
  default = true
}

resource "aws_key_pair" "chave" {
  key_name   = "contador-produtos"
  public_key = file(pathexpand(var.chave_publica))
}

resource "aws_security_group" "contador" {
  name   = "contador-produtos"
  vpc_id = data.aws_vpc.padrao.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.meu_ip]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.meu_ip]
  }

  ingress {
    from_port = 8001
    to_port   = 8001
    protocol  = "tcp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "worker" {
  count                  = var.quantidade_workers
  ami                    = data.aws_ssm_parameter.ubuntu.value
  instance_type          = var.tipo_worker
  key_name               = aws_key_pair.chave.key_name
  vpc_security_group_ids = [aws_security_group.contador.id]

  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/inicializa.sh", {
    repositorio = var.repositorio
    branch      = var.branch
    descricao   = "Worker ${count.index + 1} do contador de produtos"
    comando     = "-m distribuido.worker"
    ambiente    = { PORT = "8001", WORKER_ID = "worker${count.index + 1}" }
  })

  tags = { Name = "contador-worker${count.index + 1}" }
}

resource "aws_instance" "api" {
  ami                    = data.aws_ssm_parameter.ubuntu.value
  instance_type          = var.tipo_api
  key_name               = aws_key_pair.chave.key_name
  vpc_security_group_ids = [aws_security_group.contador.id]

  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/inicializa.sh", {
    repositorio = var.repositorio
    branch      = var.branch
    descricao   = "API do contador de produtos"
    comando     = "api.py"
    ambiente = {
      PORT        = "8000"
      MPLBACKEND  = "Agg"
      WORKER_URLS = join(",", [for w in aws_instance.worker : "http://${w.private_ip}:8001"])
    }
  })

  tags = { Name = "contador-api" }
}
