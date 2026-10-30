#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv git
git clone --branch ${branch} ${repositorio} /opt/app
cd /opt/app
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements-api.txt

cat > /etc/systemd/system/contador.service <<'UNIT'
[Unit]
Description=${descricao}
After=network-online.target

[Service]
WorkingDirectory=/opt/app
%{ for chave, valor in ambiente ~}
Environment=${chave}=${valor}
%{ endfor ~}
ExecStart=/opt/app/venv/bin/python ${comando}
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now contador
