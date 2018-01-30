#!/bin/bash

set -eu

export ROOT_DIR=app
export TF_LOGDIR=/var/logs/tensorboard
export TF_PROVIDER_USER=ec2-user
export SOURCES=src

function task_clean {
  rm -rv ./venv
  rm hosts
}

function ensure_venv {
  if [ ! -d venv ]; then
    virtualenv -p python3 venv
    ./venv/bin/pip install -r $SOURCES/requirements.txt
  fi
  set +u
  source ./venv/bin/activate

  # fix for pip to recognize lib64 packages (needed for Amazon Deep Learning Base AMI)
  export PYTHONPATH=venv/lib64/python3.4/dist-packages:$PYTHONPATH

  export LC_ALL=C # set locale for python
  set -u
}

function get_ip {
  cd deploy && terraform output ip
}

function ensure_ansible {
  ensure_venv
  pip install ansible
}

function sync {
  ensure_ssh_key
  echo "Syncing files..."
  rsync -a --include={go,src/***} --exclude="*" . -e "ssh -i $SSH_PRIVATE_KEY" "$TF_PROVIDER_USER"@"$(get_ip)":/home/"$TF_PROVIDER_USER"/app
  echo "..Done!"
}

function provision {
  ensure_ansible
  ensure_ssh_key
  echo "ec2" \
       "ansible_host=$(get_ip)" \
       "ansible_user=$TF_PROVIDER_USER" \
       "ansible_ssh_private_key_file=${SSH_PRIVATE_KEY}" \
       > hosts
  ansible-playbook -i hosts playbook.yml --extra-vars "username=$TF_PROVIDER_USER root_dir=$ROOT_DIR"
}

function ensure_ssh_key {
  if [ -z "${SSH_KEY:-}" ]; then
    echo "Please specify SSH_KEY."
    exit 1
  fi
  export SSH_PRIVATE_KEY=$SSH_KEY
  export SSH_PUBLIC_KEY=${SSH_KEY}.pub
}

function task_deploy {
  task_tf "apply"
  provision
}

function task_local_run {
  ensure_venv
  PYTHONPATH=$(pwd)/src/dqn:$PYTHONPATH python $SOURCES/main.py
}

function task_run {
  ensure_ssh_key
  if [ -z "$(get_ip)" ]; then
    echo "Please deploy your environment (./go deploy)"
    exit 1
  fi
  sync
  ssh -i "$SSH_PRIVATE_KEY" "$TF_PROVIDER_USER"@"$(get_ip)" "cd ${ROOT_DIR} && ./go local-run"
}

function task_tf {
  ensure_ssh_key
  cd deploy/
  terraform "$@" \
    -var "ssh_public_key=$SSH_PUBLIC_KEY" \
    -var "access_key=$(gopass show dev/aws-access-key)" \
    -var "secret_key=$(gopass show dev/aws-secret-key)"
  cd -
}

function task_usage {
  echo "Usage: $0 clean | deploy | local-run | run | tf"
  exit 1
}

function task_tensorboard {
  ensure_venv
  tensorboard --logdir="$TF_LOGDIR"
}

CMD=${1:-}
shift || true
case ${CMD} in
  clean) task_clean ;;
  tensorboard) task_tensorboard ;;
  deploy) task_deploy  "$@" ;;
  local-run) task_local_run ;;
  run) task_run ;;
  tf) task_tf  "$@" ;;
  *) task_usage ;;
esac
