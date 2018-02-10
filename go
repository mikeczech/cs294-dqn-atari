#!/bin/bash

set -eu

export ROOT_DIR=app
export TF_LOGDIR=$(pwd)/logs
export TF_PROVIDER_USER=ec2-user
export SOURCES=src

function task_clean {
  rm -rv ./venv
  rm -rv ./recordings
  rm -rv ./checkpoints
  rm -rv ./logs
  rm hosts
}

function ensure_venv {
  if [ ! -d venv ]; then
    virtualenv -p python3 venv
    ./venv/bin/pip install -r $SOURCES/requirements.txt
  fi
  set +u
  source ./venv/bin/activate

  # fix for pip to recognize lib64 packages / CUDA (needed for Amazon Deep Learning Base AMI)
  export PYTHONPATH=venv/lib64/python3.4/dist-packages:$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
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

function task_sync {
  ensure_ssh_key
  echo "Syncing files..."
  rsync -a --include={go,src/***} --exclude="*" . -e "ssh -i $SSH_PRIVATE_KEY" "$TF_PROVIDER_USER"@"$(get_ip)":/home/"$TF_PROVIDER_USER"/app
  # rsync -a -e "ssh -i $SSH_PRIVATE_KEY" "$TF_PROVIDER_USER"@"$(get_ip)":${ROOT_DIR}/recordings .
  # rsync -a -e "ssh -i $SSH_PRIVATE_KEY" "$TF_PROVIDER_USER"@"$(get_ip)":${ROOT_DIR}/checkpoints .
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

function ensure_aws_keys {
  if [ -z "${AWS_ACCESS_KEY:-}" ]; then
    echo "Please specify AWS_ACCESS_KEY."
    exit 1
  fi
  if [ -z "${AWS_SECRET_KEY:-}" ]; then
    echo "Please specify AWS_SECRET_KEY."
    exit 1
  fi
}

function task_deploy {
  task_tf "apply"
  provision
}

function task_train {
  ensure_venv
  mkdir -p recordings
  mkdir -p checkpoints
  mkdir -p logs
  PYTHONPATH=$(pwd)/src/dqn:$PYTHONPATH python $SOURCES/main.py
}

function task_ssh {
  ensure_ssh_key
  if [ -z "$(get_ip)" ]; then
    echo "Please deploy your environment (./go deploy)"
    exit 1
  fi
  ssh -i "$SSH_PRIVATE_KEY" -t "$TF_PROVIDER_USER"@"$(get_ip)" "cd ${ROOT_DIR} && bash -c 'tmux new-session -A -s main'"
}

function task_tf {
  ensure_ssh_key
  ensure_aws_keys
  cd deploy/
  terraform "$@" \
    -var "ssh_public_key=$SSH_PUBLIC_KEY" \
    -var "access_key=$AWS_ACCESS_KEY" \
    -var "secret_key=$AWS_SECRET_KEY"
  cd -
}

function task_usage {
  echo "Usage: $0 clean | deploy | tensorboard | ssh | learn-pong | sync | gpu-usage | tf"
  exit 1
}

function task_tensorboard {
  ensure_venv
  tensorboard --logdir="$TF_LOGDIR"
}

function task_gpu_usage {
  watch -n 1 nvidia-smi
}

CMD=${1:-}
shift || true
case ${CMD} in
  clean) task_clean ;;
  tensorboard) task_tensorboard ;;
  deploy) task_deploy  "$@" ;;
  sync) task_sync ;;
  ssh) task_ssh ;;
  train) task_train ;;
  gpu-usage) task_gpu_usage ;;
  tf) task_tf  "$@" ;;
  *) task_usage ;;
esac
