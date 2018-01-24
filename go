#!/bin/bash

set -eu

function task_usage {
  echo "Usage: $0 deploy | ssh | tf"
  exit 1
}

function ensure_venv {
  if [ ! -d venv ]; then
    virtualenv -p python3 venv
    populate_venv
  fi
  set +u
  source ./venv/bin/activate
  set -u
}

function task_tf {
  cd deploy/
  terraform "$@" \
    -var "ssh_public_key=$(gopass show dev/aws-ssh-public-key)" \
    -var "access_key=$(gopass show dev/aws-access-key)" \
    -var "secret_key=$(gopass show dev/aws-secret-key)"
}

function task_ssh {
  echo "SSH"
}

function task_deploy {
  task_tf "apply"
}

CMD=${1:-}
shift || true
case ${CMD} in
  tf) task_tf  "$@" ;;
  ssh) task_ssh  ;;
  deploy) task_deploy  "$@" ;;
  *) task_usage ;;
esac
