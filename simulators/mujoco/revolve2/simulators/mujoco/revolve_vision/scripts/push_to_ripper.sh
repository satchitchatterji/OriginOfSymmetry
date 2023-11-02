#!/bin/bash

i=$1
if [ -z $i ]
then
  echo "No ripper id provided."
  exit 1
fi

set -euo pipefail

user=kevingd
if [ $i -ge 8 ]
then
  user=kgd
fi

host=ripper$i
base=$user@$host:code

update(){
  dir=$1
  cd ../$dir
  shift
  echo "Updating from $(pwd)"
  rsync -avzhP --prune-empty-dirs $@ $base/$dir
}

update revolve_vision bin src scripts
update abrain src commands.sh CMakeLists.txt setup.py
update revolve ./{core,runners}

if [ $2 == '--compile' ]
then
  ssh $user@$host bash <<EOF
    set -euo pipefail
    cd code/abrain
    source ../venv/bin/activate
    ./commands.sh install-cached release
EOF
fi
