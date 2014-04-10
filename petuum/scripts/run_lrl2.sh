#!/bin/bash

if [ $# -ne 4 ]; then
  echo "usage: $0 <server_file> <train data path> <test data path> <num_client_threads>"
  exit
fi

# petuum parameters
server_file=$1
train_data_path=$2
test_data_path=$3
num_threads=$4

# lrl2 parameters
lambda=1
eval_interval=100
max_iter=10000
stepsize=0.001

# Find other Petuum paths by using the script's path
#app_prog=lrl2_main
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
project_root=`dirname $script_dir`
lrl2_path=$project_root/bin/lrl2_main
third_party_lib=$project_root/third_party/lib
config_file=$project_root/apps/lrl2/lrl2_tables.cfg
output_prefix=$project_root/dump

# Parse hostfile. Only spawn 1 client process per host ip.
host_file=`readlink -f $server_file`
host_list=`cat $host_file | awk '{ print $2 }' | sort | uniq`

# Spawn clients
client_rank=0
client_id=10000
for ip in $host_list; do
  echo "Running LRL2 client $client_rank"
  if [ $client_rank -eq 0 ]; then
    head_client=1
  else
    head_client=0
  fi
  sshpass -p 111 ssh $ip \
    LD_LIBRARY_PATH=$third_party_lib:${LD_LIBRARY_PATH} GLOG_logtostderr=true \
    GLOG_v=-1  GLOG_vmodule="" \
    $lrl2_path \
    --server_file=$host_file \
    --config_file=$config_file \
    --num_threads=$num_threads \
    --client_id=$(( client_id+client_rank )) \
    --train_data_file=$project_root/$train_data_path \
    --test_data_file=$project_root/$test_data_path \
    --client_rank=$client_rank \
    --data_global_idx_begin=0 \
    --aug_dim=15 \
    --eval_interval=$eval_interval \
    --lambda=$lambda \
    --stepsize=$stepsize \
    --max_iter=$max_iter &
  
  client_rank=$(( client_rank+1 ))
done
