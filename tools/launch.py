"""Launching tool for DGL distributed training"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import logging
import time
import json
import multiprocessing
from threading import Thread

DEFAULT_PORT = 30050

def execute_remote(cmd, ip, port, thread_list, username=""):
    """execute command line on remote machine via ssh"""
    ip_prefix = ""
    if username:
        ip_prefix += "{username}@".format(username=username)
    cmd = "ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} \'{cmd}\'".format(
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )

    # thread func to run the job
    def run(cmd):
        subprocess.check_call(cmd, shell = True)

    thread = Thread(target = run, args=(cmd,))
    thread.setDaemon(True)
    thread.start()
    thread_list.append(thread)

def submit_jobs(args, udf_command):
    """Submit distributed jobs (server and client processes) via ssh"""
    hosts = []
    thread_list = []
    server_count_per_machine = 0

    # Get the IP addresses of the cluster.
    ip_config = args.workspace + '/' + args.ip_config
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = DEFAULT_PORT
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of ip_config.")
            server_count_per_machine = args.num_servers
    # Get partition info of the graph data
    part_config = args.workspace + '/' + args.part_config
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'num_parts' in part_metadata, 'num_parts does not exist.'
    # The number of partitions must match the number of machines in the cluster.
    assert part_metadata['num_parts'] == len(hosts), \
            'The number of graph partitions has to match the number of machines in the cluster.'

    tot_num_clients = args.num_trainers * (1 + args.num_samplers) * len(hosts)
    # launch server tasks
    server_env_vars = 'DGL_ROLE=server DGL_NUM_SAMPLER=' + str(args.num_samplers)
    server_env_vars = server_env_vars + ' ' + 'OMP_NUM_THREADS=' + str(args.num_server_threads)
    server_env_vars = server_env_vars + ' ' + 'DGL_NUM_CLIENT=' + str(tot_num_clients)
    server_env_vars = server_env_vars + ' ' + 'DGL_CONF_PATH=' + str(part_config)
    server_env_vars = server_env_vars + ' ' + 'DGL_IP_CONFIG=' + str(ip_config)
    server_env_vars = server_env_vars + ' ' + 'DGL_NUM_SERVER=' + str(args.num_servers)
    for i in range(len(hosts)*server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        server_env_vars_cur = server_env_vars + ' ' + 'DGL_SERVER_ID=' + str(i)
        # persist env vars for entire cmd block. required if udf_command is a chain of commands
        # wrap in parens to not pollute env:
        #     https://stackoverflow.com/a/45993803
        cmd = "(export {server_env_vars}; {udf_command})".format(
            server_env_vars=server_env_vars_cur,
            udf_command=udf_command,
        )
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        print("\nserver{} cmd:\n{}\n\n".format(i, cmd))
        execute_remote(cmd, ip, args.ssh_port, thread_list, username=args.ssh_username)

    # launch client tasks
    client_env_vars = 'DGL_DIST_MODE="distributed" DGL_ROLE=client DGL_NUM_SAMPLER=' + str(args.num_samplers)
    client_env_vars = client_env_vars + ' ' + 'DGL_NUM_CLIENT=' + str(tot_num_clients)
    client_env_vars = client_env_vars + ' ' + 'DGL_CONF_PATH=' + str(part_config)
    client_env_vars = client_env_vars + ' ' + 'DGL_IP_CONFIG=' + str(ip_config)
    client_env_vars = client_env_vars + ' ' + 'DGL_NUM_SERVER=' + str(args.num_servers)
    if os.environ.get('OMP_NUM_THREADS') is not None:
        client_env_vars = client_env_vars + ' ' + 'OMP_NUM_THREADS=' + os.environ.get('OMP_NUM_THREADS')
    else:
        client_env_vars = client_env_vars + ' ' + 'OMP_NUM_THREADS=' + str(args.num_omp_threads)
    if os.environ.get('PYTHONPATH') is not None:
        client_env_vars = client_env_vars + ' ' + 'PYTHONPATH=' + os.environ.get('PYTHONPATH')

    torch_cmd = '-m torch.distributed.launch'
    torch_cmd = torch_cmd + ' ' + '--nproc_per_node=' + str(args.num_trainers)
    torch_cmd = torch_cmd + ' ' + '--nnodes=' + str(len(hosts))
    torch_cmd = torch_cmd + ' ' + '--node_rank=' + str(0)
    torch_cmd = torch_cmd + ' ' + '--master_addr=' + str(hosts[0][0])
    torch_cmd = torch_cmd + ' ' + '--master_port=' + str(1234)
    for node_id, host in enumerate(hosts):
        ip, _ = host
        new_torch_cmd = torch_cmd.replace('node_rank=0', 'node_rank='+str(node_id))
        if 'python3.7' in udf_command:
            # we use python3.7 in image-retrieval, need this otherwise things break
            new_udf_command = udf_command.replace('python3.7', 'python3.7 ' + new_torch_cmd)
        elif 'python3' in udf_command:
            new_udf_command = udf_command.replace('python3', 'python3 ' + new_torch_cmd)
        elif 'python2' in udf_command:
            new_udf_command = udf_command.replace('python2', 'python2 ' + new_torch_cmd)
        else:
            new_udf_command = udf_command.replace('python', 'python ' + new_torch_cmd)
        # persist env vars for entire cmd block. required if udf_command is a chain of commands
        # wrap in parens to not pollute env:
        #     https://stackoverflow.com/a/45993803
        cmd = "(export {client_env_vars}; {new_udf_command})".format(
            client_env_vars=client_env_vars,
            new_udf_command=new_udf_command,
        )
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        print("\n\n cmd (node_id={}):\n{}\n\n".format(node_id, cmd))
        execute_remote(cmd, ip, args.ssh_port, thread_list, username=args.ssh_username)
    print("\n\nlen(thread_list): {}\n\n".format(len(thread_list)))
    for thread in thread_list:
        thread.join()

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    parser.add_argument(
        "--ssh_username", default="",
        help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
             "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
             "instead of 'ssh 1.2.3.4 CMD'"
    )
    parser.add_argument('--workspace', type=str,
                        help='Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd')
    parser.add_argument('--num_trainers', type=int,
                        help='The number of trainer processes per machine')
    parser.add_argument('--num_omp_threads', type=int,
                        help='The number of OMP threads per trainer')
    parser.add_argument('--num_samplers', type=int, default=0,
                        help='The number of sampler processes per trainer process')
    parser.add_argument('--num_servers', type=int,
                        help='The number of server processes per machine')
    parser.add_argument('--part_config', type=str,
                        help='The file (in workspace) of the partition config')
    parser.add_argument('--ip_config', type=str,
                        help='The file (in workspace) of IP configuration for server processes')
    parser.add_argument('--num_server_threads', type=int, default=1,
                        help='The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on \
                        the same machine. By default, it is 1.')
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, 'Please provide user command line.'
    assert args.num_trainers is not None and args.num_trainers > 0, \
            '--num_trainers must be a positive number.'
    assert args.num_samplers is not None and args.num_samplers >= 0, \
            '--num_samplers must be a non-negative number.'
    assert args.num_servers is not None and args.num_servers > 0, \
            '--num_servers must be a positive number.'
    assert args.num_server_threads > 0, '--num_server_threads must be a positive number.'
    assert args.workspace is not None, 'A user has to specify a workspace with --workspace.'
    assert args.part_config is not None, \
            'A user has to specify a partition configuration file with --part_config.'
    assert args.ip_config is not None, \
            'A user has to specify an IP configuration file with --ip_config.'
    if args.num_omp_threads is None:
        # Here we assume all machines have the same number of CPU cores as the machine
        # where the launch script runs.
        args.num_omp_threads = max(multiprocessing.cpu_count() // 2 // args.num_trainers, 1)
        print('The number of OMP threads per trainer is set to', args.num_omp_threads)

    udf_command = str(udf_command[0])
    if 'python' not in udf_command:
        raise RuntimeError("DGL launching script can only support Python executable file.")
    if sys.version_info.major and sys.version_info.minor >= 8:
        if args.num_samplers > 0:
            print('WARNING! DGL does not support multiple sampler processes in Python>=3.8. '
                  + 'Set the number of sampler processes to 0.')
            args.num_samplers = 0
    submit_jobs(args, udf_command)

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
