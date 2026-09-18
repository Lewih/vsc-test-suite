import grp
import os
from sites.common import cpu_env_list

# VUB - Hydra (tier-2) and sofia (tier-1)

_flag = ''
for _group in [grp.getgrgid(x).gr_name for x in os.getgroups()]:
    if _group in ('astaff', 'badmin', 'gadminforever', 'l_sysadmin'):
        _flag = f'-A {_group}'
        break
_account = [_flag] if _flag else []

# sofia only provides the 2025a toolchains.
_sofia_envs = [e for e in cpu_env_list if e == 'standard' or '2025a' in e]

_gpu_resources = [{'name': 'gpu', 'options': ['--gpus-per-node={num_gpus}']}]

systems = [
    {
        'name': 'hydra',
        'descr': 'VUB Tier-2 Hydra',
        'hostnames': ['login1.cerberus.os', 'login2.cerberus.os', '.*hydra.*'],
        'modules_system': 'lmod',
        'partitions': [
            {
                'name': 'login',
                'scheduler': 'local',
                'modules': [],
                'access': [],
                'environs': ['standard'],
                'descr': 'tests on the login node (no job)',
                'max_jobs': 1,
                'launcher': 'local',
                'features': ['login'],
                'extras': {'num_cpus': 40},
            },
            {
                'name': 'default',
                'scheduler': 'slurm',
                'modules': [],
                'access': [],
                'environs': cpu_env_list,
                'descr': 'default-node jobs (skylake)',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 40},
            },
        ]
    },
    {
        'name': 'sofia',
        'descr': 'VSC Tier-1 sofia',
        # login0[12] carry several FQDNs; 'login01.sofia.brussel.vsc' is the
        # one that names the cluster (see vsc_host_names in config_vsc.py).
        'hostnames': ['.*sofia.*'],
        'modules_system': 'lmod',
        'partitions': [
            {
                'name': 'login',
                'scheduler': 'local',
                'modules': [],
                'access': [],
                'environs': ['standard'],
                'descr': 'tests on the login node (no job)',
                'max_jobs': 1,
                'launcher': 'local',
                'features': ['login'],
                'extras': {'num_cpus': 1},
            },
            {
                'name': 'default',
                'scheduler': 'slurm',
                'modules': [],
                'access': _account + ['-p zen5_dense'],
                'environs': _sofia_envs,
                'descr': 'default-node jobs (zen5c, 2x192 cores)',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 384},
            },
            {
                'name': 'zen5_himem',
                'scheduler': 'slurm',
                'modules': [],
                'access': _account + ['-p zen5_himem'],
                'environs': _sofia_envs,
                'descr': 'zen5 nodes, 1.5TB memory',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu'],
                'extras': {'num_cpus': 192},
            },
            {
                'name': 'zen4_h200',
                'scheduler': 'slurm',
                'modules': [],
                'access': _account + ['-p zen4_h200'],
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia H200 nodes (24 cores per GPU enforced)',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 192, 'num_gpus': 8, 'cpus_per_gpu': 24},
            },
            {
                'name': 'zen5_vis',
                'scheduler': 'slurm',
                'modules': [],
                'access': _account + ['-p zen5_vis'],
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia RTX 5000 Ada visualisation nodes',
                'max_jobs': 4,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 192, 'num_gpus': 2},
            },
        ]
    },
]
