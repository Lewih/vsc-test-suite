import grp
import os
from sites.common import cpu_env_list

# Access flag is selected at import time based on the current user's groups.
_flag = ''
for _group in [grp.getgrgid(x).gr_name for x in os.getgroups()]:
    if _group in ('astaff', 'badmin', 'gadminforever', 'l_sysadmin'):
        _flag = f'-A {_group}'
        break
_access = ([_flag] if _flag else []) + ['--exclusive']

# UGent - Hortense (tier-1)
systems = [
    {
        'name': 'hortense',
        'descr': 'VSC Tier-1 hortense',
        'hostnames': ['login.*.dodrio.os'],
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
                'extras': {'num_cpus': 64},
            },
            {
                'name': 'default',
                'scheduler': 'slurm',
                'modules': [],
                'access': _access,
                'environs': cpu_env_list,
                'descr': 'default-node jobs',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 128},
            },
        ]
    },
]
