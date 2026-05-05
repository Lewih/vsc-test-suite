from sites.common import cpu_env_list

# VUB - Hydra (tier-2)
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
]
