from sites.common import cpu_env_list

_account = '-A lpt2_vsc_test_suite'
_modulepath = [f'/apps/leuven/skylake/{v}/modules/all' for v in ('2023a', '2024a', '2025a')]
_env_vars = [['MODULEPATH', ':'.join(_modulepath)]]

# KU Leuven - Genius (tier-2)
systems = [
    {
        'name': 'genius',
        'descr': 'VSC Tier-2 Genius',
        'hostnames': ['tier2-p-login-[1-4].genius.hpc.kuleuven.be'],
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
                'env_vars': _env_vars,
                'features': ['login'],
                'extras': {'num_cpus': 36},
            },
            {
                'name': 'default',
                'scheduler': 'torque',
                'modules': [],
                'access': [_account],
                'environs': cpu_env_list,
                'descr': 'default-node jobs',
                'max_jobs': 20,
                'launcher': 'local',
                'env_vars': _env_vars,
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 36},
            },
        ]
    },
]
