from sites.common import cpu_env_list

# VUB - Hydra (tier-2) and sofia (tier-1)
#
# sofia (docs.vscentrum.be/brussel/tier1_sofia.html): a Tier-1 project is
# needed, pass it at run time with `./run.sh -J '-A <project>'`. Jobs must
# name a partition and start in a clean environment; memory overrides
# (--mem*) are rejected. On zen4_h200 Slurm enforces exactly 24 cores per
# GPU (-> extras['cpus_per_gpu']); no cluster module, srun works for MPI.

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
        # login01/login02 behind login.sofia.vub.be; the exact hostname is not
        # documented, so match loosely (--system=sofia always works).
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
                'access': ['-p zen5_dense'],
                'environs': cpu_env_list,
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
                'access': ['-p zen5_himem'],
                'environs': cpu_env_list,
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
                'access': ['-p zen4_h200'],
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
                'access': ['-p zen5_vis'],
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
