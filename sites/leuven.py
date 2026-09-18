from sites.common import cpu_env_list

# KU Leuven / UHasselt - Genius, wICE, Mindwell (all tier-2)
#
# All three are reached from the Genius login nodes, so autodetection picks
# genius; use --system=wice / --system=mindwell for the others.
# Rules (docs.vscentrum.be/leuven/slurm_specifics.html): -A and --clusters are
# mandatory; sacct polling needs the cluster too (sched_options for ReFrame
# >= 4.10, SLURM_CLUSTERS from run.sh for older ones); job scripts need
# '#!/bin/bash -l' for the cluster module (-> use_login_shell); no PMI in
# Slurm, so MPI runs with mpirun, not srun.

_account = '-A lpt2_vsc_test_suite'   # TODO confirm; upstream PR #46 uses lpt2_sysadmin
_hostnames = ['tier2-p-login-[1-4].*']
_gpu_resources = [{'name': 'gpu', 'options': ['--gpus-per-node={num_gpus}']}]

# The Rocky 9 software stack only provides toolchains >= 2024a.
_env_list = [e for e in cpu_env_list if '2023a' not in e]

def _sched(cluster):
    # ReFrame >= 4.10: add '-M <cluster>' to sacct/squeue (and to the job).
    # Older versions ignore the key; run.sh exports SLURM_CLUSTERS for them.
    return {'slurm_multi_cluster_mode': [cluster]}


# Emit '#!/bin/bash -l' in the generated job scripts (see above).
general = [
    {
        'use_login_shell': True,
        'target_systems': ['genius', 'wice', 'mindwell'],
    },
]


def _login():
    # The Genius login nodes serve all three clusters.
    return {
        'name': 'login',
        'scheduler': 'local',
        'modules': [],
        'access': [],
        'environs': ['standard'],
        'descr': 'tests on the login node (no job)',
        'max_jobs': 1,
        'launcher': 'local',
        'features': ['login'],
        'extras': {'num_cpus': 36},
    }


systems = [
    {
        'name': 'genius',
        'descr': 'VSC Tier-2 Genius (GPU nodes only, CPU nodes decommissioned)',
        'hostnames': _hostnames,
        'modules_system': 'lmod',
        'partitions': [
            _login(),
            {
                'name': 'gpu_p100',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=genius', '-p gpu_p100'],
                'sched_options': _sched('genius'),
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia P100 nodes (Pascal)',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia', 'deprecated'],
                'extras': {'num_cpus': 36, 'num_gpus': 4},
            },
            {
                'name': 'gpu_v100',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=genius', '-p gpu_v100'],
                'sched_options': _sched('genius'),
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia V100 nodes',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 36, 'num_gpus': 8},
            },
        ]
    },
    {
        'name': 'wice',
        'descr': 'VSC Tier-2 wICE',
        'hostnames': _hostnames,
        'modules_system': 'lmod',
        'partitions': [
            _login(),
            {
                'name': 'default',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=wice', '-p batch'],
                'sched_options': _sched('wice'),
                'environs': _env_list,
                'descr': 'default-node jobs (icelake)',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 72, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'batch_sapphirerapids',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=wice', '-p batch_sapphirerapids'],
                'sched_options': _sched('wice'),
                'environs': _env_list,
                'descr': 'sapphirerapids nodes',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu'],
                'extras': {'num_cpus': 96, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'bigmem',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=wice', '-p bigmem'],
                'sched_options': _sched('wice'),
                'environs': _env_list,
                'descr': 'icelake nodes, 2TB memory',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu'],
                'extras': {'num_cpus': 72, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'gpu_a100',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=wice', '-p gpu_a100'],
                'sched_options': _sched('wice'),
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia A100 nodes',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 72, 'num_gpus': 4},
            },
            {
                'name': 'gpu_h100',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=wice', '-p gpu_h100'],
                'sched_options': _sched('wice'),
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia H100 nodes',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 64, 'num_gpus': 4},
            },
        ]
    },
    {
        'name': 'mindwell',
        'descr': 'VSC Tier-2 Mindwell',
        'hostnames': _hostnames,
        'modules_system': 'lmod',
        'partitions': [
            _login(),
            {
                'name': 'default',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=mindwell', '-p batch_graniterapids'],
                'sched_options': _sched('mindwell'),
                'environs': _env_list,
                'descr': 'default-node jobs (graniterapids)',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu', 'default'],
                'extras': {'num_cpus': 192, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'bigmem',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=mindwell', '-p bigmem'],
                'sched_options': _sched('mindwell'),
                'environs': _env_list,
                'descr': 'graniterapids nodes, 1.5TB memory',
                'max_jobs': 20,
                'launcher': 'local',
                'features': ['cpu'],
                'extras': {'num_cpus': 192, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'interactive',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=mindwell', '-p interactive'],
                'sched_options': _sched('mindwell'),
                'environs': _env_list,
                'descr': 'interactive partition (max 8 cores, 16h, no credits)',
                'max_jobs': 4,
                'launcher': 'local',
                'features': ['cpu'],
                'extras': {'num_cpus': 8, 'mpi_launcher': 'mpirun'},
            },
            {
                'name': 'gpu_b200',
                'scheduler': 'slurm',
                'modules': [],
                'access': [_account, '--clusters=mindwell', '-p gpu_b200'],
                'sched_options': _sched('mindwell'),
                'environs': ['CUDA', 'standard'],
                'descr': 'Nvidia B200 nodes',
                'max_jobs': 10,
                'launcher': 'local',
                'resources': _gpu_resources,
                'features': ['gpu', 'nvidia'],
                'extras': {'num_cpus': 192, 'num_gpus': 8},
            },
        ]
    },
]
