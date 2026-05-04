import grp
import os, sys

# use 'info' to log to syslog
syslog_level = 'warning'
    
# To run jobs on the kul cluster, you need to be a member of the following
# vsc group
kul_account_string_tier2 = '-A lpt2_vsc_test_suite'

# To run jobs on the calcua cluster, you need to be a member of the following
# vsc group
calcua_account_string_tier2 = '-A ap_calcua_staff'

# By default, not all installed modules are visible on the genius cluster
genius_modulepath = []
for version in ['2023a', '2024a', '2025a']:
    genius_modulepath.append(f'/apps/leuven/skylake/{version}/modules/all')

# Specify hortense access flag in order to run jobs
# Flag is selected according to user group
hortense_access_flag = ''
groups = [grp.getgrgid(x).gr_name for x in os.getgroups()]
for admingroup in ['astaff', 'badmin', 'gadminforever', 'l_sysadmin']:
    if admingroup in groups:
        hortense_access_flag = f'-A {admingroup}'
        break

# Programming environments available on CPU partitions.
# Tests should select via feature flags (e.g. '+foss', '+intel', '+mpi'),
# never by hard-coded environment name.
cpu_env_list = [
    'standard',
    'foss-2023a', 'foss-2023a_mpi',
    'foss-2024a', 'foss-2024a_mpi',
    'foss-2025a', 'foss-2025a_mpi',
    'intel-2023a', 'intel-2023a_mpi',
    'intel-2024a', 'intel-2024a_mpi',
    'intel-2025a', 'intel-2025a_mpi',
]

# Site Configuration
site_configuration = {
    'systems': [
        # ------------------------------------------------------------------
        # VUB - Hydra (tier-2)
        # ------------------------------------------------------------------
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
                    'max_jobs': 1,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 40},
                },
            ]
        },
        # ------------------------------------------------------------------
        # UGent - Hortense (tier-1)
        # ------------------------------------------------------------------
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
                    'access': [hortense_access_flag, '--exclusive'],
                    'environs': cpu_env_list,
                    'descr': 'default-node jobs',
                    'max_jobs': 1,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 128},
                },
            ]
        },
        # ------------------------------------------------------------------
        # KUL - Genius (tier-2)
        # ------------------------------------------------------------------
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
                    'env_vars': [['MODULEPATH', ':'.join(genius_modulepath)]],
                    'features': ['login'],
                    'extras': {'num_cpus': 36},
                },
                {
                    'name': 'default',
                    'scheduler': 'torque',
                    'modules': [],
                    'access': [kul_account_string_tier2],
                    'environs': cpu_env_list,
                    'descr': 'default-node jobs',
                    'max_jobs': 1,
                    'launcher': 'local',
                    'env_vars': [['MODULEPATH', ':'.join(genius_modulepath)]],
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 36},
                },
            ]
        },
        # ------------------------------------------------------------------
        # UA - Vaughan (tier-2, calcua)
        # ------------------------------------------------------------------
        {
            'name': 'vaughan',
            'descr': 'VSC Tier-2 Vaughan',
            'hostnames': ['.*vaughan'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests on the login node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['login'],
                    'extras': {'num_cpus': 32},
                },
                {
                    'name': 'default',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': cpu_env_list,
                    'descr': 'default-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 64},
                },
                {
                    'name': 'zen2',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p zen2'],
                    'environs': cpu_env_list,
                    'descr': 'zen2 nodes',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 64},
                },
                {
                    'name': 'zen3',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p zen3'],
                    'environs': cpu_env_list,
                    'descr': 'zen3 nodes',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 64},
                },
                {
                    'name': 'zen3_512',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p zen3_512'],
                    'environs': cpu_env_list,
                    'descr': 'zen3 nodes, 512GB memory',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 64},
                },
                {
                    'name': 'nvidia',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2, '-p ampere_gpu'],
                    'environs': ['CUDA', 'standard'],
                    'descr': 'Nvidia ampere node',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'resources': [
                        {
                            'name': 'gpu',
                            'options': ['--gpus-per-node={num_gpus}'],
                        },
                    ],
                    'features': ['gpu', 'nvidia'],
                    'extras': {'num_cpus': 64, 'num_gpus': 4},
                },
                {
                    'name': 'amd',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2, '-p arcturus_gpu'],
                    'environs': ['standard'],
                    'descr': 'AMD GPU node',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'resources': [
                        {
                            'name': 'gpu',
                            'options': ['--gpus-per-node={num_gpus}'],
                        },
                    ],
                    'features': ['gpu', 'amd'],
                    'extras': {'num_cpus': 64, 'num_gpus': 2},
                },
            ]
        },
        # ------------------------------------------------------------------
        # UA - Leibniz (tier-2, calcua)
        # ------------------------------------------------------------------
        {
            'name': 'leibniz',
            'descr': 'VSC Tier-2 Leibniz',
            'hostnames': ['.*leibniz'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests on the login node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['login'],
                    'extras': {'num_cpus': 56},
                },
                {
                    'name': 'default',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': cpu_env_list,
                    'descr': 'default-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 28},
                },
                {
                    'name': 'broadwell',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p broadwell'],
                    'environs': cpu_env_list,
                    'descr': 'broadwell nodes',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 28},
                },
                {
                    'name': 'broadwell_256',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p broadwell_256'],
                    'environs': cpu_env_list,
                    'descr': 'broadwell nodes, 256GB memory',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 28},
                },
                {
                    'name': 'nvidia',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2, '-p pascal_gpu'],
                    'environs': ['CUDA', 'standard'],
                    'descr': 'Nvidia pascal nodes',
                    'max_jobs': 5,
                    'launcher': 'local',
                    'resources': [
                        {
                            'name': 'gpu',
                            'options': ['--gpus-per-node={num_gpus}'],
                        },
                    ],
                    'features': ['gpu', 'nvidia', 'deprecated'],
                    'extras': {'num_cpus': 28, 'num_gpus': 2},
                },
            ]
        },
        # ------------------------------------------------------------------
        # UA - Breniac (tier-2, calcua)
        # ------------------------------------------------------------------
        {
            'name': 'breniac',
            'descr': 'VSC Tier-2 Breniac',
            'hostnames': ['.*breniac'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests on the login node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['login'],
                    'extras': {'num_cpus': 28},
                },
                {
                    'name': 'default',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': cpu_env_list,
                    'descr': 'default-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 28},
                },
                {
                    'name': 'skylake',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2, '-p skylake'],
                    'environs': cpu_env_list,
                    'descr': 'skylake nodes',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'features': ['cpu'],
                    'extras': {'num_cpus': 28},
                },
            ]
        },
    ],
    'environments': [
        {'name': 'standard', 'cc': 'gcc', 'cxx': 'g++', 'ftn': 'gfortran',
         'features': ['default']},

        {'name': 'foss-2023a', 'cc': 'gcc', 'cxx': 'g++', 'ftn': 'gfortran',
         'modules': ['foss/2023a'], 'features': ['foss']},
        {'name': 'foss-2023a_mpi', 'cc': 'mpicc', 'cxx': 'mpicxx', 'ftn': 'mpifort',
         'modules': ['foss/2023a'], 'features': ['foss', 'mpi', 'fftw']},

        {'name': 'foss-2024a', 'cc': 'gcc', 'cxx': 'g++', 'ftn': 'gfortran',
         'modules': ['foss/2024a'], 'features': ['foss']},
        {'name': 'foss-2024a_mpi', 'cc': 'mpicc', 'cxx': 'mpicxx', 'ftn': 'mpifort',
         'modules': ['foss/2024a'], 'features': ['foss', 'mpi', 'fftw']},

        {'name': 'foss-2025a', 'cc': 'gcc', 'cxx': 'g++', 'ftn': 'gfortran',
         'modules': ['foss/2025a'], 'features': ['foss']},
        {'name': 'foss-2025a_mpi', 'cc': 'mpicc', 'cxx': 'mpicxx', 'ftn': 'mpifort',
         'modules': ['foss/2025a'], 'features': ['foss', 'mpi', 'fftw']},

        {'name': 'intel-2023a', 'cc': 'icx', 'cxx': 'icpx', 'ftn': 'ifx',
         'modules': ['intel/2023a'], 'features': ['intel']},
        {'name': 'intel-2023a_mpi', 'cc': 'mpiicc', 'cxx': 'mpiicpc', 'ftn': 'mpiifort',
         'modules': ['intel/2023a'], 'features': ['intel', 'mpi', 'fftw']},

        {'name': 'intel-2024a', 'cc': 'icx', 'cxx': 'icpx', 'ftn': 'ifx',
         'modules': ['intel/2024a'], 'features': ['intel']},
        {'name': 'intel-2024a_mpi', 'cc': 'mpiicx', 'cxx': 'mpiicpx', 'ftn': 'mpiifx',
         'modules': ['intel/2024a'], 'features': ['intel', 'mpi', 'fftw']},

        {'name': 'intel-2025a', 'cc': 'icx', 'cxx': 'icpx', 'ftn': 'ifx',
         'modules': ['intel/2025a'], 'features': ['intel']},
        {'name': 'intel-2025a_mpi', 'cc': 'mpiicx', 'cxx': 'mpiicpx', 'ftn': 'mpiifx',
         'modules': ['intel/2025a'], 'features': ['intel', 'mpi', 'fftw']},

        {'name': 'CUDA', 'cc': 'nvcc', 'cxx': 'nvcc',
         'modules': ['CUDA/12.8.0'], 'features': ['cuda']}, # startinf with CUDA 13 some gpu architecture are deprecated
    ],
    'general': [
        {
            'purge_environment': False,
            'resolve_module_conflicts': False,  # avoid loading the module before submitting the job
        }
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s',
                },
            ],
        }
    ],
}
