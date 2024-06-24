import grp
import os
from py import builtin

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
for version in ['2018a', '2019b', '2021a']:
    genius_modulepath.append(f'/apps/leuven/skylake/{version}/modules/all')

# Specify hortense access flag in order to run jobs
# Flag is selected according to user group
hortense_access_flag = ''
groups = [grp.getgrgid(x).gr_name for x in os.getgroups()]
for admingroup in ['astaff', 'badmin', 'gadminforever', 'l_sysadmin']:
    if admingroup in groups:
        hortense_access_flag = f'-A {admingroup}'
        break

# Site Configuration
site_configuration = {
    'systems': [
        {
            'name': 'hydra',
            'descr': 'Hydra',
            'hostnames': ['login1.cerberus.os', 'login2.cerberus.os', '.*hydra.*'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 1,
                    'launcher': 'local',
                },
                {
                    'name': 'single-node',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 1,
                    'launcher': 'local',
                },
                {
                    'name': 'mpi-job',
                    'scheduler': 'slurm',
                    'access': [],
                    'environs': ['foss-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 1,
                    'launcher': 'srun',
                },
            ]
        },
        {
            'name': 'hortense',
            'descr': 'VSC Tier-1 hortense',
            'hostnames': ['login.*.dodrio.os'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 1,
                    'launcher': 'local',
                },
                {
                    'name': 'single-node',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [hortense_access_flag],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 1,
                    'launcher': 'local',
                },
                {
                    'name': 'mpi-job',
                    'scheduler': 'slurm',
                    'access': [hortense_access_flag],
                    'environs': ['foss-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 1,
                    # TODO Here we actually want to set vsc-mympirun, but since
                    # this is a custom launcher not shipped with ReFrame, we
                    # can only do this in the test itself after registering the
                    # vsc-mympirun launcher
                    'launcher': 'srun',
                },
            ]
        },
        {
            'name': 'genius',
            'descr': 'VSC Tier-2 Genius',
            'hostnames': ['tier2-p-login-[1-4].genius.hpc.kuleuven.be'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 1,
                    'launcher': 'local',
                    'env_vars': [['MODULEPATH', ':'.join(genius_modulepath)]],
                },
                {
                    'name': 'single-node',
                    'scheduler': 'torque',
                    'modules': [],
                    'access': [kul_account_string_tier2],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 1,
                    'launcher': 'local',
                    'env_vars': [['MODULEPATH', ':'.join(genius_modulepath)]],
                },
                {
                    'name': 'mpi-job',
                    'scheduler': 'torque',
                    'access': [kul_account_string_tier2],
                    'environs': ['foss-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 1,
                    'launcher': 'mpirun',
                    'env_vars': [['MODULEPATH', ':'.join(genius_modulepath)]],
                },
            ]
        },
        {
            'name': 'vaughan',
            'descr': 'VSC Tier-2 Vaughan',
            'hostnames': ['login[1-2].vaughan'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                {
                    'name': 'single-node',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                {
                    'name': 'mpi-job',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2],
                    'environs': ['intel-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 10,
                    # TODO Here we actually want to set vsc-mympirun, but since
                    # this is a custom launcher not shipped with ReFrame, we
                    # can only do this in the test itself after registering the
                    # vsc-mympirun launcher
                    'launcher': 'srun',
                },
                {
                    'name': 'nvidia',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2, '-p ampere_gpu'],
                    'environs': ['CUDA', 'standard'],
                    'descr': 'Nvidia ampere node',
                    'max_jobs': 1,
                    'launcher': 'srun',
                    'resources': [
                        {
                        'name': 'gpu',
                        'options': ['--gres=gpu:{num_gpus}'],
                        },
                    ]
                }
            ]
        },
        {
            'name': 'leibniz',
            'descr': 'VSC Tier-2 Leibniz',
            'hostnames': ['login[1-2].leibniz'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                {
                    'name': 'single-node',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                                {
                    'name': 'mpi-job',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2],
                    'environs': ['intel-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 10,
                    # TODO Here we actually want to set vsc-mympirun, but since
                    # this is a custom launcher not shipped with ReFrame, we
                    # can only do this in the test itself after registering the
                    # vsc-mympirun launcher
                    'launcher': 'srun',
                },
                {
                    'name': 'nvidia',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2, '-p pascal_gpu'],
                    'environs': ['CUDA', 'standard'],
                    'descr': 'Nvidia pascal nodes',
                    'max_jobs': 2,
                    'launcher': 'srun',
                    'resources': [
                        {
                        'name': 'gpu',
                        'options': ['--gres=gpu:{num_gpus}'],
                        },
                    ]
                }
            ]
        },
        {
            'name': 'breniac',
            'descr': 'VSC Tier-2 Breniac',
            'hostnames': ['login.breniac'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'local',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'tests in the local node (no job)',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                {
                    'name': 'single-node',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [calcua_account_string_tier2],
                    'environs': ['standard'],
                    'descr': 'single-node jobs',
                    'max_jobs': 10,
                    'launcher': 'local',
                },
                                {
                    'name': 'mpi-job',
                    'scheduler': 'slurm',
                    'access': [calcua_account_string_tier2],
                    'environs': ['intel-2021a'],
                    'descr': 'MPI jobs',
                    'max_jobs': 10,
                    # TODO Here we actually want to set vsc-mympirun, but since
                    # this is a custom launcher not shipped with ReFrame, we
                    # can only do this in the test itself after registering the
                    # vsc-mympirun launcher
                    'launcher': 'srun',
                },
            ]
        },
    ],
    'environments': [
        {
            'name': 'standard', 'cc': 'gcc', 'cxx': 'g++', 'ftn': 'gfortran',},
        {
            'name': 'foss-2021a', 'cc': 'mpicc', 'cxx': 'mpicxx',
            'ftn': 'mpif90', 'modules': ['foss/2021a'],},
        {   
            'name': 'intel-2021a',
            'modules': ['intel'],
            'cc': 'mpiicc',
            'cxx': 'mpiicpc',
            'ftn': 'mpiifort',
            #'target_systems': ['vaughan', 'leibniz']
        },
        {
            'name': 'CUDA',
            'modules': ['CUDA'],
            'cc': 'nvcc', 
            'cxx': 'nvcc', 
        },
    ],
    'general': [
        {
            'purge_environment': True,
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
