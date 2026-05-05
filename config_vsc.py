import glob
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

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

# Make cpu_env_list available to site files via `from sites.common import ...`
# without a physical common.py and without circular imports.
_common = types.ModuleType('sites.common')
_common.cpu_env_list = cpu_env_list
sys.modules['sites.common'] = _common

# Auto-discover cluster definitions: any sites/*.py that exports `system`.
# Files are loaded alphabetically; vsc_generic is appended last so its
# catch-all hostname pattern ('.*') never shadows a named cluster.
_sites_dir = os.path.join(os.path.dirname(__file__), 'sites')
_systems = []
for _f in sorted(glob.glob(os.path.join(_sites_dir, '*.py'))):
    _name = os.path.basename(_f)[:-3]
    if _name.startswith('_') or _name == 'common':
        continue
    _mod = importlib.import_module(f'sites.{_name}')
    if hasattr(_mod, 'systems'):
        _systems.extend(_mod.systems)
    elif hasattr(_mod, 'system'):
        _systems.append(_mod.system)

site_configuration = {
    'systems': _systems + [
        # ------------------------------------------------------------------
        # Generic VSC fallback — always last so '.*' doesn't shadow named clusters — matches any VSC host not listed above.
        # Assumes Slurm, lmod, and a single GPU per node.
        # Override num_cpus / num_gpus in a site-specific config if needed.
        # ------------------------------------------------------------------
        {
            'name': 'vsc_generic',
            'descr': 'Generic VSC fallback system',
            'hostnames': ['.*'],
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
                    'access': [],
                    'environs': cpu_env_list,
                    'descr': 'default compute partition',
                    'max_jobs': 20,
                    'launcher': 'local',
                    'features': ['cpu', 'default'],
                    'extras': {'num_cpus': 1},
                },
                {
                    'name': 'nvidia',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [],
                    'environs': ['CUDA', 'standard'],
                    'descr': 'Nvidia GPU partition',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'resources': [
                        {'name': 'gpu', 'options': ['--gpus-per-node={num_gpus}']},
                    ],
                    'features': ['gpu', 'nvidia'],
                    'extras': {'num_cpus': 1, 'num_gpus': 1},
                },
                {
                    'name': 'amd',
                    'scheduler': 'slurm',
                    'modules': [],
                    'access': [],
                    'environs': ['standard'],
                    'descr': 'AMD GPU partition',
                    'max_jobs': 10,
                    'launcher': 'local',
                    'resources': [
                        {'name': 'gpu', 'options': ['--gpus-per-node={num_gpus}']},
                    ],
                    'features': ['gpu', 'amd'],
                    'extras': {'num_cpus': 1, 'num_gpus': 1},
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
         'modules': ['CUDA/12.8.0'], 'features': ['cuda']}, 
    ],
    'general': [
        {
            'purge_environment': False,
            'resolve_module_conflicts': False,
        }
    ],
    'logging': [
        {
            'level': 'info',
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
