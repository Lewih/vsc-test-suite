# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class MatlabLinalgBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['standard']
        self.modules = ['MATLAB']

        self.perf_patterns = {
            'dot': sn.extractsingle(
                r'Dot product:\s+(?P<dot>\S+)\s+s',
                self.stdout, 'dot', float),
            'cholesky': sn.extractsingle(
                r'Cholesky factorisation:'
                r'\s+(?P<cholesky>\S+)\s+s',
                self.stdout, 'cholesky', float),
            'lu': sn.extractsingle(
                r'LU factorisation:'
                r'\s+(?P<lu>\S+)\s+s',
                self.stdout, 'lu', float),
        }
        self.sanity_patterns = sn.assert_found(r'MATLAB Version: *',
                                               self.stdout)
        self.executable = 'cat'
        self.executable_opts = ['linalg.m | matlab -nodesktop -nosplash']
        self.num_tasks_per_node = 1
        self.tags = {'apps', 'matlab', 'performance', 'vsc'}
        self.maintainers = ['Lewih']


@rfm.simple_test
class MatlabLinalgTest(MatlabLinalgBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['leibniz:default-node',
                              'vaughan:default-node',
                              'breniac:default-node',
                              'hydra:default-node',
                              'genius:default-node']

        self.reference = {
            'leibniz:default-node': {
                'dot': (0.34, None, 0.05, 'seconds'),
                'cholesky': (0.05, None, 0.05, 'seconds'),
                'lu': (0.18, None, 0.05, 'seconds'),
            },
            'vaughan:default-node': {
                'dot': (0.28, None, 0.10, 'seconds'),
                'cholesky': (0.06, None, 0.10, 'seconds'),
                'lu': (0.18, None, 0.10, 'seconds'),
            },
            'breniac:default-node': {
                'dot': (0.28, None, 0.10, 'seconds'),
                'cholesky': (0.06, None, 0.10, 'seconds'),
                'lu': (0.24, None, 0.10, 'seconds'),
            },
            'genius:default-node': {
                'dot': (0.14, None, 0.10, 'seconds'),
                'cholesky': (0.05, None, 0.10, 'seconds'),
                'lu': (0.29, None, 0.10, 'seconds'),
            },
            'hydra:default-node': {
                'dot': (0.14, None, 0.10, 'seconds'),
                'cholesky': (0.05, None, 0.10, 'seconds'),
                'lu': (0.20, None, 0.10, 'seconds'),
            },
        }

    @run_after('setup')
    def set_num_cpus(self):
        if self.current_system.name in ['leibniz', 'breniac']:
            self.num_cpus_per_task = 28
        elif self.current_system.name == 'vaughan':
            self.num_cpus_per_task = 32
        elif self.current_system.name == 'hydra':
            self.num_cpus_per_task = 40
            self.job.options = ["--partition=skylake,skylake_mpi", "--exclusive"]
        elif self.current_system.name == 'genius':
            self.num_cpus_per_task = 36
            self.modules = ['matlab']

        self.descr = f'Test a few typical Matlab operations, cpus={self.num_cpus_per_task }'
