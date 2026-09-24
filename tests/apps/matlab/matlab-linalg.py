# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MatlabLinalgTest(rfm.RunOnlyRegressionTest):
    # class-level so that -S valid_systems/valid_prog_environs=... can override them
    valid_systems = ['+cpu +default']
    valid_prog_environs = ['+default']
    modules = ['MATLAB']
    executable = 'cat'
    executable_opts = ['linalg.m | matlab -nodesktop -nosplash']
    num_tasks_per_node = 1
    tags = {'apps', 'matlab', 'performance', 'vsc'}
    maintainers = ['Lewih']

    @run_after('init')
    def set_perf_patterns(self):
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

    @run_after('setup')
    def set_num_cpus(self):
        # cap the threading at 32 cores; MATLAB's implicit parallelism does
        # not scale past that and the big nodes have far more
        self.num_cpus_per_task = min(32, self.current_partition.extras['num_cpus'])
        self.job.options = ['--exclusive']
        self.descr = (
            f'Test a few typical Matlab operations, '
            f'cpus={self.num_cpus_per_task}'
        )

    @sanity_function
    def assert_matlab(self):
        return sn.assert_found(r'MATLAB Version: *', self.stdout)
