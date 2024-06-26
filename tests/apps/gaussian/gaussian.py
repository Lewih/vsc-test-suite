
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
import os


class GaussianBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['standard']
        self.modules = ['Gaussian/g16_c01-avx2']

        self.sanity_patterns = sn.assert_found(r' Normal termination of Gaussian',
                                               self.stdout)
        self.perf_patterns = {
            'time': (
                sn.extractsingle(
                r'^real\t(?P<minutes>\S+)m\S+s',
                self.stderr, 'minutes', float) + 
                sn.extractsingle(
                r'^real\t\S+m(?P<seconds>\S+)s',
                self.stderr, 'seconds', float) / 60.0)
        }

        self.maintainers = ['Lewih']


@rfm.simple_test
class GaussianCPUTest(GaussianBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['leibniz:default-node',
                              'vaughan:default-node',
                              'hydra:default-node']
        self.reference = {
            'leibniz:default-node': {
                'time': (33.0, None, 0.05, 'minutes'),

            },
            'vaughan:default-node': {
                'time': (15.0, None, 0.05, 'minutes'),
            },
            'hydra:default-node': {
                'time': (21.5, None, 0.05, 'minutes'),
            },
            
        }
        self.tags = {'apps', 'gaussian', 'performance', 'vsc'}


    @run_after('setup')
    def set_num_cpus(self):
        if self.current_system.name == 'leibniz':
            self.num_cpus_per_task = 28
            self.memory = 109
        elif self.current_system.name == 'vaughan':
            self.num_cpus_per_task = 64
            self.memory = 229
        elif self.current_system.name == 'hydra':
            self.num_cpus_per_task = 40
            self.job.options = ["--partition=skylake,skylake_mpi", "--exclusive"]
            self.modules = ['Gaussian/G16.A.03-intel-2017b']
            self.memory = 129

        self.executable = f'time g16 -c="0-{self.num_cpus_per_task-1}" -m={self.memory}GB < input-file.com'
    
        self.descr = f'Single Node Gaussian Test, cpus{self.num_cpus_per_task}'

    @run_before('run')
    def replace_launcher(self):
        self.job.launcher = getlauncher('local')()
