import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class GaussianBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['+default']
        self.modules = ['Gaussian']

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
        self.valid_systems = ['+cpu +default']
        self.tags = {'apps', 'gaussian', 'performance', 'vsc'}

    @run_after('setup')
    def set_num_cpus(self):
        ncpus = self.current_partition.extras['num_cpus']
        self.num_cpus_per_task = ncpus
        # request roughly 3.5 GB per core, leaving some headroom for the OS
        memory_gb = max(8, int(ncpus * 3.5))
        self.executable = (
            f'time g16 -c="0-{ncpus - 1}" -m={memory_gb}GB < input-file.com'
        )
        self.descr = f'Single Node Gaussian Test, cpus{ncpus}'

    @run_before('run')
    def replace_launcher(self):
        self.job.launcher = getlauncher('local')()
