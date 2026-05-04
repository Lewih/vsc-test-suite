import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class NamdBaseTest(rfm.RunOnlyRegressionTest):
    # This test assumes NAMD3, MPI build, is the default version
    num_nodes = parameter([1, 2, 4])

    def __init__(self, arch):
        self.descr = (
            f'NAMD check on {arch}, number of nodes: {self.num_nodes}, '
            f'apoa1 and stmv (4 nodes only)'
        )
        self.modules = ['NAMD']

        self.sanity_patterns = sn.assert_found(
            r'WRITING EXTENDED SYSTEM TO OUTPUT FILE AT STEP', self.stdout,
        )
        self.perf_patterns = {
            'days_ns': sn.avg(sn.extractall(
                r'Info: Benchmark time: \S+ CPUs \S+ '
                r's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                self.stdout, 'days_ns', float,
            ))
        }

        self.maintainers = ['Lewih']

        self.tags = {'apps', 'namd', 'performance', 'vsc'}
        self.tags.add(f'{self.num_nodes}nodes')

    @run_before('run')
    def replace_launcher(self):
        self.job.launcher = getlauncher('srun')()

    def download_material(self):
        if int(self.num_nodes) in {1, 2}:
            self.prerun_cmds = [
                'wget https://www.ks.uiuc.edu/Research/namd/utilities/apoa1.zip',
                'unzip apoa1.zip',
            ]
            return 'apoa1'
        if int(self.num_nodes) > 2:
            self.prerun_cmds = [
                'wget https://www.ks.uiuc.edu/Research/namd/utilities/stmv.zip',
                'unzip stmv.zip',
            ]
            return 'stmv'


@rfm.simple_test
class Namd_CPUTest(NamdBaseTest):
    # NAMD non-SMP CPU test

    def __init__(self):
        self.time_limit = '20m'

        self.valid_systems = ['+cpu +default']
        self.valid_prog_environs = ['+default']
        super().__init__('cpu')

    @run_after('setup')
    def set_num_cpus(self):
        # for non-SMP, we want one task per CPU, so total tasks = num_nodes * num_cpus_per_node
        self.num_tasks = int(self.num_nodes) * self.current_partition.extras['num_cpus']
        self.num_cpus_per_task = 1

        configFile = self.download_material()
        self.executable = (
            f'$EBROOTNAMD/namd3 +setcpuaffinity {configFile}/{configFile}.namd'
        )
