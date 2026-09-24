import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class NamdBaseTest(rfm.RunOnlyRegressionTest):
    # This test assumes NAMD3, MPI build, is the default version
    num_nodes = parameter([1, 2, 4], type=int)
    modules = ['NAMD']
    time_limit = '20m'
    tags = {'apps', 'namd', 'performance', 'vsc'}
    maintainers = ['Lewih']

    @run_after('init')
    def set_tags(self):
        self.tags = self.tags | {f'{self.num_nodes}nodes'}

    @run_after('init')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'days_ns': sn.avg(sn.extractall(
                r'Info: Benchmark time: \S+ CPUs \S+ '
                r's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                self.stdout, 'days_ns', float,
            ))
        }

    @run_before('run')
    def replace_launcher(self):
        # MPI launcher is a site property, see tests/micro/mpi/mpi_hello_world.py
        launcher = self.current_partition.extras.get('mpi_launcher', 'srun')
        self.job.launcher = getlauncher(launcher)()

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

    @sanity_function
    def assert_namd(self):
        return sn.assert_found(
            r'WRITING EXTENDED SYSTEM TO OUTPUT FILE AT STEP', self.stdout,
        )


@rfm.simple_test
class Namd_CPUTest(NamdBaseTest):
    # class-level so that -S valid_systems/valid_prog_environs=... can override them
    valid_systems = ['+cpu +default']
    valid_prog_environs = ['+default']

    @run_after('init')
    def set_descr(self):
        self.descr = (
            f'NAMD check on cpu, number of nodes: {self.num_nodes}, '
            f'apoa1 and stmv (4 nodes only)'
        )

    @run_after('setup')
    def set_num_cpus(self):
        # for non-SMP, we want one task per CPU, so total tasks = num_nodes * num_cpus_per_node
        self.num_tasks = int(self.num_nodes) * self.current_partition.extras['num_cpus']
        self.num_cpus_per_task = 1
        self.job.options = ['--exclusive']

        configFile = self.download_material()
        self.executable = (
            f'$EBROOTNAMD/namd3 +setcpuaffinity {configFile}/{configFile}.namd'
        )
