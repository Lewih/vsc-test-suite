import reframe as rfm
import reframe.utility.sanity as sn


class JuliaLinalgBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['+default']
        self.sanity_patterns = sn.assert_found(r'Julia version:*',
                                               self.stdout)
        self.modules = ['Julia']
        self.executable = 'julia'
        self.executable_opts = ['linalg.jl']
        self.tags = {'apps', 'julia', '1nodes', 'performance', 'vsc'}
        self.maintainers = ['Lewih']
        self.time_limit = '10m'


@rfm.simple_test
class JuliaLinalgTest(JuliaLinalgBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Test a few typical Julia LinAlg operations'
        self.valid_systems = ['+cpu +default']
        self.num_tasks_per_node = 1
        self.tags.add('performance')

        self.perf_patterns = {
            'dot': sn.extractsingle(
                r'^Dotted two 4096 x 4096 matrices in\s+(?P<dot>\S+)\s+s',
                self.stdout, 'dot', float),
            'cholesky': sn.extractsingle(
                r'^Cholesky decomposition of a 4096 x 4096 matrix in'
                r'\s+(?P<cholesky>\S+)\s+s',
                self.stdout, 'cholesky', float),
            'lu': sn.extractsingle(
                r'^LU decomposition of a 4096 x 4096 matrix in'
                r'\s+(?P<lu>\S+)\s+s',
                self.stdout, 'lu', float),
        }

    @run_after('setup')
    def set_num_cpus(self):
        self.num_cpus_per_task = self.current_partition.extras['num_cpus']
        self.executable_opts = ['linalg.jl', str(self.num_cpus_per_task)]
        jobid = '$SLURM_JOBID'
        self.env_vars = {'JULIA_DEPOT_PATH': f'$VSC_SCRATCH/rfm_julia_{jobid}'}
        self.postrun_cmds = [f'rm -rf $VSC_SCRATCH/rfm_julia_{jobid}']
        self.job.options = ['--exclusive']
        self.sanity_patterns = sn.and_(
            sn.assert_found(r'BLAS num threads:', self.stdout),
            self.sanity_patterns,
        )
