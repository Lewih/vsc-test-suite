import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class JuliaLinalgTest(rfm.RunOnlyRegressionTest):
    # class-level so that -S valid_systems/valid_prog_environs=... can override them
    valid_systems = ['+cpu +default']
    valid_prog_environs = ['+default']
    # module under test; override with -P JuliaLinalgTest.version=Julia/1.11.1
    version = parameter(['Julia'], type=str)
    descr = 'Test a few typical Julia LinAlg operations'
    executable = 'julia'
    executable_opts = ['linalg.jl']
    num_tasks_per_node = 1
    time_limit = '10m'
    tags = {'apps', 'julia', '1nodes', 'performance', 'vsc'}
    maintainers = ['Lewih']

    @run_after('init')
    def set_module(self):
        self.modules = [self.version]

    @run_after('init')
    def set_perf_patterns(self):
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
        # cap the BLAS threading at 10 cores; the test is not designed to scale past that
        self.num_cpus_per_task = min(10, self.current_partition.extras['num_cpus'])
        self.executable_opts = ['linalg.jl', str(self.num_cpus_per_task)]
        jobid = '$SLURM_JOBID'
        self.env_vars = {'JULIA_DEPOT_PATH': f'$VSC_SCRATCH/rfm_julia_{jobid}'}
        self.postrun_cmds = [f'rm -rf $VSC_SCRATCH/rfm_julia_{jobid}']
        self.job.options = ['--exclusive']

    @sanity_function
    def assert_julia(self):
        return sn.all([
            sn.assert_found(r'Julia version:*', self.stdout),
            sn.assert_found(r'BLAS num threads:', self.stdout),
        ])
