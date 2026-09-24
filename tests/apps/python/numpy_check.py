import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NumpyTest(rfm.RunOnlyRegressionTest):
    # class-level so that -S valid_systems/valid_prog_environs=... can override them
    valid_systems = ['+cpu +default']
    valid_prog_environs = ['+default']
    modules = ['SciPy-bundle']
    descr = 'Test a few typical numpy operations'
    executable = 'python3'
    executable_opts = ['np_ops.py']
    time_limit = '20m'
    tags = {'apps', 'python', 'numpy', 'performance', 'vsc'}
    maintainers = ['Lewih']

    @run_after('init')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'dot': sn.extractsingle(
                r'^Dotted two \S* matrices in\s+(?P<dot>\S+)\s+s',
                self.stdout, 'dot', float),
            'svd': sn.extractsingle(
                r'^SVD of a \S* matrix in\s+(?P<svd>\S+)\s+s',
                self.stdout, 'svd', float),
            'cholesky': sn.extractsingle(
                r'^Cholesky decomposition of a \S* matrix in'
                r'\s+(?P<cholesky>\S+)\s+s',
                self.stdout, 'cholesky', float),
            'eigendec': sn.extractsingle(
                r'^Eigendecomposition of a \S* matrix in'
                r'\s+(?P<eigendec>\S+)\s+s',
                self.stdout, 'eigendec', float),
            'inv': sn.extractsingle(
                r'^Inversion of a \S* matrix in\s+(?P<inv>\S+)\s+s',
                self.stdout, 'inv', float),
        }

    @run_after('setup')
    def set_num_cpus(self):
        # cap the threading at 6 cores; the test is not designed to scale past that
        ncpus = min(6, self.current_partition.extras['num_cpus'])
        self.num_cpus_per_task = ncpus
        self.env_vars = {
            'OMP_NUM_THREADS': str(ncpus),
            'MKL_NUM_THREADS': str(ncpus),
        }
        self.job.options = ['--exclusive']

    @sanity_function
    def assert_numpy(self):
        return sn.assert_found(r'Numpy version:\s+\S+', self.stdout)
