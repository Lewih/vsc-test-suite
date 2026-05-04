import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NumpyTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Test a few typical numpy operations'
        self.valid_systems = ['+cpu +default']
        # SciPy-bundle is part of the foss toolchain
        self.valid_prog_environs = ['+default']
        self.modules = ['SciPy-bundle']
        self.time_limit = '20m'

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

        self.sanity_patterns = sn.assert_found(r'Numpy version:\s+\S+',
                                               self.stdout)
        self.executable = 'python3'
        self.executable_opts = ['np_ops.py']
        self.tags = {'apps', 'python', 'numpy', 'performance', 'vsc'}
        self.maintainers = ['Lewih']

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
