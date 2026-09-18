import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class MPIHelloWorldTest(rfm.RegressionTest):
    descr = '''Compile and execute a simple Hello World MPI example. A job is
    launched on 2 nodes with 3 MPI processes per node. It is checked that each
    process prints a Hello World line and that indeed two nodes were used.'''
    valid_systems = ['+cpu +default']
    valid_prog_environs = ['+mpi']
    maintainers = ['stevenvdb']
    time_limit = '10m'
    num_tasks = 6
    num_tasks_per_node = 3
    num_cpus_per_task = 1
    executable = 'mpi_hello_world'
    sourcesdir = 'src_mpi_hello_world'
    launcher = parameter(['srun', 'mpirun'])
    tags = {'vsc', 'micro', 'mpi'}

    @run_after('setup')
    def skip_unsupported_launcher(self):
        # Both launchers are exercised, unless the partition declares the only
        # one that works there as extras['mpi_launcher'] (KU Leuven: mpirun,
        # its Slurm has no PMI support).
        forced = self.current_partition.extras.get('mpi_launcher')
        self.skip_if(forced and self.launcher != forced,
                     f'{self.launcher} not supported here, only {forced}')

    @run_before('run')
    def set_mpi_launcher(self):
        # Default partitions use launcher='local' so serial tests run without
        # a wrapper; MPI tests pick their launcher themselves.
        self.job.launcher = getlauncher(self.launcher)()

    @sanity_function
    def assert_number_of_hellos(self):
        # Check that the number of "Hello world" print statements equals the
        # total number of processes
        num_hellos = sn.len(sn.findall(r'^Hello world', self.stdout))
        num_hellos_ok = sn.assert_eq(num_hellos, self.num_tasks)

        # Check that the number of different hosts that printed "Hello world"
        # equals the number of nodes. The output contains lines like this:
        # Hello world from processor r25i27n07, rank 1 out of 6 processors
        # The regular expression extracts the part between processor and ,
        regex = r'^Hello world from processor (?P<node>\S+), rank'
        num_nodes = sn.count_uniq(sn.extractall(regex, self.stdout, 'node', str))
        num_nodes_ok = sn.assert_eq(num_nodes, self.num_tasks //
                                               self.num_tasks_per_node)
        return sn.assert_true(sn.and_(num_hellos_ok, num_nodes_ok))
