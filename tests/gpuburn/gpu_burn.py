import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GPU_Burn_nvidia(rfm.RunOnlyRegressionTest):
    descr = 'GPU burn test on nvidia node'
    valid_systems = ['+gpu +nvidia -deprecated'] # only on non-deprecated gpu nodes, as the test is not expected to run on old gpu nodes
    valid_prog_environs = ['+cuda']
    env_vars = {'CUDAPATH': '$EBROOTCUDA'}
    time_limit = '10m'
    prerun_cmds = [
        'wget https://github.com/wilicc/gpu-burn/archive/refs/heads/master.zip',
        'unzip master.zip && mv gpu-burn-master gpu-burn',
        'cd gpu-burn',
        'make',
    ]
    executable = 'srun --output=rfm_GPUBURN_nvidia_node-%N.out ./gpu_burn 20'
    tags = {'gpu', 'burn', 'performance', 'vsc'}
    num_devices = 0
    num_tasks = 0
    num_tasks_per_node = 1
    @run_before('run')
    def set_options(self):
        extras = self.current_partition.extras
        self.num_devices = extras['num_gpus']
        self.num_cpus_per_task = extras['num_cpus']//self.num_devices
        self.extra_resources = {'gpu': {'num_gpus': str(self.num_devices)}} # gpus per node, not total gpus
        self.descr = (
            f'Nvidia gpu burn test on {self.current_system.name} '
            f'with {self.num_devices} gpus'
        )

    @sanity_function
    def assert_job(self):
        result = True
        for n in sorted(self.job.nodelist):
            node = n.split('.')[0]
            out = f'{self.stagedir}/gpu-burn/rfm_GPUBURN_nvidia_node-{node}.out'
            result = sn.and_(
                sn.and_(sn.assert_found(r'OK', out),
                        sn.assert_not_found(r'FAULTY', out)),
                result,
            )
        return result

    @performance_function('Gflop/s')
    def get_gflops(self, device=0, node=None):
        # take starting from item -1 (last match)
        return sn.extractsingle(
            r'\((?P<gflops>\S+) Gflop/s\)',
            f'{self.stagedir}/gpu-burn/rfm_GPUBURN_nvidia_node-{node}.out',
            'gflops', float, item=(-device - 1),
        )

    @run_before('performance')
    def set_perf_variables(self):
        '''Build the dictionary with all the performance variables.'''
        self.perf_variables = {}
        if self.job.nodelist:
            for n in self.job.nodelist:
                node = n.split('.')[0]
                for x in range(self.num_devices):
                    idx = self.num_devices - x - 1
                    self.perf_variables[f'{node}_device{idx}'] = (
                        self.get_gflops(device=self.num_devices - x, node=node)
                    )
