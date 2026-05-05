import reframe as rfm
import reframe.utility.sanity as sn


class GPUJobBase(rfm.RunOnlyRegressionTest):
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1
    time_limit = '5m'
    tags = {'gpu', 'micro', 'vsc'}

    @run_before('run')
    def request_one_gpu(self):
        self.extra_resources = {'gpu': {'num_gpus': '1'}}


@rfm.simple_test
class GPUJobTest_Nvidia(GPUJobBase):
    descr = 'GPU job submission and visibility check on Nvidia partition'
    valid_systems = ['+gpu +nvidia']
    valid_prog_environs = ['+default']
    executable = 'nvidia-smi -L'

    @sanity_function
    def assert_gpu_visible(self):
        return sn.assert_found(r'GPU 0:', self.stdout)


@rfm.simple_test
class GPUJobTest_AMD(GPUJobBase):
    descr = 'GPU job submission and visibility check on AMD partition'
    valid_systems = ['+gpu +amd']
    valid_prog_environs = ['+default']
    executable = 'rocm-smi --showid'

    @sanity_function
    def assert_gpu_visible(self):
        return sn.assert_found(r'GPU\[0\]', self.stdout)
