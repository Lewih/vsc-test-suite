import os
import reframe as rfm
import reframe.utility.sanity as sn
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from envars_list import envars

@rfm.simple_test
class VSCEnvTest(rfm.RunOnlyRegressionTest):
    descr = "test environment variable "
    envar = parameter(envars.keys())
    valid_systems = ["*:local", "*:default-node"]
    valid_prog_environs = ["standard"]
    time_limit = '10m'
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1
    maintainers = ["smoors", "Lewih"]
    tags = {"vsc", "cue", "env"}

    @run_after('init')
    def set_param(self):
        self.descr += self.envar
        exe = envars[self.envar]['exe']
        # load Reframe to expose archspec in python path 
        self.executable = "ml ReFrame; python3 -c 'import os;{}'".format('\n'.join(exe))

    @sanity_function
    def assert_env(self):
        return sn.assert_found(r'^True$', self.stdout)
