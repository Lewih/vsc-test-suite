# vsc-test-suite

ReFrame test suite for VSC (Flemish Supercomputer Centre) clusters.

## Requirements

- ReFrame ≥ 4.6 available as a module (e.g. `module load ReFrame`)
- archspec available as a module
- Python 3

## How to run

```bash
cd vsc-test-suite
./run.sh [reframe options]
```

Any arguments are passed directly to `reframe`. The `--run` flag is already set by the script.

Examples:

```bash
# List all available tests
./run.sh --list

# Run all tests on the current cluster
./run.sh

# Run tests tagged 'cpu' on a specific system
./run.sh --system=genius:default --tag=cpu

# Run only MPI tests
./run.sh --system=genius:mpi-job
```

## Output location

Results, logs, and stage files are written to `$VSC_SCRATCH/reframe`.

## Supported clusters

| Cluster   | Site       |
|-----------|------------|
| hydra     | VUB        |
| hortense  | UGent      |
| genius    | KULeuven   |
| vaughan   | UAntwerpen |
| leibniz   | UAntwerpen |
| breniac   | UAntwerpen |

## Feature flags

Partitions and environments are tagged with feature flags. Tests select them via `valid_systems` and `valid_prog_environs` instead of hard-coding cluster names.

### Partition features

| Flag       | Meaning                          |
|------------|----------------------------------|
| `+cpu`     | Any CPU compute partition        |
| `+gpu`     | Any GPU compute partition        |
| `+login`   | Login node                       |
| `+default` | Default (generic) CPU partition  |
| `+nvidia`  | NVIDIA GPU partition             |
| `+amd`     | AMD GPU partition                |

### Environment features

| Flag      | Toolchain loaded                  |
|-----------|-----------------------------------|
| `+default`| No toolchain (bare environment)   |
| `+foss`   | foss/2023a – foss/2025a           |
| `+intel`  | intel/2023a – intel/2025a         |
| `+mpi`    | foss or intel with MPI + FFT libs |
| `+fftw`   | Same as `+mpi` (includes FFTW)    |
| `+cuda`   | CUDA/12.8.0                       |

### Example test selectors

```python
# Run on any CPU default partition with no special toolchain
valid_systems = ['+cpu +default']
valid_prog_environs = ['+default']

# Run only where foss is available, with MPI
valid_systems = ['+cpu +default']
valid_prog_environs = ['+foss +mpi']

# Run on NVIDIA GPU partitions
valid_systems = ['+gpu +nvidia']
valid_prog_environs = ['+cuda']
```

## Test layout

```
tests/
  apps/
    gaussian/     Gaussian application test
    julia/        Julia linear algebra benchmark
    matlab/       MATLAB linear algebra benchmark
    namd/         NAMD MD simulation (SMP and multi-node)
    python/       NumPy/SciPy performance check
  cue/            Common User Environment checks
                  (tools, env vars, shared filesystems, job submission)
  gpu/            GPU burn stress test
  micro/
    basic/        Hello-world job submission sanity check
    mpi/          MPI hello-world sanity check
```

## Configuration

Site configuration is in [config_vsc.py](config_vsc.py). Each partition declares:

- `features` — list of flags tests can match against
- `extras['num_cpus']` — cores available per node (used by tests instead of hard-coded site values)
- `extras['num_gpus']` — GPUs per node (GPU partitions only)
