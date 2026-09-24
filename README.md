# vsc-test-suite

ReFrame test suite for VSC (Flemish Supercomputer Centre) clusters.

## Requirements

- ReFrame ≥ 4.9 available as a module (e.g. `module load ReFrame`)
- archspec available as a module
- Python 3

## How to run

```bash
cd vsc-test-suite
./run.sh [reframe options]
```

Any arguments are passed directly to `reframe`. The `--run` flag is already set by the script unless `--list` is passed.

Examples:

```bash
# List all available tests
./run.sh --list

# Run all tests on the current cluster
./run.sh

# Pass a job option (e.g. when using the fallback system) 
./run.sh -J '-A ap_calcua_staff'

# Run tests tagged 'cue' and not tagged 'amd' on a specific system
./run.sh -t cue -T amd

# Run only MPI tests
./run.sh --system=vsc_generic:mpi-job
```

## Output location

Results, logs, and stage files are written to `$VSC_SCRATCH/reframe`.

## Supported clusters

| Cluster   | Site       | Config file        |
|-----------|------------|--------------------|
| hydra     | VUB        | `sites/brussel.py` |
| sofia     | VUB        | `sites/brussel.py` |
| hortense  | UGent      | `sites/gent.py`    |
| genius    | KULeuven   | `sites/leuven.py`  |
| wice      | KULeuven   | `sites/leuven.py`  |
| mindwell  | KULeuven   | `sites/leuven.py`  |
| vaughan   | UAntwerpen | `sites/antwerp.py` |
| leibniz   | UAntwerpen | `sites/antwerp.py` |
| breniac   | UAntwerpen | `sites/antwerp.py` |

A generic `vsc_generic` fallback system also matches any unlisted VSC host
so tests can be tried out on a new cluster without touching the config first.

The system is autodetected by matching the `hostnames` patterns against all
names of the login node (see `vsc_host_names` in `config_vsc.py`).

The three KU Leuven clusters share the Genius login nodes, so hostname
autodetection always picks `genius` (GPU nodes only); use `--system=wice` or
`--system=mindwell` for the others. With ReFrame >= 4.10 the partitions'
`sched_options` make job polling (`sacct`) use the right cluster; for older
versions `run.sh` exports `SLURM_CLUSTERS` for the run instead.

The Tier-1 clusters (hortense, sofia) need a compute project on every job.
For VSC staff the account is picked from their group membership at load time
(`sites/gent.py`, `sites/brussel.py`); everyone else passes it on the command
line: `./run.sh -J '-A <project>'`.

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

## Application versions

The application tests load their site's default module (`NAMD`, `Julia`,
`MATLAB`, `SciPy-bundle`). Test another build with `-P`:

```bash
./run.sh -n Namd_CPUTest -P Namd_CPUTest.version=NAMD/3.0-foss-2024a-mpi
```

Test classes: `Namd_CPUTest`, `JuliaLinalgTest`, `MatlabLinalgTest`, `NumpyTest`.

## Test layout

```
tests/
  apps/
    gaussian/     Gaussian application test
    julia/        Julia linear algebra benchmark
    matlab/       MATLAB linear algebra benchmark
    namd/         NAMD MD simulation (multi-node MPI)
    python/       NumPy/SciPy performance check
  cue/            Common User Environment checks
                  (tools, env vars, shared filesystems, job submission)
  gpuburn/        GPU burn stress test
  micro/
    basic/        Hello-world job submission sanity check
    gpu/          GPU job submission and visibility check (NVIDIA + AMD)
    mpi/          MPI hello-world sanity check
```

## Configuration

The top-level [config_vsc.py](config_vsc.py) holds the shared bits —
environments, logging, the `cpu_env_list` toolchain list, and the
`vsc_generic` fallback system. Per-cluster definitions live in
[sites/](sites/), one file per city; see [sites/README.md](sites/README.md)
for the directory contract.

Sites are auto-discovered: dropping a new file in `sites/` that exports a
`systems` list is enough to register a new cluster — no edits to
`config_vsc.py` needed. A site may also export a `general` list of ReFrame
`general` entries scoped with `target_systems` (KU Leuven uses this for
`use_login_shell`).

Each partition declares:

- `features` — list of flags tests can match against
- `extras['num_cpus']` — cores available per node (used by tests instead of
  hard-coded site values)
- `extras['num_gpus']` — GPUs per node (GPU partitions only)
- `extras['cpus_per_gpu']` — cores a job must request per GPU where the site
  enforces a fixed ratio (e.g. 24 on sofia `zen4_h200`; default 1)
- `extras['mpi_launcher']` — the only MPI launcher that works on the
  partition (e.g. `mpirun` on KU Leuven, whose Slurm has no PMI support).
  Unset means both `srun` and `mpirun` work: the MPI hello-world test then
  runs with each, and the other MPI tests use `srun`
