#!/bin/bash -l
# Wrapper around reframe for the VSC test suite; all arguments are passed on.

module load ReFrame || { echo "run.sh: cannot load the ReFrame module" >&2; exit 1; }

export RFM_CONFIG_FILES=$(dirname $0)/config_vsc.py
export RFM_CHECK_SEARCH_PATH=$(dirname $0)/tests
export RFM_OUTPUT_DIR=$VSC_SCRATCH/reframe
export RFM_PREFIX=$VSC_SCRATCH/reframe
export RFM_CHECK_SEARCH_RECURSIVE=true
export RFM_SAVE_LOG_FILES=true

# KU Leuven: genius, wice and mindwell are all reached from the Genius login
# nodes and a job is routed with --clusters=<name> (set in sites/leuven.py).
# ReFrame < 4.10 does not pass that flag on to sacct when polling, so a job
# on a non-default cluster would never be seen as finished: make the requested
# cluster the default for the whole run (sbatch/sacct/squeue/scancel honour it).
# ReFrame >= 4.10 handles this itself (sched_options in sites/leuven.py); this
# block can go once every site runs at least that version.
system=""
prev=""
for a in "$@"; do
    case "$a" in
        --system=*) system="${a#--system=}" ;;
        *) [[ "$prev" == "--system" ]] && system="$a" ;;
    esac
    prev="$a"
done
case "${system%%:*}" in
    genius|wice|mindwell) export SLURM_CLUSTERS="${system%%:*}" ;;
esac

# if --list is passsed, do not run
if [[ $* == *--list* ]]; then
    reframe --keep-stage-files --list "$@"
    exit 0
fi

# if --dry-run is passed, do not run
if [[ $* == *--dry-run* ]]; then
    reframe --keep-stage-files --dry-run "$@"
    exit 0
fi

reframe --keep-stage-files --run  "$@"
#rm $(dirname $0)/reframe.out $(dirname $0)/reframe.log
