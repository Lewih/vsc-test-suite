module load ReFrame/4.6.1

export RFM_CONFIG_FILES=$(dirname $0)/config_vsc.py
export RFM_CHECK_SEARCH_PATH=$(dirname $0)/tests
export RFM_OUTPUT_DIR=$VSC_SCRATCH/reframe
export RFM_PREFIX=$VSC_SCRATCH/reframe
export RFM_CHECK_SEARCH_RECURSIVE=true
export RFM_SAVE_LOG_FILES=true

reframe --keep-stage-files --run  "$@"
#rm $(dirname $0)/reframe.out $(dirname $0)/reframe.log
