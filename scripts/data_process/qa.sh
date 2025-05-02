WORK_DIR=/mnt/sdb/hanxu/projects/secretary-r1
TEMP_TYPE=t5
LOCAL_DIR=$WORK_DIR/data/qa/nq_hotpotqa_$TEMP_TYPE

#process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_train.py --local_dir $LOCAL_DIR --data_sources $DATA --template_type $TEMP_TYPE

#process multiple dataset search format test file
#DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_test.py --local_dir $LOCAL_DIR --data_sources $DATA --template_type $TEMP_TYPE