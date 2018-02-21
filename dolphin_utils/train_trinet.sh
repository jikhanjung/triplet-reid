DATAFOLDER=Bottle_unflipped
#DATAFOLDER=subfoldered_small
RESULTSDIR=bottle_results_unflipped
BASEDIR=fin-clustering
CONTAINERNAME=classification-dev
CHECKPOINT=checkpoint-10000
GALLERY_CSV=train_gallery.csv
TRAIN_CSV=train.csv


nvidia-docker exec -it $CONTAINERNAME python3  triplet-reid/train.py \
     --train_set $BASEDIR/$DATAFOLDER/$TRAIN_CSV \
     --image_root $BASEDIR/$DATAFOLDER \
     --experiment_root $BASEDIR/$RESULTSDIR \
     --initial_checkpoint $BASEDIR/notest_results/checkpoint-9000 \
     --batch_p 21 \
     --batch_k 4 \
     --net_input_width 224 \
     --net_input_height 224 \
     --crop_augment \
     --pre_crop_height 230 \
     --pre_crop_width 230 \
     --detailed_logs \
     --train_iterations 16000 \
     --decay_start_iteration 8000 \
     --checkpoint_frequency 1000   \
     --max_rotation 35 \
     --saturation_range 0.4 1.2 \
     --hue_range 0.2 \
     --resume
