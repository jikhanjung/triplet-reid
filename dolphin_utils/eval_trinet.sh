#!/bin/bash
#This is for evaluating a trained model
DATAFOLDER=field_test
#DATAFOLDER=subfoldered_small
RESULTSDIR=bottle_results_unflipped
BASEDIR=fin-clustering
CONTAINERNAME=classification-dev
CHECKPOINT=checkpoint-1000
SET=train
#SET=val
GALLERY_CSV=_gallery.csv
GALLERY_CSV=$SET$GALLERY_CSV
QUERY_CSV=_query.csv
QUERY_CSV=$SET$QUERY_CSV
echo $QUERY_CSV

# Embed the query images
nvidia-docker exec -it $CONTAINERNAME  python3  triplet-reid/embed.py \
    --experiment_root $BASEDIR/$RESULTSDIR \
    --image_root $BASEDIR/$DATAFOLDER \
    --dataset $BASEDIR/$DATAFOLDER/$QUERY_CSV \
    --filename query_embeddings.h5 \
    --checkpoint $CHECKPOINT

# Embed the gallery images
nvidia-docker exec -it $CONTAINERNAME  python3  triplet-reid/embed.py \
    --experiment_root $BASEDIR/$RESULTSDIR \
    --image_root $BASEDIR/$DATAFOLDER \
    --dataset $BASEDIR/$DATAFOLDER/$GALLERY_CSV \
    --filename gallery_embeddings.h5 \
    --checkpoint $CHECKPOINT

# Evaluate the embeddings
nvidia-docker exec -it $CONTAINERNAME  python3  triplet-reid/evaluate.py \
    --excluder diagonal \
    --query_dataset $BASEDIR/$DATAFOLDER/$QUERY_CSV \
    --query_embeddings $BASEDIR/$RESULTSDIR/query_embeddings.h5 \
    --gallery_dataset $BASEDIR/$DATAFOLDER/$GALLERY_CSV \
    --gallery_embeddings $BASEDIR/$RESULTSDIR/gallery_embeddings.h5 \
    --metric euclidean \
    --filename AUGMENT_RESULTS.json
