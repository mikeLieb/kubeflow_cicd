#!/bin/bash


CODEDIR=/modeling/kubeflow_cicd/kubeflow/modelCode
BUCKET=$1
TRAIN_PATH=$2
EVAL_PATH=$3
date=$(date +%s)
OUTDIR=gs://${BUCKET}/movies/train_${date}
REGION=us-central1
JOBNAME=aiplatform_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
    gcloud ai-platform jobs submit training $JOBNAME \
   --region=$REGION \
   --module-name=trainer.task \
   --package-path=${CODEDIR}/trainer \
   --job-dir=$OUTDIR \
   --staging-bucket=gs://$BUCKET \
   --scale-tier custom \
   --master-machine-type standard \
   --runtime-version 1.15 \
   --python-version 3.7 \
   --stream-logs \
   -- \
   --model_path=$OUTDIR \
   --max_steps=1000 \
   --bucket=$BUCKET \
   --train_path=$TRAIN_PATH \
   --eval_path=$EVAL_PATH
   
echo $OUTDIR > /output.txt
