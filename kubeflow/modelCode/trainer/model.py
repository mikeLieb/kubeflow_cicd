
##AWS
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import glob
import logging
def parse_function(example_proto):
    features={"movieTitle":tf.FixedLenFeature((), tf.string),
              "genres":tf.FixedLenFeature((), tf.string),
             "popularity":tf.FixedLenFeature((), tf.float32),
              "vote_count":tf.FixedLenFeature((), tf.float32),
              "revenue":tf.FixedLenFeature((), tf.float32),
              "yr_diff":tf.FixedLenFeature((), tf.float32),
              "target":tf.FixedLenFeature((), tf.float32),
              
             }
    parsed_features = tf.parse_example(example_proto, features)
    t=tf.string_split(parsed_features["genres"],',')
    parsed_features["t"]=t
    return parsed_features, parsed_features['target']

def input_fn(path,bucket,num_epochs, batch_size,buffer_size,prefetch_buffer_size,shuffle=True):
    data_lines=tf.data.TFRecordDataset([file for file in tf.gfile.Glob('gs://{}/data/{}/*'.format(bucket,path))])
    if shuffle:
        data_lines= data_lines.shuffle(buffer_size=buffer_size)
    data_iter= data_lines.prefetch(prefetch_buffer_size).repeat(num_epochs).batch(batch_size).map(parse_function).make_one_shot_iterator()
    feats = data_iter.get_next()
    return feats

def serving_input_fn():
    receiver_tensor = {}
    receiver_tensor["movieTitle"] = tf.placeholder(tf.string, shape=[None], name="did")
    receiver_tensor["target"] = tf.placeholder(tf.float32, shape=[None], name="target")
    receiver_tensor["genres"] = tf.placeholder(tf.string, shape=[None], name="genres")
    receiver_tensor["popularity"] = tf.placeholder(tf.float32, shape=[None], name="popularity")
    receiver_tensor["vote_count"] = tf.placeholder(tf.float32, shape=[None], name="vote_count")
    receiver_tensor["revenue"] = tf.placeholder(tf.float32, shape=[None], name="revenue")
    receiver_tensor["yr_diff"] = tf.placeholder(tf.float32, shape=[None], name="yr_diff")
    
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in receiver_tensor.items()
    }
    features["t"]=tf.string_split(tf.squeeze(features["genres"]),',')
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

cat_indy= tf.feature_column.indicator_column(tf.feature_column. \
                    categorical_column_with_vocabulary_list("t",["Crime","Romance","Thriller","Adventure",
                                                        "Drama","War","Family","Fantasy","History","Mystery",
                                                    "Music","Science Fiction","Horror","Western","Comedy","Action"],
                                      num_oov_buckets=1)) 

numeric= [tf.feature_column.numeric_column(key=col) for col in {"popularity","vote_count","yr_diff"}]
feats= numeric+[cat_indy]


def nnet(input_):
  #feature_columns= indic_function()
  #nums= [tf.feature_column.numeric_column(key="_c1")]
  dense_tensor= tf.feature_column.input_layer(input_, feats)
  he_init = tf.variance_scaling_initializer()
  #batch_layers_1= tf.layers.batch_normalization(dense_tensor, training=True)
  layer_1 = tf.layers.dense(dense_tensor, 25, activation=tf.nn.relu,kernel_initializer=he_init)
  layer_2 = tf.layers.dense(layer_1, 15, activation=tf.nn.relu,kernel_initializer=he_init)
  out_put_layer = tf.layers.dense(layer_2,1)
  return out_put_layer

def model_fn(features, labels, params,mode):
    values = nnet(features)
    idx= features["movieTitle"]
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction': values,
            'key':idx
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions,export_outputs=export_outputs)

    loss_op = tf.losses.mean_squared_error(labels, values[0])

    gradients = tf.gradients(loss_op, tf.trainable_variables())
    
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])#.00001
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_optimizer = optimizer.apply_gradients(list(zip(gradients, tf.trainable_variables())),
                                                global_step=tf.train.get_global_step())

        acc_op=tf.metrics.root_mean_squared_error(labels, values)

        tf.summary.scalar('accuracy_rate', acc_op[1])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=values,
            loss=loss_op,
            train_op=train_optimizer,
            eval_metric_ops={"rmse": tf.metrics.root_mean_squared_error(labels, values)})
        return estim_specs 
    if mode == tf.estimator.ModeKeys.EVAL:
        predicted_indices = values
        eval_metric_ops = {
            'rmse': tf.metrics.root_mean_squared_error(labels, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode,loss=loss_op,eval_metric_ops=eval_metric_ops)

def train_and_evaluate(args):
    params={}
    params["bucket"]=args.bucket
    params["train_path"]=args.train_path
    params["train_path"]=args.train_path
    params["eval_path"]=args.eval_path
    params["num_epochs"]=int(args.num_epochs)
    params["learning_rate"]=float(args.learning_rate)
    params["train_batch_size"]=int(args.train_batch_size)
    params["buffer_size"]=int(args.buffer_size)
    params["val_batch_size"]=int(args.val_batch_size)
    params["max_steps"]=int(args.max_steps)
    params["eval_steps"]=int(args.eval_steps)
    params["prefetch_buffer_size"]=int(args.prefetch_buffer_size)
    params["model_path"]=args.model_path
    def train_input():
        return input_fn(
            path=params["train_path"],
            bucket=params["bucket"],
            num_epochs=params["num_epochs"],
            batch_size=params["train_batch_size"],
             prefetch_buffer_size=params["prefetch_buffer_size"],
            buffer_size= params["buffer_size"])
    exporter=tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=1)
    def eval_input():
        return input_fn(
            path=params["eval_path"],
            bucket=params["bucket"],
            num_epochs=params["num_epochs"],
            batch_size=params["val_batch_size"],
             prefetch_buffer_size=params["prefetch_buffer_size"],
            shuffle=False,
            buffer_size= params["buffer_size"])
    train_spec = tf.estimator.TrainSpec(
        train_input, 
        max_steps=params["max_steps"]) 
    
    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=params["eval_steps"],
        exporters=exporter,
        throttle_secs=30,
        name='tf_income_proto1')
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config= tf.estimator.RunConfig(
                save_checkpoints_steps = 100,
                keep_checkpoint_max = 2,
                tf_random_seed = 87424), model_dir= "{model_path}".format(model_path=params["model_path"]))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator
