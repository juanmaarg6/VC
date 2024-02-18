import os
import numpy as np
import json
from .voc import parse_voc_annotation
from .yolo import create_yolov3_model, dummy_loss
from .utils.utils import makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from .callbacks import CustomModelCheckpoint, CustomTensorBoard
from .utils.multi_gpu_model import multi_gpu_model
import tensorflow.compat.v1 as tf
import keras
from keras.models import load_model

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor     = 'loss',
        min_delta   = 0.01,
        patience    = 7,
        mode        = 'min',
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5',
        monitor         = 'loss',
        verbose         = 1,
        save_best_only  = True,
        mode            = 'min',
        save_freq       = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        min_delta  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
    nb_class,
    anchors,
    max_box_per_image,
    max_grid, batch_size,
    warmup_batches,
    ignore_thresh,
    multi_gpu,
    saved_weights_name,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    backend_path,
    fine_tune = 0
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class,
                anchors             = anchors,
                max_box_per_image   = max_box_per_image,
                max_grid            = max_grid,
                batch_size          = batch_size//multi_gpu,
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale,
                finetune            = fine_tune == 4
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class,
            anchors             = anchors,
            max_box_per_image   = max_box_per_image,
            max_grid            = max_grid,
            batch_size          = batch_size,
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale,
            finetune            = fine_tune == 4
        )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights(backend_path, by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model

    # Fine-tuning
    if fine_tune == 1:
        for layer in train_model.layers:
            layer.trainable = False

        # Unfreeze large detection block (small objects)
        train_model.layers[254].trainable = True
        train_model.layers[242].trainable = True
        train_model.layers[237].trainable = True
        train_model.layers[234].trainable = True

        # Unfreeze medium detection block
        train_model.layers[252].trainable = True
        train_model.layers[241].trainable = True
        train_model.layers[217].trainable = True
        train_model.layers[214].trainable = True

        # Unfreeze small detection block
        train_model.layers[249].trainable = True
        train_model.layers[240].trainable = True
        train_model.layers[197].trainable = True
        train_model.layers[194].trainable = True

    elif fine_tune == 2:
        for layer in train_model.layers:
            layer.trainable = False

        # Unfreeze large detection block (small objects)
        train_model.layers[254].trainable = True
        train_model.layers[242].trainable = True
        train_model.layers[237].trainable = True
        train_model.layers[234].trainable = True

        # Unfreeze medium detection block
        train_model.layers[252].trainable = True
        train_model.layers[241].trainable = True
        train_model.layers[217].trainable = True
        train_model.layers[214].trainable = True

    elif fine_tune == 3:
        for layer in train_model.layers:
            layer.trainable = False

        # Unfreeze large detection block (small objects)
        train_model.layers[254].trainable = True
        train_model.layers[242].trainable = True
        train_model.layers[237].trainable = True
        train_model.layers[234].trainable = True

    optimizer = Adam(learning_rate=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model
