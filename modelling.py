# EfficientDET for Object Detection, Classification, and Segmentation

# 5.0 EFFICIENTDET UTILITY FUNCTIONS & CONSTANTS
IMAGE_SHAPE = (train_df.iloc[0].height, train_df.iloc[0].width, 3)
INPUT_SHAPE = (640,640,3)
SEG_SHAPE = (INPUT_SHAPE[0]//4, INPUT_SHAPE[1]//4, 1)
MODEL_LEVEL = "d1"
MODEL_NAME = f"efficientdet-{MODEL_LEVEL}"
BATCH_SIZE = 8
N_EVAL = 50
N_TRAIN = len(train_df)-N_EVAL
N_TEST = len(ss_df)
DEBUG = N_TEST==3
N_EPOCH = 40
N_EX_PER_REC = 280
CLASS_LABELS = list(train_df.cell_type.unique())
N_CLASSES_OD = len(CLASS_LABELS)+1 # Background + 3 Cell Types
N_CLASSES_SEG = 2 # Background + Foreground (Cells)
MAX_N_INSTANCES = int(100*np.ceil(train_df.bboxes.apply(len).max()/100))

# Whether or not we train from scratch or load
DO_TRAIN=False
PRETRAINED_MODEL_DIR="/kaggle/input/model-weights-40-epoch-efficientdet-d1-640"

print("\n ... HYPERPARAMETER CONSTANTS ...")
print(f"\t--> MODEL NAME         : {MODEL_NAME}")
print(f"\t--> BATCH SIZE         : {BATCH_SIZE}")
print(f"\t--> IMAGE SHAPE        : {IMAGE_SHAPE}")
print(f"\t--> INPUT SHAPE        : {INPUT_SHAPE}")
print(f"\t--> SEGMENTATION SHAPE : {SEG_SHAPE}")

# 5.1 LOAD EFFICIENTDET MODEL AND INITIALIZE

config = hparams_config.get_efficientdet_config(MODEL_NAME)
KEY_CONFIGS = [
    "name", "image_size", "num_classes", "seg_num_classes", "heads", "train_file_pattern",
    "val_file_pattern", "model_name", "model_dir", "pretrained_ckpt", "batch_size", "eval_samples",
    "num_examples_per_epoch", "num_epochs", "steps_per_execution", "steps_per_epoch", 
    "profile", "val_json_file", "max_instances_per_image", "mixed_precision", 
    "learning_rate", "lr_warmup_init", "mean_rgb", "stddev_rgb","scale_range",
              ]

for k in config.keys():
    if k=="model_optimizations":
        continue
    elif k=="nms_configs":
        for _k, _v in dict(config[k]).items():
            print(f"PARAMETER: {'     ' if _k not in KEY_CONFIGS else ' *** '}nms_config_{_k: <16}  ---->    VALUE:  {_v}")
        
    else:
        print(f"PARAMETER: {'     ' if k not in KEY_CONFIGS else ' *** '}{k: <27}  ---->    VALUE:  {config[k]}")

DO_ADV_PROP=True
MODEL_DIR = f"/kaggle/working/{MODEL_NAME}-finetune"

if TPU:
    TFRECORD_DIR = os.path.join(KaggleDatasets().get_gcs_path('effdet-d5-dataset-sartorius'), "tfrecords")
else:
    TFRECORD_DIR = "/kaggle/working/tfrecords"

os.makedirs(MODEL_DIR, exist_ok=True)
config = hparams_config.get_efficientdet_config(MODEL_NAME)
overrides = dict(
    train_file_pattern=os.path.join(TFRECORD_DIR, "train", "*.tfrec"),
    val_file_pattern=os.path.join(TFRECORD_DIR, "val", "*.tfrec"),
    test_file_pattern=os.path.join(TFRECORD_DIR, "test", "*.tfrec"),
    model_name=MODEL_NAME,
    model_dir=MODEL_DIR,
    pretrained_ckpt=MODEL_NAME,
    batch_size=BATCH_SIZE,
    eval_samples=N_EVAL,
    num_examples_per_epoch=N_TRAIN,
    num_epochs=N_EPOCH,
    steps_per_execution=1,
    steps_per_epoch=N_TRAIN//BATCH_SIZE,
    profile=None, val_json_file=None,
    heads = ['object_detection', 'segmentation'],
    image_size = INPUT_SHAPE[:-1],
    num_classes = N_CLASSES_OD,
    seg_num_classes = N_CLASSES_SEG,
    max_instances_per_image = MAX_N_INSTANCES,
    input_rand_hflip=False, jitter_min=0.99, jitter_max=1.01,
    skip_crowd_during_training=False,
    )
config.override(overrides, True)
config.nms_configs.max_output_size = MAX_N_INSTANCES

# Change how input preprocessing is done
if DO_ADV_PROP:
    config.override(dict(mean_rgb=0.0, stddev_rgb=1.0, scale_range=True), True)


tf.keras.backend.clear_session()

model = efficientdet_keras.EfficientDetModel(config=config)
model.build((1,*INPUT_SHAPE))

print("\n... MODEL PREDICTIONS ...\n")
preds = model.predict(np.zeros((1,*INPUT_SHAPE)))
for i, name in enumerate(["bboxes", "confidences", "classes", "valid_len", "segmentation map"]):
    print(name)
    print(preds[i].shape)
    try:
        if preds[i].shape[-2]==64:
            print(preds[i][0, 0, 0, :5])
        else:
            print(preds[i][0, :5])
        
    except:
        print(preds[i][0])
    print()

# 5.2 CREATE A DATASET WITH THE CORRECT STRUCTURE

#     INPUT
#         Raw Image (256x256x3)

#     OUTPUT/TARGET
#         Bounding Boxes
#         Instance Classes
#         Segmented Image (64x64x3)

def create_id_to_iloc_map(df):
    """
    Create mapping to allow for numeric file-names
        --> index in original df --> id
    """
    return {v:k for k,v in df.id.to_dict().items()}

TRAIN_ID_2_ILOC = create_id_to_iloc_map(train_df)
TEST_ID_2_ILOC = create_id_to_iloc_map(ss_df)


def tf_load_image(path, resize_to=INPUT_SHAPE):
    """ Load an image with the correct shape using only TF
    
    Args:
        path (tf.string): Path to the image to be loaded
        resize_to (tuple, optional): Size to reshape image
    
    Returns:
        3 channel tf.Constant image ready for training/inference
    
    """
    
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=resize_to[-1])
    img = tf.image.resize(img, resize_to[:-1])
    img = tf.cast(img, tf.uint8)
    
    return img

def load_npz(path, resize_to=SEG_SHAPE, to_binary=True):
    np_arr = np.load(path)["arr_0"]
    if to_binary:
        return np.where(cv2.resize(np_arr, resize_to[:-1])>0, 1, 0).reshape(resize_to).astype(np.uint8)
    else:
        return cv2.resize(np_arr, resize_to[:-1]).reshape(resize_to).astype(np.int32)

def image_preprocess(image, image_size, mean_rgb=config.mean_rgb, stddev_rgb=config.stddev_rgb):
    """Preprocess image for inference.
    Args:
        image: input image, can be a tensor or a numpy arary.
        image_size: single integer of image size for square image or tuple of two
            integers, in the format of (image_height, image_width).
        mean_rgb: Mean value of RGB, can be a list of float or a float value.
        stddev_rgb: Standard deviation of RGB, can be a list of float or a float
            value.
    Returns:
        (image, scale): a tuple of processed image and its scale.
  """
    input_processor = dataloader.DetectionInputProcessor(image, image_size)
    input_processor.normalize_image(mean_rgb, stddev_rgb)
    input_processor.set_scale_factors_to_output_size()
    image = input_processor.resize_and_crop_image()
    image_scale = input_processor.image_scale_to_original
    return image, image_scale


def _bytes_feature(value, is_list=False):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    
    if not is_list:
        value = [value]
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value, is_list=False):
    """Returns a float_list from a float / double."""
        
    if not is_list:
        value = [value]
        
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value, is_list=False):
    """Returns an int64_list from a bool / enum / int / uint."""
        
    if not is_list:
        value = [value]
        
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_raw(example_data):
    """
    Creates a tf.Example message ready to be written to a file from 4 features.

    Args:
        example_data: Everything from pandas row
        style (str, optional): Which subset to do... [train|val]
            [test] will be processed through a different function
    
    Returns:
        A tf.Example Message ready to be written to file
    """
    
    image_object_mask = tf.io.encode_png(load_npz(example_data["seg_path"]))
    
    image_height = INPUT_SHAPE[0]
    image_width = INPUT_SHAPE[1]
    image_source_id = image_filename = f"{TRAIN_ID_2_ILOC[example_data['id']]:>05}".encode('utf8')
    
    image_encoded = tf.io.encode_png(tf_load_image(example_data["img_path"]))
    image_key_sha256 = hashlib.sha256(image_encoded).hexdigest().encode('utf8')
    image_format = example_data["img_path"][-4:].encode('utf8') #png
    
    image_object_bbox_xmins, image_object_bbox_xmaxs  = [], []
    image_object_bbox_ymins, image_object_bbox_ymaxs  = [], []
    image_object_class_text, image_object_class_label = [], []
    image_object_is_crowd, image_object_area = [], []
    for i, box in enumerate(example_data["scaled_bboxes"]):
        if box and example_data["bbox_areas"][i]>0.0:
            image_object_bbox_xmins.append(box[0][0])
            image_object_bbox_xmaxs.append(box[1][0])
            image_object_bbox_ymins.append(box[0][1])
            image_object_bbox_ymaxs.append(box[1][1])
            image_object_class_text.append(example_data["cell_type"].encode('utf8'))
            image_object_class_label.append(ARB_SORT_MAP[example_data["cell_type"]])
            image_object_is_crowd.append(0)
            image_object_area.append(example_data["scaled_bbox_areas"][i])
    
    # Create a dictionary mapping the feature name to the 
    # tf.Example-compatible data type.
    feature_dict = {
        'image/height': _int64_feature(image_height),
        'image/width': _int64_feature(image_width),
        'image/filename': _bytes_feature(image_filename),
        'image/source_id': _bytes_feature(image_source_id),
        'image/key/sha256': _bytes_feature(image_key_sha256),
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/object/bbox/xmin': _float_feature(image_object_bbox_xmins, is_list=True),
        'image/object/bbox/xmax': _float_feature(image_object_bbox_xmaxs, is_list=True),
        'image/object/bbox/ymin': _float_feature(image_object_bbox_ymins, is_list=True),
        'image/object/bbox/ymax': _float_feature(image_object_bbox_ymaxs, is_list=True),
        'image/object/class/text': _bytes_feature(image_object_class_text, is_list=True),
        'image/object/class/label': _int64_feature(image_object_class_label, is_list=True),
        'image/object/is_crowd': _int64_feature(image_object_is_crowd, is_list=True),
        'image/object/area': _float_feature(image_object_area, is_list=True),
        'image/object/mask': _bytes_feature(image_object_mask),
    }
       
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def serialize_test_raw(example_data):
    """
    Creates a tf.Example message ready to be written to a file

    Args:
        example_data: Everything from pandas row
    
    Returns:
        A tf.Example Message ready to be written to file
    """
    
    image_height = INPUT_SHAPE[0]
    image_width = INPUT_SHAPE[1]
    image_source_id = image_filename = f"{TEST_ID_2_ILOC[example_data['id']]:>05}".encode('utf8')
    
    image_encoded = tf.io.encode_png(tf_load_image(example_data["img_path"]))
    image_key_sha256 = hashlib.sha256(image_encoded).hexdigest().encode('utf8')
    image_format = example_data["img_path"][-4:].encode('utf8') #png
    
    # Create a dictionary mapping the feature name to the 
    # tf.Example-compatible data type.
    feature_dict = {
        'image/height': _int64_feature(image_height),
        'image/width': _int64_feature(image_width),
        'image/filename': _bytes_feature(image_filename),
        'image/source_id': _bytes_feature(image_source_id),
        'image/key/sha256': _bytes_feature(image_key_sha256),
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
    }
       
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def write_tfrecords(df, n_ex, n_ex_per_rec=50, serialize_fn=serialize_raw, out_dir="/kaggle/working/tfrecords", ds_type="train"):
    """"""
    n_recs = int(np.ceil(n_ex/n_ex_per_rec))
    
    # Make dataframe iterable
    iter_df = df.iterrows()
        
    out_dir = os.path.join(out_dir, ds_type)
    # Create folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    # Create tfrecords
    for i in tqdm(range(n_recs), total=n_recs):
        print(f"\n... Writing {ds_type.title()} TFRecord {i+1} of {n_recs} ...\n")
        tfrec_path = os.path.join(out_dir, f"{ds_type}__{(i+1):02}_{n_recs:02}.tfrec")
        
        # This makes the tfrecord
        with tf.io.TFRecordWriter(tfrec_path) as writer:
            for ex in tqdm(range(n_ex_per_rec), total=n_ex_per_rec):
                try:
                    example = serialize_fn(next(iter_df)[1])
                    writer.write(example)
                except:
                    break

# TRAIN
write_tfrecords(train_df.iloc[:-N_EVAL], N_TRAIN, n_ex_per_rec=N_EX_PER_REC, serialize_fn=serialize_raw, out_dir=TFRECORD_DIR, ds_type="train")
    
# VAL
write_tfrecords(train_df[-N_EVAL:], N_EVAL, n_ex_per_rec=N_EX_PER_REC, serialize_fn=serialize_raw, out_dir=TFRECORD_DIR, ds_type="val")

# VAL
write_tfrecords(ss_df, N_TEST, n_ex_per_rec=N_EX_PER_REC, serialize_fn=serialize_test_raw, out_dir=TFRECORD_DIR, ds_type="test")

# 5.3 INSTANIATE OUR DATALOADER¶

# Augmentations are breaking the masks... so disablled for now

train_dl = dataloader.InputReader(
    file_pattern=config.train_file_pattern,
    is_training="train" in config.train_file_pattern,
    max_instances_per_image=config.max_instances_per_image
)(config.as_dict())

val_dl = dataloader.InputReader(
    file_pattern=config.val_file_pattern,
    is_training="train" in config.val_file_pattern,
    max_instances_per_image=config.max_instances_per_image
)(config.as_dict())

test_dl = dataloader.InputReader(
    file_pattern=config.test_file_pattern,
    is_training="train" in config.test_file_pattern,
    max_instances_per_image=config.max_instances_per_image
)(config.as_dict(), batch_size=1)


print("\n... TRAIN DATALOADER ...\n")
print(train_dl)

print("\n\n... VALIDATION DATALOADER ...\n")
print(val_dl)

print("\n\n... TEST DATALOADER ...\n")
print(test_dl)

print("\n\n\n\n LETS SEE AN EXAMPLE FROM OUR TRAIN DATALOADER ...\n\n")

x = next(iter(train_dl))

print(int(x[1]["source_ids"][0]))
img, msk = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[int(x[1]["source_ids"][0])].to_dict(), )
plot_img_and_mask(img, msk)

plt.figure(figsize=(20,10))

plt.subplot(1,3,1)
plt.imshow(x[0][0])
plt.axis(False)
plt.title("Cell Image", fontweight="bold")

plt.subplot(1,3,2)
plt.imshow(x[1]["image_masks"][0][0])
plt.axis(False)
plt.title("Segmentation Mask Overlay", fontweight="bold")

merged = cv2.addWeighted(np.array(x[0][0]), 0.75, np.clip(cv2.resize(np.tile(np.expand_dims(x[1]["image_masks"][0][0], axis=-1), 3), INPUT_SHAPE[:-1]), 0, 1)*255, 0.25, 0.0,)
plt.subplot(1,3,3)
plt.imshow(merged)
plt.axis(False)
plt.title("Cell Image w/ Instance Segmentation Mask Overlay", fontweight="bold")

plt.tight_layout()
plt.show()

# 5.4 CREATE MODEL AND LOAD PRETRAINED WEIGHTS
# COCO weights
if DO_TRAIN:
    if not os.path.isdir(MODEL_NAME):
        if DO_ADV_PROP:
            !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/{MODEL_NAME}.tar.gz
        else:
            !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/{MODEL_NAME}.tar.gz
        !tar -zxf {MODEL_NAME}.tar.gz
        !rm -rf {MODEL_NAME}.tar.gz
    
with strategy.scope():
    model = train_lib.EfficientDetNetTrain(config=config)
    model = setup_model(model, config)
    
    if DO_TRAIN:
        util_keras.restore_ckpt(
          model=model,
          ckpt_path_or_file=tf.train.latest_checkpoint(MODEL_NAME),
          ema_decay=config.moving_average_decay,
          exclude_layers=['class_net']
        )
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'ckpt-{epoch:d}'),
            verbose=1, save_freq="epoch", save_weights_only=True)
    else:
        model.load_weights(os.path.join(PRETRAINED_MODEL_DIR, "ckpt"))
model.summary()

# 5.5 TRAIN THE MODEL
if DO_TRAIN:
    history = model.fit(
        train_dl,
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=[ckpt_cb,],
        validation_data=val_dl,
        validation_steps=N_EVAL//BATCH_SIZE
    )
else:
    print(model.evaluate(train_dl, steps=config.steps_per_epoch))
    print(model.evaluate(val_dl, steps=N_EVAL//BATCH_SIZE))

# 5.6 VALIDATE THE MODEL IS LEARNING
def plot_gt(_image, _gt_classes, _gt_boxes, _gt_mask):
    img_class = int(_gt_classes.numpy()[0])
    img_boxes = _gt_boxes.numpy().astype(np.int32)[np.where(_gt_classes!=-1)[0]]    
    _image = _image.numpy()
    _gt_dummy_mask = np.zeros_like(_image)
    _gt_dummy_mask[..., img_class] = cv2.resize(np.expand_dims(_gt_mask, axis=-1), INPUT_SHAPE[:-1])
    _gt_mask = _gt_dummy_mask
    
    
    plt.figure(figsize=(20,7))
    
    plt.subplot(1,3,1)
    plt.imshow(_image, cmap="inferno")
    plt.axis(False)
    plt.title("Original Image After Preprocessing", fontweight="bold")
    
    mask_merged = cv2.addWeighted(_image, 0.55, _gt_mask, 1.25, 0.0)
    plt.subplot(1,3,2)
    plt.imshow(mask_merged)
    plt.axis(False)
    plt.title(f"Original Image Mask  (CLASS={img_class})", fontweight="bold")
    
    plt.subplot(1,3,3)
    box_image = np.zeros_like(_image)
    for box in img_boxes:
        ymin, xmin, ymax, xmax = box
        box_image = cv2.rectangle(img=box_image, thickness=1,  pt1=(xmin, ymin), pt2=(xmax, ymax), 
                                  color=[0 if i!=img_class else 255 for i in range(3)])
     
    box_merged = cv2.addWeighted(_image, 0.55, box_image, 1.25 if img_class==2 else 0.45, 0.0,)
    plt.imshow(box_merged)
    plt.axis(False)
    plt.title(f"Original Image Bounding Boxes  (CLASS={img_class})", fontweight="bold")

    plt.tight_layout()
    plt.show()
    
def plot_pred(_image, _pred_boxes, _pred_scores, _pred_classes, _pred_mask, conf_thresh=0.25, iou_thresh=0.0001):
    """"""
    
    if iou_thresh is not None:
        _indices, _pred_scores = tf.image.non_max_suppression_with_scores(
            _pred_boxes, _pred_scores, 800, iou_threshold=iou_thresh,
            score_threshold=conf_thresh/5, soft_nms_sigma=0.0
        )
        _pred_boxes = tf.gather(_pred_boxes, _indices)

    
    above_thresh_idx = np.where(_pred_scores.numpy()>conf_thresh)[0]
    if len(above_thresh_idx)==0:
        print("\n... NO PREDS OVER CONF THRESH... SAMPLING UP-TO FIFTY SAMPLES ...\n")
        above_thresh_idx = np.arange(min(50, len(_pred_scores)))

    _image = _image.numpy()
    _pred_class = int(np.round(_pred_classes.numpy()[above_thresh_idx].mean()))

    _pred_scores = _pred_scores.numpy()[above_thresh_idx]
    _pred_boxes = _pred_boxes.numpy().astype(np.int32)[above_thresh_idx]
    _pred_mask = np.where(_pred_mask[..., 1]>_pred_mask[..., 0], 1.0, 0.0)
    _dummy_mask = np.zeros_like(_image)
    _dummy_mask[..., _pred_class] = cv2.resize(np.expand_dims(_pred_mask, axis=-1), INPUT_SHAPE[:-1])
    _pred_mask = _dummy_mask
    
    
    plt.figure(figsize=(20,7))
    
    plt.subplot(1,3,1)
    plt.imshow(_image, cmap="inferno")
    plt.axis(False)
    plt.title("Original Image After Preprocessing", fontweight="bold")
    
    mask_merged = cv2.addWeighted(_image, 0.55, _pred_mask, 1.25, 0.0,)
    plt.subplot(1,3,2)
    plt.imshow(mask_merged)
    plt.axis(False)
    plt.title(f"Predicted Image Mask  (CLASS={_pred_class})", fontweight="bold")
    
    plt.subplot(1,3,3)
    box_image = np.zeros_like(_image)
    for box in _pred_boxes:
        ymin, xmin, ymax, xmax = box
        box_image = cv2.rectangle(img=box_image, thickness=1, pt1=(xmin, ymin), pt2=(xmax, ymax), 
                                  color=[0 if i!=_pred_class else 255 for i in range(3)])
     
    box_merged = cv2.addWeighted(_image, 0.55, box_image, 1.25 if _pred_class==2 else 0.45, 0.0,)
    plt.imshow(box_merged)
    plt.axis(False)
    plt.title(f"Predicted Image Bounding Boxes  (CLASS={_pred_class})", fontweight="bold")

    plt.tight_layout()
    plt.show()

def plot_diff(_image, _gt_classes, _gt_boxes, _gt_mask, _pred_boxes, _pred_scores, _pred_classes, _pred_mask, conf_thresh=0.25, iou_thresh=0.0001):
    """"""
    
    if iou_thresh is not None:
        _indices, _pred_scores = tf.image.non_max_suppression_with_scores(
            _pred_boxes, _pred_scores, 800, iou_threshold=iou_thresh,
            score_threshold=conf_thresh/5, soft_nms_sigma=0.0
        )
        _pred_boxes = tf.gather(_pred_boxes, _indices)
    
    _image = _image.numpy()
    
    above_thresh_idx = np.where(_pred_scores.numpy()>conf_thresh)[0]
    gt_idxs = np.where(_gt_classes!=-1)[0]
    
    if len(above_thresh_idx)==0:
        print("\n... NO PREDS OVER CONF THRESH... SAMPLING UP-TO FIFTY SAMPLES ...\n")
        above_thresh_idx = np.arange(min(50, len(_pred_scores)))
    
    _img_class = int(_gt_classes.numpy()[0])
    _pred_class = int(np.round(_pred_classes.numpy()[above_thresh_idx].mean()))
    
    img_boxes = _gt_boxes.numpy().astype(np.int32)[gt_idxs]
    _pred_boxes = _pred_boxes.numpy().astype(np.int32)[above_thresh_idx]
    
    _pred_scores = _pred_scores.numpy()[above_thresh_idx]
    
    _combo_mask = np.zeros_like(_image)
    _combo_mask[..., 0] = cv2.resize(np.expand_dims(_gt_mask, axis=-1), INPUT_SHAPE[:-1])        
    _pred_mask = np.where(_pred_mask[..., -1]>_pred_mask[..., 0], 1.0, 0.0)
    _combo_mask[..., 1] = cv2.resize(np.expand_dims(_pred_mask, axis=-1), INPUT_SHAPE[:-1])
    
    plt.figure(figsize=(20,7))
    
    plt.subplot(1,3,1)
    plt.imshow(_image, cmap="inferno")
    plt.axis(False)
    plt.title("Original Image After Preprocessing", fontweight="bold")
    
    mask_merged = cv2.addWeighted(_image, 0.55, _combo_mask, 1.25, 0.0,)
    plt.subplot(1,3,2)
    plt.imshow(mask_merged)
    plt.axis(False)
    plt.title(f"Combo Image Mask\n(RED=GT, GREEN=PRED, YELLOW=CONSENSUS)", fontweight="bold")
    
    plt.subplot(1,3,3)
    box_image = np.zeros_like(_image)
    for box in img_boxes:
        ymin, xmin, ymax, xmax = box
        box_image = cv2.rectangle(img=box_image, thickness=1, pt1=(xmin, ymin), pt2=(xmax, ymax), 
                                  color=(255,0,0))
    for box in _pred_boxes:
        ymin, xmin, ymax, xmax = box
        box_image = cv2.rectangle(img=box_image, thickness=1, pt1=(xmin, ymin), pt2=(xmax, ymax), 
                                  color=(0,255,0))
     
    box_merged = cv2.addWeighted(_image, 0.55, box_image, 1.25, 0.0)
    plt.imshow(box_merged)
    plt.axis(False)
    plt.title(f"Predicted Image Bounding Boxes\n(RED=GT, GREEN=PRED)", fontweight="bold")

    plt.tight_layout()
    plt.show()
    
    
def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    
    return iou[1:, 1:]  # exclude background


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(truths, preds, verbose=1):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


def get_pred_instance_mask(_pred_boxes, _pred_scores, _pred_mask, iou_thresh=0.0, conf_thresh=0.25):
    _indices, _pred_scores = tf.image.non_max_suppression_with_scores(
        _pred_boxes, _pred_scores, 800, iou_threshold=iou_thresh,
        score_threshold=conf_thresh/5, soft_nms_sigma=0.0
    )
    _pred_boxes = tf.gather(_pred_boxes, _indices)
    
    above_thresh_idx = np.where(_pred_scores.numpy()>conf_thresh)[0]
    if len(above_thresh_idx)==0:
        above_thresh_idx = np.arange(min(50, len(_pred_scores)))

    _pred_scores = _pred_scores.numpy()[above_thresh_idx]
    _pred_boxes = _pred_boxes.numpy().astype(np.int32)[above_thresh_idx]
    _pred_mask = cv2.resize(_pred_mask, INPUT_SHAPE[:-1], interpolation=cv2.INTER_NEAREST)
    _pred_mask = np.where(_pred_mask[..., 1]>_pred_mask[..., 0], 1.0, 0.0)
    _instance_mask = np.zeros_like(_pred_mask)
    for i, _box in enumerate(_pred_boxes):
        _instance_mask[_box[0]:_box[2], _box[1]:_box[3]] = (i+1)*_pred_mask[_box[0]:_box[2], _box[1]:_box[3]]
    _instance_mask = cv2.resize(_instance_mask, IMAGE_SHAPE[-2::-1], interpolation=cv2.INTER_NEAREST)
    return _instance_mask