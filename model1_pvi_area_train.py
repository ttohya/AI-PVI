"""
AI Model 1: PVI Area Segmentation Training
7-level U-Net with ConvMixer blocks
Two-phase curriculum learning: all data -> recurrence-free cases
"""
import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import time

# Configuration
FILTERS = [16, 32, 48, 72, 104, 144, 192]
BG_RGB = np.array([0.239, 0.231, 0.184], dtype=np.float32)
IMAGE_SIZE = 512
BATCH_SIZE = 4

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def detect_atrium_mask(imgs):
    """Detect atrium region (non-background pixels)"""
    diff = np.abs(imgs - BG_RGB)
    return (diff.max(axis=-1, keepdims=True) > 0.1).astype(np.float32)

def load_data(csv_path, input_dir, mask_dir):
    """Load and split data based on AF recurrence and k-fold"""
    test_subjids = set()
    target_train = {'imgs': [], 'masks': []}
    test_data = {'imgs': [], 'masks': []}
    all_data = {'imgs': [], 'masks': []}

    # Load from CSV (labeled data with outcome)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            af_rec = float(row['AF_rec'])
            khold = int(row['khold'])

            img = np.array(Image.open(row['input_img']).resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            mask = np.array(Image.open(row['mask1_img']).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            mask = mask[:, :, np.newaxis]

            if af_rec == 0.0 and khold == 4:
                test_subjids.add(row['subjid'])
                test_data['imgs'].append(img)
                test_data['masks'].append(mask)
            elif af_rec == 0.0:
                target_train['imgs'].append(img)
                target_train['masks'].append(mask)

    # Load additional data (all masks, excluding test subjects)
    processed = set()
    for mask_file in os.listdir(mask_dir):
        if not mask_file.startswith('m1_'):
            continue
        parts = mask_file.replace('m1_', '').replace('.jpg', '').split('_')
        subjid = parts[0] + '_' + parts[1]

        if subjid in test_subjids:
            continue

        base_name = mask_file.replace('m1_', '')
        if base_name in processed:
            continue
        processed.add(base_name)

        input_path = os.path.join(input_dir, base_name)
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(input_path):
            img = np.array(Image.open(input_path).resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            mask = np.array(Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            all_data['imgs'].append(img)
            all_data['masks'].append(mask[:, :, np.newaxis])

    return all_data, target_train, test_data

def augment(img, atrium, target):
    """Data augmentation: flip and rotation"""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        atrium = tf.image.flip_left_right(atrium)
        target = tf.image.flip_left_right(target)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        atrium = tf.image.flip_up_down(atrium)
        target = tf.image.flip_up_down(target)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    atrium = tf.image.rot90(atrium, k)
    target = tf.image.rot90(target, k)
    return img, atrium, target

def convmixer_block(x, n_filters, kernel=7):
    """ConvMixer block: depthwise conv + pointwise conv"""
    dw = tf.keras.layers.DepthwiseConv2D(kernel, padding='same')(x)
    dw = tf.keras.layers.BatchNormalization()(dw)
    dw = tf.keras.layers.ReLU()(dw)
    x = tf.keras.layers.Add()([x, dw])
    x = tf.keras.layers.Conv2D(n_filters, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def conv_block(x, n_filters):
    """Double ConvMixer block with residual connection"""
    shortcut = tf.keras.layers.Conv2D(n_filters, 1, padding="same")(x)
    x = convmixer_block(x, n_filters)
    x = convmixer_block(x, n_filters)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def attention_gate(x, g, n_filters):
    """Attention gate for skip connections"""
    theta_x = tf.keras.layers.Conv2D(n_filters, 1, padding="same")(x)
    phi_g = tf.keras.layers.Conv2D(n_filters, 1, padding="same")(g)
    add = tf.keras.layers.Add()([theta_x, phi_g])
    act = tf.keras.layers.ReLU()(add)
    psi = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(act)
    return tf.keras.layers.Multiply()([x, psi])

def downsample(x, n_filters):
    f = conv_block(x, n_filters)
    p = tf.keras.layers.Conv2D(n_filters, 3, strides=2, padding="same")(f)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.ReLU()(p)
    p = tf.keras.layers.Dropout(0.1)(p)
    return f, p

def upsample(x, skip, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    skip = attention_gate(skip, x, max(n_filters // 2, 4))
    x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Dropout(0.1)(x)
    x = conv_block(x, n_filters)
    return x

def build_model():
    """Build 7-level U-Net with ConvMixer blocks"""
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    f1, p1 = downsample(inputs, FILTERS[0])
    f2, p2 = downsample(p1, FILTERS[1])
    f3, p3 = downsample(p2, FILTERS[2])
    f4, p4 = downsample(p3, FILTERS[3])
    f5, p5 = downsample(p4, FILTERS[4])
    f6, p6 = downsample(p5, FILTERS[5])
    bottleneck = conv_block(p6, FILTERS[6])
    u1 = upsample(bottleneck, f6, FILTERS[5])
    u2 = upsample(u1, f5, FILTERS[4])
    u3 = upsample(u2, f4, FILTERS[3])
    u4 = upsample(u3, f3, FILTERS[2])
    u5 = upsample(u4, f2, FILTERS[1])
    u6 = upsample(u5, f1, FILTERS[0])
    outputs = tf.keras.layers.Conv2D(2, 1, activation="softmax")(u6)
    return tf.keras.Model(inputs, outputs)

@tf.function
def masked_ce(y_true, y_pred, atrium_mask):
    """Masked cross-entropy loss (atrium region only)"""
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mask_sq = tf.squeeze(atrium_mask, axis=-1)
    return tf.reduce_sum(ce * mask_sq) / (tf.reduce_sum(mask_sq) + 1e-6)

@tf.function
def soft_iou_loss(y_true, y_pred, atrium_mask):
    """Soft IoU loss for segmentation"""
    y_true_fg = y_true[..., 1]
    y_pred_fg = y_pred[..., 1]
    mask = atrium_mask[..., 0]
    y_true_m = y_true_fg * mask
    y_pred_m = y_pred_fg * mask
    intersection = tf.reduce_sum(y_true_m * y_pred_m)
    union = tf.reduce_sum(y_true_m) + tf.reduce_sum(y_pred_m) - intersection
    return 1.0 - (intersection + 1e-6) / (union + 1e-6)

def calc_iou(y_true, y_pred, atrium_mask):
    """Calculate IoU within atrium region"""
    atrium_flat = atrium_mask[:,:,:,0].flatten() > 0.5
    y_true_m = y_true.flatten()[atrium_flat]
    y_pred_m = y_pred.flatten()[atrium_flat]
    inter = np.sum((y_true_m == 1) & (y_pred_m == 1))
    union = np.sum((y_true_m == 1) | (y_pred_m == 1))
    return inter / (union + 1e-6)

def main(args):
    setup_gpu()
    tf.keras.utils.set_random_seed(args.seed)

    print(f"=== PVI Area Model {args.seed} ===")

    # Load data
    all_data, target_train, test_data = load_data(
        args.csv_path, args.input_dir, args.mask_dir
    )

    all_imgs = np.array(all_data['imgs'], dtype=np.float32)
    all_masks = np.array(all_data['masks'], dtype=np.float32)
    all_atrium = detect_atrium_mask(all_imgs)
    all_binary = (all_masks > 0.5).astype(np.float32)
    all_target = np.concatenate([1 - all_binary, all_binary], axis=-1)

    target_imgs = np.array(target_train['imgs'], dtype=np.float32)
    target_masks = np.array(target_train['masks'], dtype=np.float32)
    target_atrium = detect_atrium_mask(target_imgs)
    target_binary = (target_masks > 0.5).astype(np.float32)
    target_target = np.concatenate([1 - target_binary, target_binary], axis=-1)

    test_imgs = np.array(test_data['imgs'], dtype=np.float32)
    test_masks = np.array(test_data['masks'], dtype=np.float32)
    test_atrium = detect_atrium_mask(test_imgs)
    test_binary = (test_masks > 0.5).astype(np.float32)

    print(f"Phase 1: {len(all_imgs)}, Phase 2: {len(target_imgs)}, Test: {len(test_imgs)}")

    # Build model
    model = build_model()
    print(f"Parameters: {model.count_params():,}")

    # Phase 1: Pre-training
    print(f"\n[Model {args.seed}] Phase 1: Pre-training on all data")
    all_ds = tf.data.Dataset.from_tensor_slices((all_imgs, all_atrium, all_target))
    all_ds = all_ds.cache().shuffle(len(all_imgs)).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(imgs, atrium_masks, targets):
        with tf.GradientTape() as tape:
            pred = model(imgs, training=True)
            ce = masked_ce(targets, pred, atrium_masks)
            iou_loss = soft_iou_loss(targets, pred, atrium_masks)
            loss = ce + iou_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return ce, iou_loss

    for epoch in range(30):
        start = time.time()
        for imgs, atrium_masks, targets in all_ds:
            train_step(imgs, atrium_masks, targets)
        val_pred = model.predict(test_imgs, verbose=0, batch_size=2)
        y_pred = np.argmax(val_pred, axis=-1)
        y_true = (test_binary[:,:,:,0] > 0.5).astype(np.int32)
        iou = calc_iou(y_true, y_pred, test_atrium)
        print(f"[M{args.seed}-P1] Epoch {epoch+1:2d}/30 IoU:{iou:.4f} [{(time.time()-start)/60:.1f}min]")

    # Phase 2: Fine-tuning
    print(f"\n[Model {args.seed}] Phase 2: Fine-tuning on recurrence-free cases")
    target_ds = tf.data.Dataset.from_tensor_slices((target_imgs, target_atrium, target_target))
    target_ds = target_ds.cache().shuffle(len(target_imgs)).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    best_iou = 0.0
    patience = 0
    for epoch in range(40):
        start = time.time()
        for imgs, atrium_masks, targets in target_ds:
            train_step(imgs, atrium_masks, targets)
        val_pred = model.predict(test_imgs, verbose=0, batch_size=2)
        y_pred = np.argmax(val_pred, axis=-1)
        y_true = (test_binary[:,:,:,0] > 0.5).astype(np.int32)
        iou = calc_iou(y_true, y_pred, test_atrium)
        print(f"[M{args.seed}-P2] Epoch {epoch+1:2d}/40 IoU:{iou:.4f} (best:{best_iou:.4f}) [{(time.time()-start)/60:.1f}min]")

        if iou > best_iou:
            best_iou = iou
            patience = 0
            model.save_weights(os.path.join(args.output_dir, f"pvi_area_ens_{args.seed}.weights.h5"))
        else:
            patience += 1

        if patience >= 15:
            break

    print(f"\n*** Model {args.seed} Best IoU: {best_iou:.4f} ***")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PVI Area Model")
    parser.add_argument("seed", type=int, nargs="?", default=1, help="Random seed (1-5)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV with labeled data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with mask images")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory for weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
