# XL-Mix PV Line Prediction (within Atrium area)
# Input: Original image + PVI GT mask overlay (semi-transparent)
# Loss: Calculated within Atrium area (not just PVI)
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import time

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
FILTERS = [16, 32, 48, 72, 104, 144, 192]
BG_RGB = np.array([0.239, 0.231, 0.184], dtype=np.float32)
print(f"=== PV Line Model {seed} (Atrium loss) ===")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 4

def overlay_pvi_mask(img, pvi_mask, alpha=0.4):
    """Overlay PVI GT mask on input image (green semi-transparent)"""
    result = img.copy()
    mask = pvi_mask[:, :, 0] > 0.5
    result[mask, 0] = img[mask, 0] * (1 - alpha)
    result[mask, 1] = img[mask, 1] * (1 - alpha) + alpha
    result[mask, 2] = img[mask, 2] * (1 - alpha)
    return result

def detect_atrium_mask(img):
    """Detect atrium region (non-background pixels)"""
    diff = np.abs(img - BG_RGB)
    return (diff.max(axis=-1, keepdims=True) > 0.1).astype(np.float32)

def load_data():
    csv_path = "/mnt/s/Workfolder/Ablation/PV_area/csv/Analysis_df_3dir.csv"
    test_subjids = set()
    target_train_data = {'imgs': [], 'atrium_masks': [], 'pvi_masks': [], 'line_masks': []}
    test_data = {'imgs': [], 'atrium_masks': [], 'pvi_masks': [], 'line_masks': []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            af_rec = float(row['AF_rec'])
            khold = int(row['khold'])

            if af_rec == 0.0 and khold == 4:
                test_subjids.add(row['subjid'])
                img = np.array(Image.open(row['input_img']).resize((512, 512))) / 255.0
                atrium_mask = detect_atrium_mask(img)
                pvi_mask = np.array(Image.open(row['mask1_img']).convert('L').resize((512, 512))) / 255.0
                pvi_mask = pvi_mask[:, :, np.newaxis]
                line_mask = np.array(Image.open(row['mask2_img']).convert('L').resize((512, 512))) / 255.0
                img_with_pvi = overlay_pvi_mask(img, pvi_mask)
                test_data['imgs'].append(img_with_pvi)
                test_data['atrium_masks'].append(atrium_mask)
                test_data['pvi_masks'].append(pvi_mask)
                test_data['line_masks'].append(line_mask[:, :, np.newaxis])
            elif af_rec == 0.0:
                img = np.array(Image.open(row['input_img']).resize((512, 512))) / 255.0
                atrium_mask = detect_atrium_mask(img)
                pvi_mask = np.array(Image.open(row['mask1_img']).convert('L').resize((512, 512))) / 255.0
                pvi_mask = pvi_mask[:, :, np.newaxis]
                line_mask = np.array(Image.open(row['mask2_img']).convert('L').resize((512, 512))) / 255.0
                img_with_pvi = overlay_pvi_mask(img, pvi_mask)
                target_train_data['imgs'].append(img_with_pvi)
                target_train_data['atrium_masks'].append(atrium_mask)
                target_train_data['pvi_masks'].append(pvi_mask)
                target_train_data['line_masks'].append(line_mask[:, :, np.newaxis])

    # Load all data (excluding test IDs)
    mask_dir = "/mnt/s/Workfolder/Ablation/PV_area/img/mask_img/"
    input_dir = "/mnt/s/Workfolder/Ablation/PV_area/img/input_img/"

    all_data = {'imgs': [], 'atrium_masks': [], 'pvi_masks': [], 'line_masks': []}
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
        pvi_path = os.path.join(mask_dir, mask_file)
        line_path = os.path.join(mask_dir, 'm2_' + base_name)

        if os.path.exists(input_path) and os.path.exists(line_path):
            img = np.array(Image.open(input_path).resize((512, 512))) / 255.0
            atrium_mask = detect_atrium_mask(img)
            pvi_mask = np.array(Image.open(pvi_path).convert('L').resize((512, 512))) / 255.0
            pvi_mask = pvi_mask[:, :, np.newaxis]
            line_mask = np.array(Image.open(line_path).convert('L').resize((512, 512))) / 255.0
            img_with_pvi = overlay_pvi_mask(img, pvi_mask)
            all_data['imgs'].append(img_with_pvi)
            all_data['atrium_masks'].append(atrium_mask)
            all_data['pvi_masks'].append(pvi_mask)
            all_data['line_masks'].append(line_mask[:, :, np.newaxis])

    return all_data, target_train_data, test_data, test_subjids

all_data, target_train_data, test_data, test_subjids = load_data()

all_imgs = np.array(all_data['imgs'], dtype=np.float32)
all_atrium = np.array(all_data['atrium_masks'], dtype=np.float32)
all_pvi = np.array(all_data['pvi_masks'], dtype=np.float32)
all_line = np.array(all_data['line_masks'], dtype=np.float32)

target_imgs = np.array(target_train_data['imgs'], dtype=np.float32)
target_atrium = np.array(target_train_data['atrium_masks'], dtype=np.float32)
target_pvi = np.array(target_train_data['pvi_masks'], dtype=np.float32)
target_line = np.array(target_train_data['line_masks'], dtype=np.float32)

test_imgs = np.array(test_data['imgs'], dtype=np.float32)
test_atrium = np.array(test_data['atrium_masks'], dtype=np.float32)
test_pvi = np.array(test_data['pvi_masks'], dtype=np.float32)
test_line = np.array(test_data['line_masks'], dtype=np.float32)

# Binary masks
all_line_bin = (all_line > 0.5).astype(np.float32)
target_line_bin = (target_line > 0.5).astype(np.float32)
test_pvi_bin = (test_pvi > 0.5).astype(np.float32)
test_line_bin = (test_line > 0.5).astype(np.float32)

# Targets (2-channel)
all_target = np.concatenate([1 - all_line_bin, all_line_bin], axis=-1)
target_target = np.concatenate([1 - target_line_bin, target_line_bin], axis=-1)

print(f"Phase 1: {len(all_imgs)}, Phase 2: {len(target_imgs)}, Test: {len(test_imgs)}")

def augment(img, atrium_mask, target):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        atrium_mask = tf.image.flip_left_right(atrium_mask)
        target = tf.image.flip_left_right(target)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        atrium_mask = tf.image.flip_up_down(atrium_mask)
        target = tf.image.flip_up_down(target)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    atrium_mask = tf.image.rot90(atrium_mask, k)
    target = tf.image.rot90(target, k)
    return img, atrium_mask, target

def convmixer_block(x, n_filters, kernel=7):
    dw = tf.keras.layers.DepthwiseConv2D(kernel, padding='same')(x)
    dw = tf.keras.layers.BatchNormalization()(dw)
    dw = tf.keras.layers.ReLU()(dw)
    x = tf.keras.layers.Add()([x, dw])
    x = tf.keras.layers.Conv2D(n_filters, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def conv_block(x, n_filters):
    shortcut = tf.keras.layers.Conv2D(n_filters, 1, padding="same")(x)
    x = convmixer_block(x, n_filters)
    x = convmixer_block(x, n_filters)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def attention_gate(x, g, n_filters):
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
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
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
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mask_sq = tf.squeeze(atrium_mask, axis=-1)
    return tf.reduce_sum(ce * mask_sq) / (tf.reduce_sum(mask_sq) + 1e-6)

@tf.function
def soft_iou_loss(y_true, y_pred, atrium_mask):
    y_true_fg = y_true[..., 1]
    y_pred_fg = y_pred[..., 1]
    mask = atrium_mask[..., 0]
    y_true_m = y_true_fg * mask
    y_pred_m = y_pred_fg * mask
    intersection = tf.reduce_sum(y_true_m * y_pred_m)
    union = tf.reduce_sum(y_true_m) + tf.reduce_sum(y_pred_m) - intersection
    return 1.0 - (intersection + 1e-6) / (union + 1e-6)

def calc_iou(y_true, y_pred, pvi_mask):
    """Evaluate IoU within PVI area (for fair comparison)"""
    pvi_flat = pvi_mask[:,:,:,0].flatten() > 0.5
    y_true_m = y_true.flatten()[pvi_flat]
    y_pred_m = y_pred.flatten()[pvi_flat]
    inter = np.sum((y_true_m == 1) & (y_pred_m == 1))
    union = np.sum((y_true_m == 1) | (y_pred_m == 1))
    return inter / (union + 1e-6)

tf.keras.utils.set_random_seed(seed)
model = build_model()
print(f"Parameters: {model.count_params():,}")

# Phase 1
print(f"\n[Model {seed}] Phase 1: Pre-training on all data (Atrium loss)")
all_ds = tf.data.Dataset.from_tensor_slices((all_imgs, all_atrium, all_target))
all_ds = all_ds.cache().shuffle(len(all_imgs)).map(augment).batch(batch_size).prefetch(tf.data.AUTOTUNE)
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

best_iou_p1 = 0.0
for epoch in range(30):
    start = time.time()
    for imgs, atrium_masks, targets in all_ds:
        train_step(imgs, atrium_masks, targets)
    val_pred = model.predict(test_imgs, verbose=0, batch_size=2)
    y_pred = np.argmax(val_pred, axis=-1)
    y_true = (test_line_bin[:,:,:,0] > 0.5).astype(np.int32)
    iou = calc_iou(y_true, y_pred, test_pvi_bin)
    if iou > best_iou_p1:
        best_iou_p1 = iou
    print(f"[M{seed}-P1] Epoch {epoch+1:2d}/30 IoU:{iou:.4f} (best:{best_iou_p1:.4f}) [{(time.time()-start)/60:.1f}min]")

# Phase 2
print(f"\n[Model {seed}] Phase 2: Fine-tuning on AF_rec=0 (Atrium loss)")
target_ds = tf.data.Dataset.from_tensor_slices((target_imgs, target_atrium, target_target))
target_ds = target_ds.cache().shuffle(len(target_imgs)).map(augment).batch(batch_size).prefetch(tf.data.AUTOTUNE)
optimizer = tf.keras.optimizers.Adam(1e-4)

best_iou_p2 = 0.0
patience = 0
for epoch in range(40):
    start = time.time()
    for imgs, atrium_masks, targets in target_ds:
        train_step(imgs, atrium_masks, targets)
    val_pred = model.predict(test_imgs, verbose=0, batch_size=2)
    y_pred = np.argmax(val_pred, axis=-1)
    y_true = (test_line_bin[:,:,:,0] > 0.5).astype(np.int32)
    iou = calc_iou(y_true, y_pred, test_pvi_bin)
    print(f"[M{seed}-P2] Epoch {epoch+1:2d}/40 IoU:{iou:.4f} (best:{best_iou_p2:.4f}) [{(time.time()-start)/60:.1f}min]")
    if iou > best_iou_p2:
        best_iou_p2 = iou
        patience = 0
        model.save_weights(f"/mnt/s/Workfolder/Ablation/PV_area/model/pvline_ens_{seed}.weights.h5")
    else:
        patience += 1
    if patience >= 15:
        break

print(f"\n*** Model {seed} Best IoU: {best_iou_p2:.4f} ***")
