# PV Line 5-Ensemble + TTA Evaluation
# Input: Original image + PVI GT mask overlay (semi-transparent)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import csv
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

FILTERS = [16, 32, 48, 72, 104, 144, 192]

def overlay_pvi_mask(img, pvi_mask, alpha=0.4):
    """Overlay PVI GT mask on input image (green semi-transparent)"""
    result = img.copy()
    mask = pvi_mask[:, :, 0] > 0.5
    result[mask, 0] = img[mask, 0] * (1 - alpha)
    result[mask, 1] = img[mask, 1] * (1 - alpha) + alpha
    result[mask, 2] = img[mask, 2] * (1 - alpha)
    return result

print("Loading test data...")
csv_path = "/mnt/s/Workfolder/Ablation/PV_area/csv/Analysis_df_3dir.csv"
test_imgs_list, test_pvi_list, test_line_list = [], [], []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if float(row['AF_rec']) == 0.0 and int(row['khold']) == 4:
            img = np.array(Image.open(row['input_img']).resize((512, 512))) / 255.0
            pvi_mask = np.array(Image.open(row['mask1_img']).convert('L').resize((512, 512))) / 255.0
            pvi_mask = pvi_mask[:, :, np.newaxis]
            line_mask = np.array(Image.open(row['mask2_img']).convert('L').resize((512, 512))) / 255.0
            img_with_pvi = overlay_pvi_mask(img, pvi_mask)
            test_imgs_list.append(img_with_pvi)
            test_pvi_list.append(pvi_mask)
            test_line_list.append(line_mask[:, :, np.newaxis])

test_imgs = np.array(test_imgs_list, dtype=np.float32)
test_pvi = np.array(test_pvi_list, dtype=np.float32)
test_line = np.array(test_line_list, dtype=np.float32)
test_pvi_bin = (test_pvi > 0.5).astype(np.float32)
test_line_bin = (test_line > 0.5).astype(np.float32)
print(f"Test: {len(test_imgs)}")

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

def calc_iou(y_true, y_pred, pvi_mask):
    pvi_flat = pvi_mask[:,:,:,0].flatten() > 0.5
    y_true_m = y_true.flatten()[pvi_flat]
    y_pred_m = y_pred.flatten()[pvi_flat]
    inter = np.sum((y_true_m == 1) & (y_pred_m == 1))
    union = np.sum((y_true_m == 1) | (y_pred_m == 1))
    return inter / (union + 1e-6)

def predict_tta(model, imgs):
    preds = [model.predict(imgs, verbose=0, batch_size=2)]
    preds.append(np.flip(model.predict(np.flip(imgs, axis=2), verbose=0, batch_size=2), axis=2))
    preds.append(np.flip(model.predict(np.flip(imgs, axis=1), verbose=0, batch_size=2), axis=1))
    for k in [1, 2, 3]:
        r = np.rot90(imgs, k=k, axes=(1, 2))
        p = model.predict(r, verbose=0, batch_size=2)
        preds.append(np.rot90(p, k=-k, axes=(1, 2)))
    return np.mean(preds, axis=0)

y_true = (test_line_bin[:,:,:,0] > 0.5).astype(np.int32)

print("\n=== Single Models ===")
all_preds, all_tta_preds = [], []
for seed in [1, 2, 3, 4, 5]:
    tf.keras.backend.clear_session()
    model = build_model()
    path = f"/mnt/s/Workfolder/Ablation/PV_area/model/pvline_ens_{seed}.weights.h5"
    if not os.path.exists(path):
        print(f"Model {seed}: not found")
        continue
    model.load_weights(path)

    pred = model.predict(test_imgs, verbose=0, batch_size=2)
    iou = calc_iou(y_true, np.argmax(pred, axis=-1), test_pvi_bin)
    print(f"Model {seed}: IoU = {iou:.4f}")
    all_preds.append(pred)

    pred_tta = predict_tta(model, test_imgs)
    iou_tta = calc_iou(y_true, np.argmax(pred_tta, axis=-1), test_pvi_bin)
    print(f"Model {seed} + TTA: IoU = {iou_tta:.4f}")
    all_tta_preds.append(pred_tta)

if len(all_preds) >= 2:
    print("\n=== Ensemble ===")
    ens_pred = np.mean(all_preds, axis=0)
    iou = calc_iou(y_true, np.argmax(ens_pred, axis=-1), test_pvi_bin)
    print(f"Ensemble ({len(all_preds)} models): IoU = {iou:.4f}")

if len(all_tta_preds) >= 2:
    print("\n=== Ensemble + TTA ===")
    ens_tta = np.mean(all_tta_preds, axis=0)
    iou = calc_iou(y_true, np.argmax(ens_tta, axis=-1), test_pvi_bin)
    print(f"Ensemble + TTA ({len(all_tta_preds)} models): IoU = {iou:.4f}")
