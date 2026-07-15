# ==========================================================
# IMPORTS
# ==========================================================

import os
import random
from glob import glob

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import transforms
from torchvision import models

import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = r"D:\chestxrayproject\data"

CSV_PATH = os.path.join(
    DATASET_PATH,
    "Data_Entry_2017.csv"
)

IMAGE_SIZE = 320

BATCH_SIZE = 16

EPOCHS = 0

PATIENCE = 8

LEARNING_RATE = 1e-4

NO_FINDING_SAMPLES = 15000

BEST_MODEL_PATH = "best_densenet121.pth"

CHECKPOINT_PATH = "checkpoint_densenet121.pth"

HISTORY_PATH = "densenet121_history.csv"

# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(CSV_PATH)

if "Unnamed: 11" in df.columns:

    df = df.drop(
        columns=["Unnamed:11"],
        errors="ignore"
    )

print(df.shape)

# ==========================================================
# REMOVE HERNIA
# ==========================================================

df = df[
    ~df["Finding Labels"].str.contains(
        "Hernia"
    )
].reset_index(drop=True)

print()

print("After Removing Hernia")

print(df.shape)

# ==========================================================
# IMAGE PATH MAPPING
# ==========================================================

image_paths = glob(

    os.path.join(

        DATASET_PATH,

        "images_*",

        "images",

        "*.png"
    )
)

print()

print("Images Found:", len(image_paths))

image_dict = {

    os.path.basename(path): path

    for path in image_paths
}

df["Image Path"] = df[
    "Image Index"
].map(image_dict)

print(

    "Missing Paths:",

    df["Image Path"].isnull().sum()
)

df = df.dropna(

    subset=["Image Path"]

).reset_index(drop=True)

print(

    "Dataset After Removing Missing Paths:",

    df.shape
)

# ==========================================================
# MULTI LABEL ENCODING
# ==========================================================

labels = df[
    "Finding Labels"
].apply(

    lambda x: x.split("|")
)

mlb = MultiLabelBinarizer()

encoded = mlb.fit_transform(labels)

encoded_df = pd.DataFrame(

    encoded,

    columns=mlb.classes_
)

df = pd.concat(

    [

        df,

        encoded_df

    ],

    axis=1
)

disease_cols = list(

    mlb.classes_
)

print()

print("Disease Classes")

print(disease_cols)

print()

print("Total Diseases:", len(disease_cols))

# ==========================================================
# PATIENT-WISE TRAIN / VAL / TEST SPLIT
# ==========================================================

unique_patients = df[
    "Patient ID"
].unique()

train_patients, temp_patients = train_test_split(

    unique_patients,

    test_size=0.30,

    random_state=SEED
)

val_patients, test_patients = train_test_split(

    temp_patients,

    test_size=0.50,

    random_state=SEED
)

train_df = df[
    df["Patient ID"].isin(train_patients)
].copy()

val_df = df[
    df["Patient ID"].isin(val_patients)
].copy()

test_df = df[
    df["Patient ID"].isin(test_patients)
].copy()

print()

print("Patient Wise Split")

print("Train Images :", len(train_df))

print("Validation Images :", len(val_df))

print("Test Images :", len(test_df))

# ==========================================================
# UNDERSAMPLE NO FINDING
# ==========================================================

no_finding_df = train_df[

    train_df["Finding Labels"] == "No Finding"

]

disease_df = train_df[

    train_df["Finding Labels"] != "No Finding"

]

print()

print("Before Undersampling")

print("Train :", len(train_df))

print("No Finding :", len(no_finding_df))

print("Disease :", len(disease_df))

no_finding_df = no_finding_df.sample(

    n=NO_FINDING_SAMPLES,

    random_state=SEED
)

train_df = pd.concat(

    [

        no_finding_df,

        disease_df

    ],

    ignore_index=True
)

train_df = train_df.sample(

    frac=1,

    random_state=SEED

).reset_index(drop=True)

print()

print("After Undersampling")

print("Train :", len(train_df))

print()

print(

    train_df[disease_cols]

    .sum()

    .sort_values(

        ascending=False
    )
)

# ==========================================================
# IMAGE AUGMENTATION
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize(

        (

            IMAGE_SIZE,

            IMAGE_SIZE

        )

    ),

    transforms.RandomHorizontalFlip(

        p=0.5

    ),

    transforms.RandomRotation(

        degrees=10

    ),

    transforms.RandomAffine(

        degrees=0,

        translate=(0.05,0.05),

        scale=(0.95,1.05)

    ),

    transforms.ColorJitter(

        brightness=0.10,

        contrast=0.10

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[

            0.485,

            0.456,

            0.406

        ],

        std=[

            0.229,

            0.224,

            0.225

        ]
    )

])

val_transform = transforms.Compose([

    transforms.Resize(

        (

            IMAGE_SIZE,

            IMAGE_SIZE

        )

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[

            0.485,

            0.456,

            0.406

        ],

        std=[

            0.229,

            0.224,

            0.225

        ]
    )

])

# ==========================================================
# DATASET CLASS
# ==========================================================

class ChestXrayDataset(Dataset):

    def __init__(

        self,

        dataframe,

        transform=None

    ):

        self.df = dataframe.reset_index(

            drop=True

        )

        self.transform = transform

    def __len__(self):

        return len(

            self.df

        )

    def __getitem__(

        self,

        idx

    ):

        image_path = self.df.loc[

            idx,

            "Image Path"

        ]

        image = Image.open(

            image_path

        ).convert(

            "RGB"

        )

        label = self.df.loc[

            idx,

            disease_cols

        ].values.astype(

            np.float32

        )

        label = torch.tensor(

            label

        )

        if self.transform:

            image = self.transform(

                image

            )

        return image, label

# ==========================================================
# DATASETS
# ==========================================================

train_dataset = ChestXrayDataset(

    train_df,

    train_transform

)

val_dataset = ChestXrayDataset(

    val_df,

    val_transform

)

test_dataset = ChestXrayDataset(

    test_df,

    val_transform

)
# ==========================================================
# WEIGHTED RANDOM SAMPLER
# ==========================================================

sample_weights = []

for idx in range(len(train_df)):

    labels = train_df.loc[
        idx,
        disease_cols
    ].values

    weight = 1.0

    positives = np.sum(labels)

    if positives > 0:

        weight = 1.0 + positives * 2.0

    sample_weights.append(weight)

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)

# ==========================================================
# DATALOADERS
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=True
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)

print()

print("Train Loader :", len(train_loader))

print("Validation Loader :", len(val_loader))

print("Test Loader :", len(test_loader))

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print()

print("CUDA Available :", torch.cuda.is_available())

print("Device :", device)

if torch.cuda.is_available():

    print(

        "GPU :",

        torch.cuda.get_device_name(0)

    )

# ==========================================================
# DENSENET121
# ==========================================================

model = models.densenet121(

    weights=models.DenseNet121_Weights.DEFAULT

)

num_features = model.classifier.in_features

model.classifier = nn.Linear(

    num_features,

    len(disease_cols)
)

model = model.to(device)

print()

print(model.classifier)

# ==========================================================
# ASYMMETRIC LOSS
# ==========================================================

class AsymmetricLoss(nn.Module):

    def __init__(

        self,

        gamma_neg=4,

        gamma_pos=1,

        clip=0.05,

        eps=1e-8

    ):

        super().__init__()

        self.gamma_neg = gamma_neg

        self.gamma_pos = gamma_pos

        self.clip = clip

        self.eps = eps

    def forward(

        self,

        logits,

        targets

    ):

        probs = torch.sigmoid(logits)

        xs_pos = probs

        xs_neg = 1.0 - probs

        if self.clip is not None:

            xs_neg = (

                xs_neg + self.clip

            ).clamp(max=1)

        loss_pos = (

            targets

            *

            torch.log(

                xs_pos.clamp(

                    min=self.eps

                )

            )

            *

            (

                (1 - xs_pos)

                ** self.gamma_pos

            )

        )

        loss_neg = (

            (1 - targets)

            *

            torch.log(

                xs_neg.clamp(

                    min=self.eps

                )

            )

            *

            (

                probs

                ** self.gamma_neg

            )

        )

        loss = loss_pos + loss_neg

        return -loss.mean()

criterion = AsymmetricLoss()

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)

# ==========================================================
# LR SCHEDULER
# ==========================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS
)

# ==========================================================
# MIXED PRECISION
# ==========================================================

scaler = torch.amp.GradScaler("cuda")

# ==========================================================
# CHECKPOINT VARIABLES
# ==========================================================

start_epoch = 0

best_val_loss = float("inf")

history = []

patience_counter = 0

# ==========================================================
# RESUME CHECKPOINT
# ==========================================================

if os.path.exists(CHECKPOINT_PATH):

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=device
    )

    model.load_state_dict(

        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(

        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(

        checkpoint["scaler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    history = checkpoint["history"]

    print()

    print(

        f"Resuming From Epoch {start_epoch}"

    )

else:

    print()

    print("Training From Scratch")
# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(start_epoch, EPOCHS):

    print("\n" + "=" * 60)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print("=" * 60)

    # ======================================================
    # TRAIN
    # ======================================================

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(

                f"Epoch {epoch+1}/{EPOCHS} | "

                f"Batch {batch_idx}/{len(train_loader)} | "

                f"Loss: {loss.item():.4f}"

            )

    train_loss = running_loss / len(train_loader)

    # ======================================================
    # VALIDATION
    # ======================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):

                outputs = model(images)

                loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step()

    # ======================================================
    # PRINT
    # ======================================================

    print()

    print(

        f"Train Loss : {train_loss:.4f}"

    )

    print(

        f"Validation Loss : {val_loss:.4f}"

    )

    print(

        f"Learning Rate : "

        f"{optimizer.param_groups[0]['lr']:.8f}"

    )

    # ======================================================
    # SAVE HISTORY
    # ======================================================

    history.append({

        "Epoch": epoch + 1,

        "Train Loss": train_loss,

        "Validation Loss": val_loss,

        "Learning Rate":

            optimizer.param_groups[0]["lr"]

    })

    history_df = pd.DataFrame(history)

    history_df.to_csv(

        HISTORY_PATH,

        index=False

    )

    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    torch.save({

        "epoch": epoch,

        "model_state_dict":

            model.state_dict(),

        "optimizer_state_dict":

            optimizer.state_dict(),

        "scheduler_state_dict":

            scheduler.state_dict(),

        "scaler_state_dict":

            scaler.state_dict(),

        "best_val_loss":

            best_val_loss,

        "history":

            history

    },

    CHECKPOINT_PATH)

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(

            model.state_dict(),

            BEST_MODEL_PATH

        )

        print()

        print("✅ Best Model Saved")

        patience_counter = 0

    else:

        patience_counter += 1

        print()

        print(

            f"No Improvement "

            f"({patience_counter}/{PATIENCE})"

        )

    # ======================================================
    # EARLY STOPPING
    # ======================================================

    if patience_counter >= PATIENCE:

        print()

        print("=" * 60)

        print("EARLY STOPPING")

        print("=" * 60)

        break

print()

print("=" * 60)

print("TRAINING FINISHED")

print("=" * 60)
# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print("\nLoading Best Model...")

model.load_state_dict(

    torch.load(

        BEST_MODEL_PATH,

        map_location=device

    )

)

model.eval()

print("Best Model Loaded")

# ==========================================================
# PREDICT TEST SET
# ==========================================================

all_labels = []

all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            probs = torch.sigmoid(outputs)

        all_labels.append(labels.numpy())

        all_probs.append(probs.cpu().numpy())

all_labels = np.vstack(all_labels)

all_probs = np.vstack(all_probs)

print()

print("Labels Shape :", all_labels.shape)

print("Probabilities Shape :", all_probs.shape)

# ==========================================================
# SAVE PROBABILITIES
# ==========================================================

pd.DataFrame(

    all_probs,

    columns=disease_cols

).to_csv(

    "1densenet121_prediction_probabilities.csv",

    index=False

)

print("Prediction Probabilities Saved")
# ==========================================================
# PER-DISEASE THRESHOLD SEARCH
# ==========================================================

print()

print("=" * 70)
print("PER-DISEASE THRESHOLD SEARCH")
print("=" * 70)

thresholds = np.arange(

    0.05,

    0.96,

    0.01

)

best_thresholds = {}

threshold_results = []

for i, disease in enumerate(disease_cols):

    best_threshold = 0.50

    best_precision = 0.0

    best_recall = 0.0

    best_f1 = 0.0

    for threshold in thresholds:

        preds = (

            all_probs[:, i] >= threshold

        ).astype(np.int32)

        precision = precision_score(

            all_labels[:, i],

            preds,

            zero_division=0

        )

        recall = recall_score(

            all_labels[:, i],

            preds,

            zero_division=0

        )

        f1 = f1_score(

            all_labels[:, i],

            preds,

            zero_division=0

        )

        if (

            (f1 > best_f1)

            or

            (

                abs(f1 - best_f1) < 1e-6

                and

                precision > best_precision

            )

        ):

            best_threshold = threshold

            best_precision = precision

            best_recall = recall

            best_f1 = f1

    best_thresholds[disease] = best_threshold

    threshold_results.append({

        "Disease": disease,

        "Best Threshold": round(best_threshold,2),

        "Precision": round(best_precision,4),

        "Recall": round(best_recall,4),

        "F1": round(best_f1,4)

    })

threshold_df = pd.DataFrame(

    threshold_results

)

threshold_df.to_csv(

    "1densenet121_best_thresholds.csv",

    index=False

)

print()

print(threshold_df)

print()

print("Best Thresholds Saved")

# ==========================================================
# APPLY PER-DISEASE THRESHOLDS
# ==========================================================

preds = np.zeros_like(

    all_probs,

    dtype=np.int32

)

for i, disease in enumerate(disease_cols):

    preds[:, i] = (

        all_probs[:, i]

        >=

        best_thresholds[disease]

    ).astype(np.int32)

# ==========================================================
# FINAL METRICS
# ==========================================================

accuracy = accuracy_score(

    all_labels.flatten(),

    preds.flatten()

)

precision = precision_score(

    all_labels,

    preds,

    average="macro",

    zero_division=0

)

recall = recall_score(

    all_labels,

    preds,

    average="macro",

    zero_division=0

)

micro_f1 = f1_score(

    all_labels,

    preds,

    average="micro",

    zero_division=0

)

macro_f1 = f1_score(

    all_labels,

    preds,

    average="macro",

    zero_division=0

)

macro_auroc = roc_auc_score(

    all_labels,

    all_probs,

    average="macro"

)

macro_prauc = average_precision_score(

    all_labels,

    all_probs,

    average="macro"

)

final_metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Macro AUROC",

        "Macro PR-AUC",

        "Precision",

        "Recall",

        "Micro F1",

        "Macro F1"

    ],

    "Value":[

        accuracy,

        macro_auroc,

        macro_prauc,

        precision,

        recall,

        micro_f1,

        macro_f1

    ]

})

final_metrics.to_csv(

    "1densenet121_final_metrics.csv",

    index=False

)

print()

print("="*70)

print("FINAL RESULTS")

print("="*70)

print(final_metrics)

# ==========================================================
# PER-DISEASE METRICS
# ==========================================================

results = []

for i, disease in enumerate(disease_cols):

    results.append({

        "Disease": disease,

        "Threshold": round(

            best_thresholds[disease],

            2

        ),

        "AUROC": roc_auc_score(

            all_labels[:, i],

            all_probs[:, i]

        ),

        "PR_AUC": average_precision_score(

            all_labels[:, i],

            all_probs[:, i]

        ),

        "Precision": precision_score(

            all_labels[:, i],

            preds[:, i],

            zero_division=0

        ),

        "Recall": recall_score(

            all_labels[:, i],

            preds[:, i],

            zero_division=0

        ),

        "F1": f1_score(

            all_labels[:, i],

            preds[:, i],

            zero_division=0

        )

    })

per_disease_df = pd.DataFrame(

    results

)

per_disease_df.to_csv(

    "1densenet121_per_disease_metrics.csv",

    index=False

)

print()

print("="*70)

print("PER-DISEASE METRICS")

print("="*70)

print(per_disease_df)

print()

print("All CSV files saved successfully.")

# ==========================================================
# TRAINING CURVE
# ==========================================================

history_df = pd.DataFrame(history)

plt.figure(figsize=(8,5))

plt.plot(

    history_df["Epoch"],

    history_df["Train Loss"],

    label="Train"

)

plt.plot(

    history_df["Epoch"],

    history_df["Validation Loss"],

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("DenseNet121 Loss")

plt.grid(True)

plt.legend()

plt.savefig(

    "1densenet121_training_curve.png",

    dpi=300

)

plt.close()

print()

print("Training Curve Saved")

print()

print("ALL FILES SAVED SUCCESSFULLY")    