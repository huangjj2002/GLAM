import torch
import time
import warnings
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
from collections import Counter
from glob import glob
import torchvision.transforms as transforms
import random
import pydicom as dicom
from .transforms import OtsuCut
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from .constants_val import *
from .utils import get_imgs, read_from_dicom, get_tokenizer


class VinDr(torch.utils.data.Dataset):

    def __init__(
        self,
        split="train",
        transform=None,
        imsize=1024,
        data_pct=1.0,
        llm_type="gpt",
        pred_density=False,
        pred_mass=False,
        pred_calc=False,
        uniform_norm=False,
        max_words=64,
        structural_cap=False,
        natural_cap=False,
        simple_cap=False,
        raw_caption=False,
        load_jpg=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.df = pd.read_csv(VINDR_CSV_DIR)
        self.data_path = VINDR_IMAGE_DIR
        self.transform = transform
        self.imsize = imsize
        self.uniform_norm = uniform_norm
        self.pred_density = pred_density
        self.pred_mass = pred_mass
        self.pred_calc = pred_calc
        self.llm_type = llm_type
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.natural_cap = natural_cap
        self.simple_cap = simple_cap
        self.raw_caption = raw_caption
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.load_jpg = load_jpg
        self.n_classes = 5 if not self.pred_density else 4
        if split == "test":
            self.df = self.df[self.df["split"] == "test"]
        else:
            self.df = self.df[self.df["split"] == "training"]
        self.findings_df = pd.read_csv(VINDR_DET_CSV_DIR)

        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=data_pct)

        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.labels = []
        self.path2label = {}
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            if self.pred_density:
                label = entry["breast_density"].split(" ")[-1]
                label = VINDR_DENSITY_LETTER2DIGIT[label]
            elif self.pred_mass or self.pred_calc:
                image_id = entry["image_id"]
                findings = self.findings_df[self.findings_df["image_id"] == image_id][
                    "finding_categories"
                ]
                findings_list = findings.to_list()
                findings_str = " ".join(findings_list)
                if self.pred_mass:
                    label = 2 if "Mass" in findings_str else 1
                else:
                    label = 2 if "Suspicious Calcification" in findings_str else 1
            else:
                # BIRADS 1 ~ 5
                label = int(entry["breast_birads"].split(" ")[-1])
            sid = entry["study_id"]
            imid = entry["image_id"]
            dicom_path = os.path.join(self.data_path, f"{sid}/{imid}.dicom")
            self.filenames.append(dicom_path)
            self.labels.append(label - 1)
            self.path2label[dicom_path] = label - 1
        print("### Sampled split distribution: ", Counter(self.labels))

    def __len__(self):
        return len(self.df)

    def get_caption(self, series_sents):
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        tokens["masked_ids"] = tokens["input_ids"]

        return tokens, x_len

    def get_zeroshot_caption(self):
        base_captions = ""
        zero_shot_caps = []
        zero_shot_caps_len = []
        if self.pred_density:
            for density, density_desc in EMBED_DENSITY_DESC.items():
                if density == 5:
                    continue
                if self.structural_cap:
                    density_desc = EMBED_DENSITY_DESC[density]
                    captions = (
                        base_captions
                        + EMBED_DENSITY
                        + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                            "{{DENSITY}}", density_desc
                        )
                    )
                    if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                        captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                elif self.natural_cap:
                    density_desc = EMBED_DENSITY_DESC[density]
                    captions = base_captions + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                        "{{DENSITY}}", density_desc
                    )
                    if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                        captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                else:
                    captions = (
                        base_captions
                        + BREAST_BASE_CAPTION
                        + BREAST_DENSITY_CAPTION
                        + str(density)
                        + ": "
                        + density_desc
                        + "."
                    )
                # Update caption type if using raw style caption
                if self.raw_caption:
                    captions = captions.replace(".", " " + self.tokenizer.sep_token)
                    captions = captions.replace(";", " " + self.tokenizer.sep_token)
                    if self.llm_type != "bert":
                        captions = (
                            self.tokenizer.bos_token
                            + " "
                            + captions
                            + " "
                            + self.tokenizer.eos_token
                        )
                else:
                    captions = captions.replace("\n", " ").lower()
                cap, cap_len = self.get_caption([captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)
        else:
            for digits in range(0, 5):
                asses = VINDR_BIRADS_DIGIT2LETTER[digits]
                birads_desc = EMBED_BIRADS_DESC[asses]
                # VinDr only consider BIRADS 1 ~ 5
                if asses in ["A", "K"]:
                    continue
                birads = EMBED_LETTER_TO_BIRADS[asses]
                # build density caption following training format
                if self.structural_cap:
                    # findings
                    mass_info = EMBED_MASS_CAPTION[asses]
                    captions = (
                        base_captions
                        + EMBED_FINDINGS
                        + EMBED_FINDS_CAPTION
                        + mass_info
                        + " "
                    )
                    # impression
                    impression_desc = EMBED_BIRADS_DESC[asses]
                    captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                        "{{BIRADS}}", str(birads)
                    ).replace("{{BIRADS_DESC}}", impression_desc)
                    # overall assesment
                    captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                elif self.natural_cap:
                    # findings
                    mass_info = EMBED_MASS_CAPTION[asses]
                    captions = base_captions + EMBED_FINDS_CAPTION + mass_info + " "
                    # impression
                    impression_desc = EMBED_BIRADS_DESC[asses]
                    captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                        "{{BIRADS}}", str(birads)
                    ).replace("{{BIRADS_DESC}}", impression_desc)
                else:
                    captions = (
                        base_captions
                        + BREAST_BASE_CAPTION
                        + BREAST_BIRADS_CAPTION
                        + str(birads)
                        + ": "
                        + birads_desc
                        + "."
                    )
                # Update caption type if using raw style caption
                if self.raw_caption:
                    captions = captions.replace(".", " " + self.tokenizer.sep_token)
                    captions = captions.replace(";", " " + self.tokenizer.sep_token)
                    if self.llm_type != "bert":
                        captions = (
                            self.tokenizer.bos_token
                            + " "
                            + captions
                            + " "
                            + self.tokenizer.eos_token
                        )
                else:
                    captions = captions.replace("\n", " ").lower()
                cap, cap_len = self.get_caption([captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)

        stacked_caps = {}
        for cap in zero_shot_caps:
            for k, v in cap.items():
                if k not in stacked_caps:
                    stacked_caps[k] = v
                else:
                    stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
        zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        self.zero_shot_caps = stacked_caps
        self.zero_shot_caps_len = zero_shot_caps_len

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        sid = entry["study_id"]
        imid = entry["image_id"]
        view = entry["laterality"] + entry["view_position"]
        label = self.labels[idx]
        dicom_path = os.path.join(self.data_path, f"{sid}/{imid}.dicom")

        if self.load_jpg:
            img_path = dicom_path.replace("vindr-1.0.0", "vindr-1.0.0-resized-1024")
            img_path = img_path.replace(".dicom", "_resized.png")
            assert os.path.exists(img_path)
            img = get_imgs(img_path, scale=self.imsize, transform=self.transform)
        else:
            assert os.path.exists(dicom_path)
            img = read_from_dicom(dicom_path, transform=self.transform)
        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        if self.zero_shot_caps is None:
            self.get_zeroshot_caption()

        return (
            img,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            dicom_path,
            one_hot_labels,
            self.zero_shot_caps,
            one_hot_labels,
            img,
        )


class RSNAMammo(torch.utils.data.Dataset):

    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        llm_type="gpt",
        max_words=72,
        img_size=1024,
        structural_cap=False,
        natural_cap=False,
        simple_cap=False,
        balanced_test=False,
        raw_caption=False,
        less_train_neg=0,
        k_fold=0,
        fold=0,
        csv_path=None,
        img_root=None,
        path_pattern=None,
        patient_col="patient_id",
        image_col="image_id",
        label_col="cancer",
        split_col="split",
        train_split_value="training",
        test_split_value="test",
        *args,
        **kwargs,
    ):
        # Keep official behavior: map 'test' to 'valid' for RSNA default CSVs.
        # For custom CSVs (e.g., EMBED-derived), we allow an explicit 'test' split when split_col exists.
        if csv_path is None and split == "test" and (not k_fold or k_fold == 0):
            split = "valid"
        assert split in ["train", "valid", "test"]

        self.transform = transform
        self.llm_type = llm_type
        self.img_size = img_size
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.natural_cap = natural_cap
        self.simple_cap = simple_cap
        self.raw_caption = raw_caption
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.n_classes = 2

        # ---------------------------
        # Load dataframe
        # ---------------------------
        if csv_path is None:
            # Official RSNA behavior
            # Keep original behavior when k_fold==0:
            #   - train uses RSNA_MAMMO_TRAIN_CSV
            #   - valid/test use RSNA_MAMMO_TEST_CSV (and 'test' is mapped to 'valid' above)
            # When k_fold>0, train/valid must come from the SAME pool (train CSV),
            # and test should come from the official test CSV.
            if k_fold and k_fold > 0:
                if split in ["train", "valid"]:
                    self.df = pd.read_csv(RSNA_MAMMO_TRAIN_CSV)
                else:  # split == "test"
                    if balanced_test:
                        warnings.warn(
                            "Balanced test set is not supported for RSNA Mammography dataset, move to original test set"
                        )
                    self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)
            else:
                if split == "train":
                    self.df = pd.read_csv(RSNA_MAMMO_TRAIN_CSV)
                else:
                    if balanced_test:
                        warnings.warn(
                            "Balanced test set is not supported for RSNA Mammography dataset, move to original test set"
                        )
                    self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)
            _patient_col = "patient_id"
            _image_col = "image_id"
            _label_col = "cancer"
            _data_root = RSNA_MAMMO_JPEG_DIR
            _path_pattern = "{pid}/{iid}_resized.jpg"
        else:
            # Custom CSV (e.g., EMBED-derived cancer CSV) using RSNA training pipeline
            self.df = pd.read_csv(csv_path)

            # Auto-detect common EMBED column names if user kept defaults
            _label_col = label_col
            if _label_col not in self.df.columns and "Cancer" in self.df.columns:
                _label_col = "Cancer"
            _patient_col = patient_col if patient_col in self.df.columns else "patient_id"
            _image_col = image_col if image_col in self.df.columns else "image_id"

            # Optional split filtering (e.g., split in {'training','test'})
            # Per your requirement:
            #   - when k_fold==0: keep behavior close to original project (do not force training/valid filtering);
            #     still allow explicit test filtering if user requests split="test"
            #   - when k_fold>0: train/valid are drawn from 'training', and test is drawn from 'test'
            if split_col in self.df.columns:
                if k_fold and k_fold > 0:
                    if split in ["train", "valid"]:
                        self.df = self.df[self.df[split_col] == train_split_value]
                    elif split == "test":
                        self.df = self.df[self.df[split_col] == test_split_value]
                else:
                    if split == "test":
                        self.df = self.df[self.df[split_col] == test_split_value]

            _data_root = img_root if img_root is not None else RSNA_MAMMO_JPEG_DIR
            _path_pattern = path_pattern if path_pattern is not None else "{pid}/{iid}"

        # Validate path pattern early to avoid crashing deep in training loop.
        # Expected placeholders: {pid}, {iid}
        try:
            _ = _path_pattern.format(pid="sample_pid", iid="sample_iid")
        except Exception as e:
            warnings.warn(
                f"Invalid path_pattern='{_path_pattern}' ({e}). "
                "Falling back to '{pid}/{iid}'."
            )
            _path_pattern = "{pid}/{iid}"

        # ---------------------------
        # Patient-level K-fold (RSNA pipeline only)
        # ---------------------------
        if k_fold and k_fold > 0 and split in ["train", "valid"]:
            assert k_fold >= 2, "k_fold must be >= 2 for KFold"
            assert 0 <= fold < k_fold, f"fold must be in [0, {k_fold-1}]"
            from sklearn.model_selection import KFold

            uniq_pids = self.df[_patient_col].astype(str).unique()
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            splits = list(kf.split(uniq_pids))
            train_idx, val_idx = splits[fold]
            train_pids = set(uniq_pids[train_idx])
            val_pids = set(uniq_pids[val_idx])

            if split == "train":
                self.df = self.df[self.df[_patient_col].astype(str).isin(train_pids)]
            else:
                self.df = self.df[self.df[_patient_col].astype(str).isin(val_pids)]
            self.df = self.df.reset_index(drop=True)

        # stash for later path/label construction
        self._patient_col = _patient_col
        self._image_col = _image_col
        self._label_col = _label_col
        self._data_root = _data_root
        self._path_pattern = _path_pattern
        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=data_pct)
        if less_train_neg > 0 and _label_col in self.df.columns:
            df_neg = self.df[self.df[_label_col] == 0]
            df_pos = self.df[self.df[_label_col] == 1]
            df_neg = df_neg.sample(frac=less_train_neg)
            self.df = pd.concat([df_neg, df_pos])

        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.path2label = {}
        self.labels = []
        missing_cnt = 0
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            label = int(entry[self._label_col])
            pid = str(entry[self._patient_col])
            iid = str(entry[self._image_col])
            rel_path = self._path_pattern.format(pid=pid, iid=iid)
            path = os.path.join(self._data_root, rel_path)
            if not os.path.exists(path):
                missing_cnt += 1
            self.labels.append(label)
            self.filenames.append(path)
            self.path2label[path] = label
        print("### Sampled split distribution: ", Counter(self.labels))

    def __len__(self):
        return len(self.df)

    def get_caption(self, series_sents):
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        tokens["masked_ids"] = tokens["input_ids"]

        return tokens, x_len

    def get_zeroshot_caption(self):
        base_captions = ""
        zero_shot_caps = []
        zero_shot_caps_len = []
        for label, canncer_desc in RSNA_MAMMO_CANCER_DESC.items():
            # build density caption following training format
            birads, birads_desc = RSNA_MAMMO_BIRADS_DESC[label]
            if self.structural_cap:
                # findings
                captions = (
                    base_captions
                    + EMBED_FINDINGS
                    + EMBED_FINDS_CAPTION
                    + canncer_desc
                    + " "
                )
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", birads
                ).replace("{{BIRADS_DESC}}", birads_desc)
                # overall assesment
                captions += EMBED_ASSESSMENT + birads_desc
            elif self.natural_cap:
                # findings
                captions = base_captions + EMBED_FINDS_CAPTION + canncer_desc + " "
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", birads_desc)
            else:
                captions = (
                    base_captions
                    + BREAST_BASE_CAPTION
                    + BREAST_BIRADS_CAPTION
                    + str(birads)
                    + ": "
                    + birads_desc
                    + "."
                )
            # Update caption type if using raw style caption
            if self.raw_caption:
                captions = captions.replace(".", " " + self.tokenizer.sep_token)
                captions = captions.replace(";", " " + self.tokenizer.sep_token)
                if self.llm_type != "bert":
                    captions = (
                        self.tokenizer.bos_token
                        + " "
                        + captions
                        + " "
                        + self.tokenizer.eos_token
                    )
            else:
                captions = captions.replace("\n", " ").lower()
            cap, cap_len = self.get_caption([captions])
            zero_shot_caps.append(cap)
            zero_shot_caps_len.append(cap_len)

        stacked_caps = {}
        for cap in zero_shot_caps:
            for k, v in cap.items():
                if k not in stacked_caps:
                    stacked_caps[k] = v
                else:
                    stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
        zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        self.zero_shot_caps = stacked_caps
        self.zero_shot_caps_len = zero_shot_caps_len

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        label = self.labels[idx]
        path = self.filenames[idx]

        img = get_imgs(path, scale=self.img_size, transform=self.transform)
        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        if self.zero_shot_caps is None:
            self.get_zeroshot_caption()

        return (
            img,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            path,
            one_hot_labels,
            self.zero_shot_caps,
            one_hot_labels,
            img,
        )


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            OtsuCut(),
            transforms.Resize((512, 512)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1550), (0.1521)),
        ]
    )
    dataset = VinDr(
        "data/csv/breast-level_annotations.csv",
        transform=transform,
        binary=False,
        test=False,
    )

    img, label = dataset.__getitem__(10)
    print(torch.mean(img), label)
    plt.imsave("./tmp/vindr_img.jpg", img.squeeze().numpy())