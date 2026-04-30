"""
CHASEModel 训练脚本
说明：
- 保留：PCA、Whisper逐维/全局标准化、word_id、valid_phn/utt/word 指标、保存 preds、best_audio_model.pth、result.csv
- 模型来源：当前目录下的 CHASE/src/model.py -> CHASEModel
"""

import os
import sys
import time
import math
import pickle
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.decomposition import PCA

# 使用当前目录下的模型文件（src/model.py）
from model import CHASEModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


class TriStageLRScheduler(_LRScheduler):
    """Tri-Stage LR scheduler: warmup -> hold -> exponential decay -> final constant."""

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        phase_ratio: Optional[Tuple[float, float, float]] = None,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        decay_steps: int = 0,
        total_steps: int = 0,
    ):
        self.peak_lr = peak_lr
        self.init_lr = max(peak_lr * init_lr_scale, 1e-8)
        self.final_lr = max(peak_lr * final_lr_scale, 1e-8)
        if phase_ratio is not None:
            assert abs(sum(phase_ratio) - 1.0) < 1e-6, "phase ratios must sum to 1"
            assert total_steps > 0
            self.warmup_steps = int(total_steps * phase_ratio[0])
            self.hold_steps = int(total_steps * phase_ratio[1])
            self.decay_steps = max(total_steps - self.warmup_steps - self.hold_steps, 1)
        else:
            self.warmup_steps = warmup_steps
            self.hold_steps = hold_steps
            self.decay_steps = decay_steps
        self.decay_steps = max(self.decay_steps, 1)
        self.warmup_steps = max(self.warmup_steps, 0)
        self.hold_steps = max(self.hold_steps, 0)
        self.total_steps = max(total_steps, 1)
        self.update_steps = 0
        init_lr = self.init_lr
        for g in optimizer.param_groups:
            g["lr"] = init_lr
        super().__init__(optimizer)

    def _current_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        offset = self.warmup_steps
        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset
        offset += self.hold_steps
        if self.update_steps < offset + self.decay_steps:
            return 2, self.update_steps - offset
        return 3, self.update_steps - (offset + self.decay_steps)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        stage, steps_in_stage = self._current_stage()
        if stage == 0 and self.warmup_steps > 0:
            lr = self.init_lr + (self.peak_lr - self.init_lr) * (steps_in_stage / self.warmup_steps)
        elif stage == 1:
            lr = self.peak_lr
        elif stage == 2:
            decay_progress = steps_in_stage / self.decay_steps
            lr = self.peak_lr * math.exp(math.log(self.final_lr / self.peak_lr) * decay_progress)
        else:
            lr = self.final_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.update_steps += 1
        self._last_lr = [lr]
        return lr


class PCATransformer:
    """PCA主成分分析工具类：仅对有效 token 拟合。"""

    def __init__(self, n_components=None, explained_variance_ratio=0.95, whiten=False):
        self.n_components = n_components
        self.explained_variance_ratio = explained_variance_ratio
        self.whiten = whiten
        self.pca = None
        self.is_fitted = False
        self.retained_variance = None

    def fit(self, X):
        if len(X.shape) == 3:
            _, _, D = X.shape
            X_flat = X.reshape(-1, D)
            valid_mask = np.any(X_flat != 0, axis=1)
            X_valid = X_flat[valid_mask]
        else:
            X_valid = X

        if self.n_components is None:
            temp_pca = PCA()
            temp_pca.fit(X_valid)
            cumsum_var = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum_var >= self.explained_variance_ratio) + 1
            print(
                f"[PCA] 自动确定主成分数量: {self.n_components} (保留 {cumsum_var[self.n_components-1]:.2%} 的方差)"
            )

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(X_valid)
        self.is_fitted = True
        self.retained_variance = float(np.sum(self.pca.explained_variance_ratio_))
        print(
            f"[PCA] 拟合完成: {X_valid.shape[1]}维 -> {self.n_components}维 "
            f"(保留 {self.retained_variance:.2%} 的方差, whiten={self.whiten})"
        )

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCA模型尚未拟合，请先调用fit()方法")
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X_np = X.cpu().numpy()
        else:
            X_np = X

        if len(X_np.shape) == 3:
            N, T, D = X_np.shape
            X_flat = X_np.reshape(-1, D)
            X_transformed = self.pca.transform(X_flat).reshape(N, T, -1)
        else:
            X_transformed = self.pca.transform(X_np)

        if is_torch:
            return torch.tensor(X_transformed, dtype=X.dtype, device=X.device)
        return X_transformed

    def save(self, path):
        if not self.is_fitted:
            raise ValueError("PCA模型尚未拟合")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pca": self.pca,
                    "n_components": self.n_components,
                    "target_variance_ratio": self.explained_variance_ratio,
                    "retained_variance_ratio": self.retained_variance,
                    "explained_variance_spectrum": self.pca.explained_variance_ratio_,
                    "whiten": self.whiten,
                },
                f,
            )
        print(f"[PCA] 模型已保存到: {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pca = data["pca"]
        self.n_components = data["n_components"]
        self.explained_variance_ratio = data.get("target_variance_ratio", self.explained_variance_ratio)
        self.retained_variance = data.get("retained_variance_ratio", None)
        self.whiten = data.get("whiten", self.whiten)
        self.is_fitted = True
        print(f"[PCA] 模型已加载: {path} (n_components={self.n_components})")


def gen_result_header():
    phn_header = ["epoch", "phone_train_mse", "phone_train_pcc", "phone_test_mse", "phone_test_pcc", "learning rate"]
    utt_header_set = ["utt_train_mse", "utt_train_pcc", "utt_test_mse", "utt_test_pcc"]
    utt_header_score = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    word_header_set = ["word_train_pcc", "word_test_pcc"]
    word_header_score = ["accuracy", "stress", "total"]
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset + "_" + x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset + "_" + x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header


def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            if target[i, j] >= 0:
                pred_val = audio_output[i, j]
                target_val = target[i, j]
                if isinstance(pred_val, torch.Tensor):
                    pred_val = pred_val.cpu()
                if isinstance(target_val, torch.Tensor):
                    target_val = target_val.cpu()
                valid_token_pred.append(pred_val)
                valid_token_target.append(target_val)
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)
    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr


def valid_utt(audio_output, target):
    if isinstance(audio_output, torch.Tensor):
        audio_output = audio_output.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean((audio_output[:, i] - target[:, i]) ** 2)
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr


def valid_word(audio_output, target):
    if isinstance(audio_output, torch.Tensor):
        audio_output = audio_output.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target_np = target.cpu().numpy()
    else:
        target_np = target

    word_id = target_np[:, :, -1]
    target = target_np[:, :, 0:3]

    valid_token_pred = []
    valid_token_target = []

    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        for j in range(target.shape[1]):
            cur_w_id = int(word_id[i, j])
            if cur_w_id != prev_w_id:
                if start_id < j and prev_w_id >= 0:
                    valid_token_pred.append(np.mean(audio_output[i, start_id:j, :], axis=0))
                    valid_token_target.append(np.mean(target[i, start_id:j, :], axis=0))
                if start_id < j and prev_w_id >= 0:
                    if len(np.unique(target[i, start_id:j, 1])) != 1:
                        print(target[i, start_id:j, 0])
                if cur_w_id == -1:
                    break
                else:
                    prev_w_id = cur_w_id
                    start_id = j

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target


def validate(audio_model, val_loader, args, best_mse=999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    with torch.no_grad():
        for i, (
            audio_input,
            phn_label,
            phns,
            utt_label,
            word_label,
            word_ids,
            dur_feat,
            energy_feat,
            whisper_feat,
        ) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            phn_label = phn_label.to(device)
            utt_label = utt_label.to(device)
            word_label = word_label.to(device)
            word_ids = word_ids.to(device)
            dur_feat = dur_feat.to(device)
            energy_feat = energy_feat.to(device)
            whisper_feat = whisper_feat.to(device)

            u1, u2, u3, u4, u5, p, w1, w2, w3, gate_w = audio_model(
                audio_input,
                phns,
                dur_feat,
                energy_feat,
                whisper_feat=whisper_feat,
                word_ids=word_ids,
            )
            p = p.to("cpu").detach()
            u1, u2, u3, u4, u5 = (
                u1.to("cpu").detach(),
                u2.to("cpu").detach(),
                u3.to("cpu").detach(),
                u4.to("cpu").detach(),
                u5.to("cpu").detach(),
            )
            w1, w2, w3 = w1.to("cpu").detach(), w2.to("cpu").detach(), w3.to("cpu").detach()

            A_phn.append(p)
            A_phn_target.append(phn_label.to("cpu").detach())

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label.to("cpu").detach())

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            word_label_with_id = torch.cat([word_label[:, :, 0:3], word_ids.unsqueeze(-1)], dim=-1)
            A_word_target.append(word_label_with_id.to("cpu").detach())

        A_phn, A_phn_target = torch.cat(A_phn), torch.cat(A_phn_target)
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = (
            torch.cat(A_u1),
            torch.cat(A_u2),
            torch.cat(A_u3),
            torch.cat(A_u4),
            torch.cat(A_u5),
            torch.cat(A_utt_target),
        )
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)

        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        if phn_mse < best_mse:
            print("new best phn mse {:.3f}, now saving predictions.".format(phn_mse))

            if os.path.exists(args.exp_dir + "/preds") is False:
                os.mkdir(args.exp_dir + "/preds")

            if os.path.exists(args.exp_dir + "/preds/phn_target.npy") is False:
                np.save(args.exp_dir + "/preds/phn_target.npy", A_phn_target.numpy())
                np.save(args.exp_dir + "/preds/word_target.npy", valid_word_target)
                np.save(args.exp_dir + "/preds/utt_target.npy", A_utt_target.numpy())

            np.save(args.exp_dir + "/preds/phn_pred.npy", A_phn.numpy())
            np.save(args.exp_dir + "/preds/word_pred.npy", valid_word_pred)
            np.save(args.exp_dir + "/preds/utt_pred.npy", A_utt.numpy())

    audio_model.train()
    stats_phn = [phn_mse, phn_corr]
    stats_utt = [utt_mse, utt_corr]
    stats_word = word_corr
    return stats_phn, stats_utt, stats_word


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on " + str(device))

    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print("Total parameter number is : {:.3f} k".format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print("Total trainable parameter number is : {:.3f} k".format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=1e-4, betas=(0.95, 0.999))

    total_steps = args.n_epochs * len(train_loader)
    scheduler = None
    if args.lr_scheduler == "tristage":
        scheduler = TriStageLRScheduler(
            optimizer,
            peak_lr=args.lr,
            init_lr_scale=args.tri_init_scale,
            final_lr_scale=args.tri_final_scale,
            phase_ratio=(args.tri_warmup_ratio, args.tri_hold_ratio, args.tri_decay_ratio),
            total_steps=total_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    loss_fn = nn.MSELoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 32])
    max_grad_norm = getattr(args, "max_grad_norm", 0.0)

    while epoch < args.n_epochs:
        audio_model.train()
        for i, (
            audio_input,
            phn_label,
            phns,
            utt_label,
            word_label,
            word_ids,
            dur_feat,
            energy_feat,
            whisper_feat,
        ) in enumerate(train_loader):
            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
            word_ids = word_ids.to(device, non_blocking=True)
            dur_feat = dur_feat.to(device, non_blocking=True)
            energy_feat = energy_feat.to(device, non_blocking=True)
            whisper_feat = whisper_feat.to(device, non_blocking=True)

            if args.lr_scheduler != "tristage":
                warm_up_step = 100
                if global_step <= warm_up_step and global_step % 5 == 0:
                    warm_lr = (global_step / warm_up_step) * args.lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warm_lr
                    print("warm-up learning rate is {:f}".format(optimizer.param_groups[0]["lr"]))

            noise = (torch.rand([audio_input.shape[0], audio_input.shape[1], audio_input.shape[2]]) - 1) * args.noise
            noise = noise.to(device, non_blocking=True)
            audio_input = audio_input + noise

            u1, u2, u3, u4, u5, p, w1, w2, w3, gate_w = audio_model(
                audio_input,
                phns,
                dur_feat,
                energy_feat,
                whisper_feat=whisper_feat,
                word_ids=word_ids,
            )

            mask = phn_label >= 0
            p = p.squeeze(2)
            p = p * mask
            phn_label_masked = phn_label * mask
            loss_phn = loss_fn(p, phn_label_masked)
            loss_phn = loss_phn * (mask.shape[0] * mask.shape[1]) / torch.sum(mask)

            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds, utt_label)

            word_label_phn = word_label[:, :, 0:3]
            word_mask = word_label_phn >= 0
            word_pred = torch.cat((w1, w2, w3), dim=2)
            word_pred = word_pred * word_mask
            word_label_phn = word_label_phn * word_mask
            loss_word = loss_fn(word_pred, word_label_phn)
            loss_word = loss_word * (word_mask.shape[0] * word_mask.shape[1] * word_mask.shape[2]) / torch.sum(word_mask)

            phone_from_word = (w3.squeeze(2)) * mask
            phone_pred_detach = p.detach()
            loss_consist = loss_fn(phone_pred_detach, phone_from_word)
            loss_consist = loss_consist * (mask.shape[0] * mask.shape[1]) / torch.sum(mask)

            loss = (
                args.loss_w_phn * loss_phn
                + args.loss_w_utt * loss_utt
                + args.loss_w_word * loss_word
                + args.loss_w_consist * loss_consist
            )

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(audio_model.parameters(), max_grad_norm)
            optimizer.step()
            if args.lr_scheduler == "tristage":
                scheduler.step()
            global_step += 1

        stats_phn_train, stats_utt_train, stats_word_train = validate(audio_model, train_loader, args, best_mse)
        stats_phn_test, stats_utt_test, stats_word_test = validate(audio_model, test_loader, args, best_mse)

        mse_phn_train, pcc_phn_train = stats_phn_train
        mse_phn_test, pcc_phn_test = stats_phn_test

        print("-------------------validation results-------------------")
        print("Phone: Train MSE: {:.6f}, PCC: {:.6f}".format(mse_phn_train, pcc_phn_train))
        print("Phone: Test MSE: {:.6f}, PCC: {:.6f}".format(mse_phn_test, pcc_phn_test))
        print(
            "Utterance Train PCC: ACC={:.3f}, COM={:.3f}, FLU={:.3f}, PROC={:.3f}, Total={:.3f}".format(
                stats_utt_train[1][0],
                stats_utt_train[1][1],
                stats_utt_train[1][2],
                stats_utt_train[1][3],
                stats_utt_train[1][4],
            )
        )
        print(
            "Utterance Test PCC: ACC={:.3f}, COM={:.3f}, FLU={:.3f}, PROC={:.3f}, Total={:.3f}".format(
                stats_utt_test[1][0],
                stats_utt_test[1][1],
                stats_utt_test[1][2],
                stats_utt_test[1][3],
                stats_utt_test[1][4],
            )
        )
        print(
            "Word Train PCC: ACC={:.3f}, Stress={:.3f}, Total={:.3f}".format(
                stats_word_train[0], stats_word_train[1], stats_word_train[2]
            )
        )
        print(
            "Word Test PCC: ACC={:.3f}, Stress={:.3f}, Total={:.3f}".format(
                stats_word_test[0], stats_word_test[1], stats_word_test[2]
            )
        )
        print("--------------------------------------------------------")

        result[epoch, 0] = epoch
        result[epoch, 1] = mse_phn_train
        result[epoch, 2] = pcc_phn_train
        result[epoch, 3] = mse_phn_test
        result[epoch, 4] = pcc_phn_test
        result[epoch, 5] = optimizer.param_groups[0]["lr"]
        result[epoch, 6:11] = stats_utt_train[0]
        result[epoch, 11:16] = stats_utt_train[1]
        result[epoch, 16:21] = stats_utt_test[0]
        result[epoch, 21:26] = stats_utt_test[1]
        result[epoch, 26:29] = stats_word_train
        result[epoch, 29:32] = stats_word_test

        os.makedirs(exp_dir, exist_ok=True)
        np.savetxt(exp_dir + "/result.csv", result, delimiter=",", header=",".join(gen_result_header()), comments="")
        print("result saved to " + exp_dir + "/result.csv")

        if mse_phn_test < best_mse:
            best_mse = mse_phn_test
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/best_audio_model.pth" % (exp_dir))
            print("best model saved to " + exp_dir + "/best_audio_model.pth")

        if args.lr_scheduler != "tristage":
            scheduler.step()
        print("Epoch-{0} lr: {1}".format(epoch, optimizer.param_groups[0]["lr"]))

        epoch += 1


class GoPWhisperDataset(Dataset):
    """仅加载 GOP + dur/energy (+可选Whisper) 特征。"""

    def __init__(
        self,
        set,
        am="librispeech",
        use_pca=False,
        pca_components=None,
        pca_variance_ratio=0.95,
        pca_model_path=None,
        pca_transformer=None,
        pca_target_dim=None,
        whisper_cfg=None,
    ):
        if am == "librispeech":
            dir = "seq_data_librispeech"
            norm_mean, norm_std = 3.203, 4.045
        elif am == "paiia":
            dir = "seq_data_paiia"
            norm_mean, norm_std = -0.652, 9.737
        elif am == "paiib":
            dir = "seq_data_paiib"
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError("Acoustic Model Unrecognized.")

        data_root = str(DATA_ROOT)

        if set == "train":
            self.feat = torch.tensor(np.load(f"{data_root}/{dir}/tr_feat.npy").astype(np.float32), dtype=torch.float32)
            self.phn_label = torch.tensor(
                np.load(f"{data_root}/{dir}/tr_label_phn.npy").astype(np.float32), dtype=torch.float32
            )
            self.utt_label = torch.tensor(
                np.load(f"{data_root}/{dir}/tr_label_utt.npy").astype(np.float32), dtype=torch.float32
            )
            self.word_label = torch.tensor(
                np.load(f"{data_root}/{dir}/tr_label_word.npy").astype(np.float32), dtype=torch.float32
            )
            self.word_id = torch.tensor(np.load(f"{data_root}/{dir}/tr_word_id.npy"), dtype=torch.long)
            self.dur_feat = torch.tensor(
                np.load(f"{data_root}/{dir}/tr_dur_feat.npy").astype(np.float32), dtype=torch.float32
            )
            self.energy_feat = torch.tensor(
                np.load(f"{data_root}/{dir}/tr_energy_feat.npy").astype(np.float32), dtype=torch.float32
            )
        elif set == "test":
            self.feat = torch.tensor(np.load(f"{data_root}/{dir}/te_feat.npy").astype(np.float32), dtype=torch.float32)
            self.phn_label = torch.tensor(
                np.load(f"{data_root}/{dir}/te_label_phn.npy").astype(np.float32), dtype=torch.float32
            )
            self.utt_label = torch.tensor(
                np.load(f"{data_root}/{dir}/te_label_utt.npy").astype(np.float32), dtype=torch.float32
            )
            self.word_label = torch.tensor(
                np.load(f"{data_root}/{dir}/te_label_word.npy").astype(np.float32), dtype=torch.float32
            )
            self.word_id = torch.tensor(np.load(f"{data_root}/{dir}/te_word_id.npy"), dtype=torch.long)
            self.dur_feat = torch.tensor(
                np.load(f"{data_root}/{dir}/te_dur_feat.npy").astype(np.float32), dtype=torch.float32
            )
            self.energy_feat = torch.tensor(
                np.load(f"{data_root}/{dir}/te_energy_feat.npy").astype(np.float32), dtype=torch.float32
            )
        else:
            raise ValueError("set must be train/test")

        # normalize GOP
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

        # PCA
        self.use_pca = use_pca
        self.original_feat_dim = self.feat.shape[-1]
        if use_pca:
            if pca_target_dim is not None:
                pca_components = pca_target_dim
                print(f"[Dataset] 使用目标维度: {pca_target_dim}维")
            if set == "train":
                self.pca_transformer = PCATransformer(n_components=pca_components, explained_variance_ratio=pca_variance_ratio)
                self.pca_transformer.fit(self.feat.numpy())
                self.feat = torch.tensor(self.pca_transformer.transform(self.feat.numpy()), dtype=torch.float32)
                if pca_model_path:
                    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
                    self.pca_transformer.save(pca_model_path)
            else:
                if pca_transformer is not None:
                    self.pca_transformer = pca_transformer
                elif pca_model_path and os.path.exists(pca_model_path):
                    self.pca_transformer = PCATransformer()
                    self.pca_transformer.load(pca_model_path)
                else:
                    raise ValueError("测试集需要提供PCA模型（pca_transformer或pca_model_path）")
                self.feat = torch.tensor(self.pca_transformer.transform(self.feat.numpy()), dtype=torch.float32)
            print(f"[Dataset] PCA降维: {self.original_feat_dim}维 -> {self.feat.shape[-1]}维")
        else:
            self.pca_transformer = None

        # normalize labels
        self.utt_label = self.utt_label / 5
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

        # whisper
        self.sample_ids = []
        self.whisper_cfg = whisper_cfg or {}
        self.use_whisper = self.whisper_cfg.get("enabled", False)
        self.whisper_feat_dim = 0
        self.whisper_file_paths = None
        if self.use_whisper:
            self._init_whisper_features(
                set_name=set,
                am=am,
                # 默认 Whisper 特征目录
                feat_root=self.whisper_cfg.get(
                    "feat_root",
                    str(PROJECT_ROOT / "data" / "whisper_feature" / "feature_aligned" / "whisper_block25_features"),
                ),
                norm_mean=self.whisper_cfg.get("norm_mean", 0.441998),
                norm_std=self.whisper_cfg.get("norm_std", 2.464786),
            )

    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def _init_whisper_features(self, set_name, am, feat_root, norm_mean, norm_std):
        split_prefix = "tr" if set_name == "train" else "te"
        keys_dir = str(DATA_ROOT / "raw_kaldi_gop" / am)
        keys_file = os.path.join(keys_dir, f"{split_prefix}_keys_phn.csv")
        if not os.path.exists(keys_file):
            raise FileNotFoundError(f"Keys file not found: {keys_file}")

        with open(keys_file, "r") as f:
            all_keys = [line.strip() for line in f.readlines()]
        sample_ids = list(dict.fromkeys([k.split(".")[0] for k in all_keys]))
        data_len = self.feat.shape[0]
        if len(sample_ids) != data_len:
            min_len = min(len(sample_ids), data_len)
            print(f"[Whisper] Warning: sample_id count ({len(sample_ids)}) != data count ({data_len}), trimming to {min_len}")
            sample_ids = sample_ids[:min_len]
            self.feat = self.feat[:min_len]
            self.phn_label = self.phn_label[:min_len]
            self.utt_label = self.utt_label[:min_len]
            self.word_label = self.word_label[:min_len]
            self.word_id = self.word_id[:min_len]
            self.dur_feat = self.dur_feat[:min_len]
            self.energy_feat = self.energy_feat[:min_len]
        self.sample_ids = sample_ids

        self.whisper_raw_dim = 1280
        whisper_split = "train" if set_name == "train" else "test"
        feat_dir = os.path.join(feat_root, whisper_split)
        if not os.path.isdir(feat_dir):
            raise FileNotFoundError(f"Whisper feature directory not found: {feat_dir}")
        files = {
            os.path.splitext(f)[0]: os.path.join(feat_dir, f)
            for f in os.listdir(feat_dir)
            if f.endswith(".npy")
        }
        self.whisper_file_paths = []
        matched = 0
        for sid in self.sample_ids:
            path = files.get(sid)
            self.whisper_file_paths.append(path)
            if path is not None:
                matched += 1
        print(f"[Whisper] {set_name} set: matched {matched}/{len(self.sample_ids)} samples with block25 features")

        self._whisper_norm_eps = 1e-6
        if isinstance(norm_mean, (list, tuple)):
            norm_mean = np.asarray(norm_mean, dtype=np.float32)
        if isinstance(norm_std, (list, tuple)):
            norm_std = np.asarray(norm_std, dtype=np.float32)
        if isinstance(norm_mean, np.ndarray):
            if norm_mean.shape[0] != self.whisper_raw_dim:
                raise ValueError(f"[Whisper] norm_mean 维度 {norm_mean.shape} 与期望 {self.whisper_raw_dim} 不符")
        else:
            norm_mean = float(norm_mean)
        if isinstance(norm_std, np.ndarray):
            if norm_std.shape[0] != self.whisper_raw_dim:
                raise ValueError(f"[Whisper] norm_std 维度 {norm_std.shape} 与期望 {self.whisper_raw_dim} 不符")
            norm_std = np.maximum(norm_std, self._whisper_norm_eps)
        self.whisper_norm_mean = norm_mean
        if isinstance(norm_std, np.ndarray):
            self.whisper_norm_std = norm_std.astype(np.float32)
        else:
            self.whisper_norm_std = float(norm_std)
            if self.whisper_norm_std < self._whisper_norm_eps:
                self.whisper_norm_std = self._whisper_norm_eps
        self.whisper_feat_dim = self.whisper_raw_dim

    def _normalize_whisper_array(self, feat: np.ndarray) -> np.ndarray:
        if isinstance(self.whisper_norm_mean, np.ndarray):
            return (feat - self.whisper_norm_mean[None, :]) / (self.whisper_norm_std[None, :] + self._whisper_norm_eps)
        else:
            return (feat - self.whisper_norm_mean) / (self.whisper_norm_std + self._whisper_norm_eps)

    def _pad_or_truncate(self, feat, target_len):
        T = feat.shape[0]
        if T == target_len:
            return feat
        if T > target_len:
            return feat[:target_len]
        pad = np.zeros((target_len - T, feat.shape[1]), dtype=feat.dtype)
        return np.concatenate([feat, pad], axis=0)

    def _load_whisper_tensor(self, idx, seq_len):
        if not self.use_whisper:
            return torch.zeros(seq_len, 0, dtype=torch.float32)
        path = self.whisper_file_paths[idx]
        if path is None or not os.path.exists(path):
            feat = np.zeros((seq_len, self.whisper_raw_dim), dtype=np.float32)
        else:
            feat = np.load(path).astype(np.float32)
            if feat.ndim != 2:
                feat = np.zeros((seq_len, self.whisper_raw_dim), dtype=np.float32)
            else:
                feat = self._normalize_whisper_array(feat)
                feat = self._pad_or_truncate(feat, seq_len)
        return torch.tensor(feat, dtype=torch.float32)

    def __getitem__(self, idx):
        dur_feat = self.dur_feat[idx, :]
        energy_feat = self.energy_feat[idx, :]
        seq_len = self.feat[idx, :].shape[0]
        whisper_feat = self._load_whisper_tensor(idx, seq_len)
        return (
            self.feat[idx, :],
            self.phn_label[idx, :, 1],
            self.phn_label[idx, :, 0],
            self.utt_label[idx, :],
            self.word_label[idx, :],
            self.word_id[idx, :],
            dur_feat,
            energy_feat,
            whisper_feat,
        )

    def __len__(self):
        return self.feat.shape[0]


def collate_fn(batch):
    (
        gop_feats,
        phn_labels,
        phns,
        utt_labels,
        word_labels,
        word_ids,
        dur_feats,
        energy_feats,
        whisper_feats,
    ) = zip(*batch)
    return (
        torch.stack(gop_feats),
        torch.stack(phn_labels),
        torch.stack(phns),
        torch.stack(utt_labels),
        torch.stack(word_labels),
        torch.stack(word_ids),
        torch.stack(dur_feats),
        torch.stack(energy_feats),
        torch.stack(whisper_feats),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-train", type=str, default="train", help="training data json")
    parser.add_argument("--data-eval", type=str, default="test", help="evaluation data json")
    parser.add_argument("--exp-dir", type=str, default="./exp/whisper_gradformer", help="directory to dump experiments")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument("--am", type=str, default="librispeech", help="acoustic model type")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=25, help="batch size")
    parser.add_argument("--loss_w_phn", type=float, default=1.0, help="weight for phoneme-level loss")
    parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
    parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
    parser.add_argument("--loss_w_consist", type=float, default=0.2, help="weight for phone↔word consistency loss")
    parser.add_argument("--noise", type=float, default=0.02, help="noise augmentation")
    parser.add_argument("--lr-scheduler", type=str, choices=["multistep", "tristage"], default="tristage", help="LR scheduler type")
    parser.add_argument("--tri-warmup-ratio", type=float, default=0.1, help="warmup ratio for tri-stage scheduler")
    parser.add_argument("--tri-hold-ratio", type=float, default=0.4, help="hold ratio for tri-stage scheduler")
    parser.add_argument("--tri-decay-ratio", type=float, default=0.5, help="decay ratio for tri-stage scheduler")
    parser.add_argument("--tri-init-scale", type=float, default=0.01, help="initial LR scale relative to peak for tri-stage")
    parser.add_argument("--tri-final-scale", type=float, default=0.05, help="final LR scale relative to peak for tri-stage")

    parser.add_argument("--embed-dim", type=int, default=24, help="embedding dimension")
    parser.add_argument("--goptdepth", type=int, default=1, help="depth of gopt models")
    parser.add_argument("--goptheads", type=int, default=1, help="heads of gopt models")
    parser.add_argument("--feat-drop", type=float, default=0.1, help="Whisper feature dropout rate")
    parser.add_argument("--conv-kernel", type=int, default=31, help="Depthwise conv kernel size")
    parser.add_argument("--conv-dropout", type=float, default=0.1, help="Depthwise conv dropout rate")
    parser.add_argument("--dur-dim", type=int, default=1, help="duration feature dimension")
    parser.add_argument("--energy-dim", type=int, default=7, help="energy feature dimension")
    parser.add_argument("--max-grad-norm", type=float, default=5.0, help="gradient clipping max norm (0 to disable)")
    parser.add_argument("--word-aspect-fusion-layers", type=int, default=3, help="Word级Aspect融合层数")
    parser.add_argument("--word-aspect-fusion-dropout", type=float, default=0.15, help="Word级Aspect融合的dropout")

    parser.add_argument("--use-whisper-feat", action="store_true", help="启用Whisper对齐特征")
    parser.add_argument("--no-whisper-feat", dest="use_whisper_feat", action="store_false", help="禁用Whisper特征")
    parser.add_argument(
        "--whisper-feat-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "whisper_feature" / "feature_aligned" / "whisper_block25_features"),
        help="Whisper特征根目录（包含train/test子目录）",
    )
    parser.add_argument("--whisper-block-id", type=int, default=25, help="Whisper Block编号（仅用于日志）")
    parser.add_argument(
        "--whisper-stat-path",
        type=str,
        default=str(DATA_ROOT / "whisper_feature.npz"),
        help="Whisper特征逐维统计(npz，含mean/std)",
    )
    parser.add_argument("--whisper-mean", type=float, default=0.441998, help="Whisper特征标准化均值")
    parser.add_argument("--whisper-std", type=float, default=2.464786, help="Whisper特征标准化标准差")

    parser.add_argument("--use-pca", action="store_true", help="对GOP特征使用PCA降维")
    parser.add_argument("--no-pca", dest="use_pca", action="store_false", help="禁用PCA降维")
    parser.add_argument("--pca-components", type=int, default=None, help="PCA主成分数量（None则根据方差比例自动确定）")
    parser.add_argument("--pca-variance-ratio", type=float, default=0.95, help="PCA保留的方差比例（0-1之间）")

    parser.set_defaults(use_pca=True, use_whisper_feat=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.exp_dir + "/preds", exist_ok=True)

    print("Now train CHASEModel with GOP+Whisper features and phone-level alignment")

    aux_dim = args.dur_dim + args.energy_dim
    if args.use_pca:
        if args.pca_components is not None:
            pca_target_dim = args.pca_components
        else:
            pca_target_dim = 14
    else:
        pca_target_dim = None
    if pca_target_dim is not None:
        print(f"[PCA] 目标维度: GOP降到{pca_target_dim}维，再拼接{args.dur_dim}+{args.energy_dim}={aux_dim}维辅助特征")

    whisper_cfg_train = None
    if args.use_whisper_feat:
        whisper_norm_mean = args.whisper_mean
        whisper_norm_std = args.whisper_std
        if args.whisper_stat_path:
            with np.load(args.whisper_stat_path) as stats_npz:
                whisper_norm_mean = stats_npz["mean"].astype(np.float32)
                whisper_norm_std = stats_npz["std"].astype(np.float32)
                stat_tokens = int(stats_npz["count"]) if "count" in stats_npz.files else -1
                stat_dim = whisper_norm_mean.shape[0]
            token_str = stat_tokens if stat_tokens >= 0 else "unknown"
            print(f"[Whisper] 从 {args.whisper_stat_path} 加载逐维统计: dim={stat_dim}, tokens={token_str}")
        whisper_cfg_train = {
            "enabled": True,
            "feat_root": args.whisper_feat_root,
            "block": args.whisper_block_id,
            "norm_mean": whisper_norm_mean,
            "norm_std": whisper_norm_std,
        }

    pca_model_path = os.path.join(args.exp_dir, "pca_model.pkl") if args.use_pca else None
    train_dataset = GoPWhisperDataset(
        args.data_train,
        am=args.am,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        pca_variance_ratio=args.pca_variance_ratio,
        pca_model_path=pca_model_path,
        pca_target_dim=pca_target_dim,
        whisper_cfg=whisper_cfg_train,
    )
    pca_transformer = train_dataset.pca_transformer if args.use_pca else None
    actual_input_dim = train_dataset.feat.shape[-1] if args.use_pca else 84
    whisper_feat_dim = train_dataset.whisper_feat_dim if args.use_whisper_feat else 0

    whisper_cfg_val = None
    if args.use_whisper_feat:
        whisper_cfg_val = dict(whisper_cfg_train)
    val_dataset = GoPWhisperDataset(
        args.data_eval,
        am=args.am,
        use_pca=args.use_pca,
        pca_transformer=pca_transformer,
        pca_model_path=pca_model_path,
        pca_target_dim=pca_target_dim,
        whisper_cfg=whisper_cfg_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2500, shuffle=False, collate_fn=collate_fn, num_workers=2)

    audio_model = CHASEModel(
        embed_dim=args.embed_dim,
        depth=args.goptdepth,
        input_dim=actual_input_dim,
        num_heads=args.goptheads,
        dur_dim=args.dur_dim,
        energy_dim=args.energy_dim,
        feat_drop=args.feat_drop,
        conv_kernel=args.conv_kernel,
        conv_dropout=args.conv_dropout,
        whisper_dim=whisper_feat_dim,
        word_aspect_fusion_layers=args.word_aspect_fusion_layers,
        word_aspect_fusion_dropout=args.word_aspect_fusion_dropout,
    )
    print(f"[Model] 输入特征维度: {actual_input_dim} (原始: 84, PCA: {'启用' if args.use_pca else '禁用'})")
    if args.use_whisper_feat:
        print(f"[Model] Whisper特征维度: {whisper_feat_dim}")

    train(audio_model, train_loader, val_loader, args)

