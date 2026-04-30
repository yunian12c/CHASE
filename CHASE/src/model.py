"""
CHASEModel + Whisper 多分支融合模型
- Phone层：GLMFBlock + Whisper特征融合
- Word层：phone特征聚合 + GAT
- Sentence层：word特征 + Whisper句子级特征融合
"""

import sys
import os
import math

# 相对路径引用本工程内的基础子模块（src/model/hipama.py）
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
if _LOCAL_MODEL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_MODEL_DIR)
from hipama import HiPAMA
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 模型架构定义 ==========
class GLMFBlock(nn.Module):
    """
    V3版多尺度特征融合块（支持局部+全局注意力）
    - scales: 不同比例的线性降维
    - 每个 scale 都有:
        - 全局注意力
        - 局部注意力（卷积近似）
        - Depthwise Conv
        - FFN
    - 多尺度输出加权融合
    - 残差门控 + RMSNorm
    """
    def __init__(self, dim, heads=4, scales=(1,2,4), ff_mult=2, conv_kernel=5, local_window=7):
        super().__init__()
        self.scales = scales
        self.local_window = local_window
        self.blocks = nn.ModuleList()
        self.scale_weight = nn.Parameter(torch.ones(len(scales)))
        # 全局/局部注意力融合权重
        self.attn_weight = nn.Parameter(torch.tensor([0.5,0.5]))
        self.res_gate = nn.Parameter(torch.tensor(1.0))
        self.fuse_norm = RMSNorm(d_model=dim)

        for s in scales:
            scale_dim = dim // s
            scale_heads = math.gcd(scale_dim, heads)
            if scale_heads == 0:
                raise ValueError(f"Scale {s} produces invalid dimension {scale_dim} for dim={dim}.")
            if scale_heads != heads:
                print(f"[V3] 调整 scale={s} 的多头数为 {scale_heads} 以匹配维度 {scale_dim}.")
            self.blocks.append(nn.ModuleDict({
                "down": nn.Linear(dim, scale_dim),
                "attn_global": nn.MultiheadAttention(scale_dim, scale_heads, batch_first=True),
                "conv_local": nn.Conv1d(scale_dim, scale_dim, conv_kernel, padding=(conv_kernel-1)//2, groups=scale_dim),
                "conv_ff": nn.Sequential(
                    nn.Conv1d(scale_dim, scale_dim, 1),
                    nn.SiLU(),
                ),
                "ff": nn.Sequential(
                    RMSNorm(d_model=scale_dim),
                    nn.Linear(scale_dim, ff_mult * scale_dim * 2),
                    nn.GLU(dim=-1),
                    nn.Linear(ff_mult * scale_dim, scale_dim),
                ),
                "up": nn.Linear(scale_dim, dim),
                "norm": RMSNorm(d_model=dim),
            }))

    def forward(self, x):
        outs = []
        attn_w = torch.softmax(self.attn_weight, dim=0)  # 全局/局部注意力权重

        for blk in self.blocks:
            h = blk["down"](x)

            # --- 全局注意力 ---
            h_global, _ = blk["attn_global"](h, h, h)

            # --- 局部注意力（用卷积近似） ---
            h_local = blk["conv_local"](h.transpose(1,2)).transpose(1,2)
            h_local = blk["conv_ff"](h_local.transpose(1,2)).transpose(1,2)

            # --- 注意力融合 ---
            h = h + attn_w[0]*h_global + attn_w[1]*h_local

            # --- FFN ---
            h = h + blk["ff"](h)

            # --- 升维回原维度 ---
            out = blk["up"](h)
            outs.append(out)

        # --- 多尺度加权融合 ---
        scale_w = torch.softmax(self.scale_weight, dim=0)
        fused = sum(w*o for w,o in zip(scale_w, outs))

        # --- 残差门控 + RMSNorm ---
        out = x + torch.sigmoid(self.res_gate)*fused
        return self.fuse_norm(out)

class OPAP(nn.Module):
    """
    OPAP: Order-Preserving Aggregation & Re-projection
    - 根据连续的 word_id 做有序聚合（phone -> word）
    - 再把 word 级别预测高效地回投到 phone 级别（word -> phone）
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def aggregate(word_ids: torch.Tensor, values: torch.Tensor):
        """
        根据 word_id 将逐 phone 的特征或标签聚合到词级别（按样本逐一重新编号）。
        使用 unique_consecutive 保序，适合 word_id 在时间上连续的场景。
        """
        if word_ids is None or values is None:
            raise ValueError("word_ids and values must be provided for aggregation")
        B, T = word_ids.shape
        D = values.shape[-1]
        device = word_ids.device
        per_sample_aggs = []
        per_sample_masks = []
        max_words = 1
        for b in range(B):
            valid_mask = word_ids[b] >= 0
            valid_ids = word_ids[b][valid_mask]
            valid_vals = values[b][valid_mask]
            if valid_ids.numel() == 0:
                per_sample_aggs.append(values.new_zeros((1, D)))
                per_sample_masks.append(torch.zeros(1, dtype=torch.bool, device=device))
                continue
            # 使用 unique_consecutive 保序（适合同一 word_id 在时间上连续的假设）
            uniq_ids, inverse = torch.unique_consecutive(valid_ids, return_inverse=True)
            num_words = uniq_ids.size(0)
            max_words = max(max_words, num_words)
            agg = values.new_zeros((num_words, D))
            agg.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, D), valid_vals)
            counts = values.new_zeros((num_words, 1))
            counts.scatter_add_(0, inverse.unsqueeze(-1), torch.ones_like(valid_vals[:, :1]))
            agg = agg / counts.clamp(min=1.0)
            per_sample_aggs.append(agg)
            mask = torch.ones(num_words, dtype=torch.bool, device=device)
            per_sample_masks.append(mask)
        agg_tensor = values.new_zeros((B, max_words, D))
        valid_tensor = torch.zeros((B, max_words), dtype=torch.bool, device=device)
        for b, (agg_b, mask_b) in enumerate(zip(per_sample_aggs, per_sample_masks)):
            agg_tensor[b, :agg_b.size(0)] = agg_b
            valid_tensor[b, :mask_b.size(0)] = mask_b
        return agg_tensor, valid_tensor

    @staticmethod
    def expand(word_ids: torch.Tensor, word_pred: torch.Tensor):
        """
        将word级别的预测扩展回phone级别（与 aggregate 反向操作）。
        使用 unique_consecutive 保序，并通过 inverse 映射直接 gather，复杂度 O(T)。
        """
        if word_ids is None or word_pred is None:
            raise ValueError("word_ids and word_pred must be provided")
        B, T = word_ids.shape
        D = word_pred.shape[-1]
        device = word_ids.device
        phone_pred = word_pred.new_zeros((B, T, D))
        
        for b in range(B):
            valid_mask = word_ids[b] >= 0
            if valid_mask.sum() == 0:
                continue
            
            ids = word_ids[b][valid_mask]
            # 使用 unique_consecutive 保序（与 aggregate 逻辑一致）
            uniq_ids, inverse = torch.unique_consecutive(ids, return_inverse=True)
            # inverse: [num_valid]，每个 phone 对应它的词索引
            
            # 直接使用 inverse 映射来 gather，O(T) 复杂度
            phone_pred[b, valid_mask] = word_pred[b, inverse]
        
        return phone_pred


def compute_word_relative_positions(word_ids: torch.Tensor) -> torch.Tensor:
    """
    计算每个 phone 在所属单词中的相对位置（0~1线性刻度）。
    word_ids: [B, T]，同一个单词的 id 在时间维度上连续，相同 id 代表同一单词。
    返回: [B, T, 1] 的 float32 tensor，padding（id<0）的相对位置为 0。
    """
    if word_ids is None:
        raise ValueError("word_ids must be provided to compute relative positions")
    B, T = word_ids.shape
    device = word_ids.device
    rel_pos = torch.zeros(B, T, 1, device=device, dtype=torch.float32)

    for b in range(B):
        ids_b = word_ids[b]
        start = None
        prev_id = None
        for t in range(T):
            curr = ids_b[t].item()
            if curr < 0:
                if start is not None:
                    length = t - start
                    if length > 0:
                        positions = torch.linspace(0.0, 1.0, steps=length, device=device)
                        rel_pos[b, start:t, 0] = positions
                    start = None
                    prev_id = None
                continue
            if start is None:
                start = t
                prev_id = curr
            elif curr != prev_id:
                end = t
                length = end - start
                positions = torch.linspace(0.0, 1.0, steps=length, device=device)
                rel_pos[b, start:end, 0] = positions
                start = t
                prev_id = curr
        if start is not None:
            end = T
            length = end - start
            positions = torch.linspace(0.0, 1.0, steps=length, device=device)
            rel_pos[b, start:end, 0] = positions

    return rel_pos


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.scale


class DepthwiseConvModule(nn.Module):
    """深度可分离卷积模块：Depthwise Conv + Pointwise Conv"""
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        assert 3 <= kernel_size <= 15, "kernel_size should be between 3 and 15"
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.pointwise_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.layer_norm = RMSNorm(d_model)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x + residual


class LinearBranch(nn.Module):
    """轻量分支：单层线性网络 + GELU（输出 5 维：ACC/COM/FLU/PROC/TOTAL）。"""
    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)   # 从 1 改成 out_dim=5
        )

    def forward(self, x):
        # x: [B, D] → [B, 5]
        return self.net(x)


class AspectFusionLayer(nn.Module):
    """Aspect融合层"""
    
    def __init__(self, embed_dim, alpha=0.2, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.Linear(embed_dim, 1)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        q = self.Wq(x).unsqueeze(2)
        k = self.Wk(x).unsqueeze(1)
        e = self.leaky_relu(self.attn(torch.tanh(q + k))).squeeze(-1)
        alpha = torch.softmax(e, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.bmm(alpha, x)
        return self.ln(out + x)


class AspectFusionModule(nn.Module):
    """Aspect融合模块"""
    
    def __init__(self, embed_dim, num_layers=2, alpha=0.2, dropout=0.1):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.layers = nn.ModuleList([
            AspectFusionLayer(embed_dim, alpha=alpha, dropout=dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class GatedMultiBranchRegressionHead(nn.Module):
    """门控多分支回归头（句子层 5 维打分）。"""
    def __init__(self, embed_dim, num_branches=5, hidden_dim=256, temperature=0.7, branch_dropout=0.1):
        super().__init__()
        assert num_branches > 0, "num_branches must be positive"
        self.num_branches = num_branches
        self.temperature = temperature

        self.input_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, num_branches)
        self.softmax = nn.Softmax(dim=-1)

        # 每个分支都输出 5 维
        self.branches = nn.ModuleList([
            LinearBranch(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=branch_dropout, out_dim=5)
            for _ in range(num_branches)
        ])

    def forward(self, utt_emb):
        """
        utt_emb: [B, D]
        返回:
            branch_outputs: [B, num_branches, 5]
            fused_output:   [B, 5]        # 门控多分支融合后的 5 维评分
            gate_weights:   [B, num_branches]
            total_out:      [B, 5]        # 为了兼容旧代码，= fused_output
        """
        x = self.input_norm(utt_emb)                # [B, D]

        # gate 权重
        gate_logits = self.gate(x) / self.temperature   # [B, E]
        gate_weights = self.softmax(gate_logits)        # [B, E]

        # 所有分支的输出
        branch_scores = [branch(x) for branch in self.branches]  # E 个 [B,5]
        branch_outputs = torch.stack(branch_scores, dim=1)       # [B, E, 5]

        # 软集成：所有分支按 gate 权重全量参与
        weights = gate_weights.unsqueeze(-1)  # [B, E, 1]
        
        # 门控多分支融合：对分支输出做加权和
        fused_output = torch.sum(weights * branch_outputs, dim=1)         # [B, 5]

        # 为了兼容原来的四个输出形式，最后一个就继续返回 fused_output
        return branch_outputs, fused_output, gate_weights, fused_output


class CHASEModel(nn.Module):
    """CHASEModel: GOP + dur/energy + Whisper（辅助）的多分支评估模型。"""
    def __init__(
        self,
        embed_dim,
        depth,
        input_dim=84,
        num_heads=4,
        dur_dim=1,
        energy_dim=7,
        feat_drop=0.1,
        conv_kernel=31,
        conv_dropout=0.1,
        whisper_dim=0,
        word_aspect_fusion_layers=3,
        word_aspect_fusion_dropout=0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dur_dim = dur_dim
        self.energy_dim = energy_dim
        self.whisper_dim = whisper_dim
        aux_dim = dur_dim + energy_dim
        
        # 本地基础子模块（音素投影与低层读出头）
        self.hipama = HiPAMA(embed_dim=embed_dim, depth=depth, input_dim=input_dim, num_heads=num_heads)
        # 使用 V3 多尺度融合块（含全局/局部注意力加权）
        self.phone_zip = GLMFBlock(embed_dim, heads=4, scales=(1, 2, 4), ff_mult=2, conv_kernel=5)
        self.phone_zip_post_norm = nn.LayerNorm(embed_dim)
        self.phone_zip_post_ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.phone_base_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.phone_conv_module = DepthwiseConvModule(embed_dim, kernel_size=7)
        self.phone_ln = nn.LayerNorm(embed_dim)
        
        self.aux_norm = nn.LayerNorm(aux_dim)
        self.aux_drop = nn.Dropout(feat_drop)

        # 底层只融合 GOP + dur/energy，Whisper 作为辅助信号在后续阶段注入
        fused_input_dim = input_dim + aux_dim
        self.fuse_linear = nn.Sequential(
            nn.Linear(fused_input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(feat_drop),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        print(f"[Model] 输入融合 MLP: {fused_input_dim}维 -> {embed_dim}维")
        
        if self.whisper_dim > 0:
            self.whisper_sent_proj = nn.Sequential(
                nn.LayerNorm(self.whisper_dim),
                nn.Dropout(feat_drop),
                nn.Linear(self.whisper_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(feat_drop),
            )
            self.sentence_input_merge = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.whisper_sent_proj = None
            self.sentence_input_merge = None

        # Word级
        self.aspect_gat = AspectFusionModule(
            embed_dim,
            num_layers=word_aspect_fusion_layers,
            dropout=word_aspect_fusion_dropout,
        )
        
        # Aspect融合：使用Linear替代mean
        # 将3个aspect拼接后通过Linear投影融合
        self.aspect_fuse = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(feat_drop),
        )
        
        # 句子级门控多分支回归头
        self.sentence_ln = nn.LayerNorm(embed_dim)
        self.gated_head = GatedMultiBranchRegressionHead(embed_dim=embed_dim, num_branches=5, hidden_dim=256)
        
        
        # Word级：恢复多尺度Zipformer增强
        self.word_zip = GLMFBlock(embed_dim, heads=4, scales=(1, 2), ff_mult=2, conv_kernel=5)
        self.word_zip_norm = nn.LayerNorm(embed_dim)
        self.word_zip_ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.word_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.word_attn_norm = nn.LayerNorm(embed_dim)
        self.word_attn_ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        # OPAP: Order-Preserving Aggregation & Re-projection
        self.opap = OPAP()
        # 新增：Word层双向LSTM后处理
        self.word_lstm = nn.LSTM(
            embed_dim,
            embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.word_lstm_dropout = nn.Dropout(0.1)
        self.word_lstm_gate = nn.Parameter(torch.tensor(-1.0))  # 初始让LSTM注入小一点
        self.word_head = nn.Linear(embed_dim, 3)
        self.use_word_relative_pos = True
        if self.use_word_relative_pos:
            self.word_pos_proj = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Dropout(word_aspect_fusion_dropout),
            )
        else:
            self.word_pos_proj = None
        
    def forward(self, x, phn, dur_feat, energy_feat, whisper_feat=None, word_ids=None):
        """
        x: GOP特征 [B, seq_len, input_dim] (PCA降维后，如14维)
        phn: 音素标签 [B, seq_len]
        dur_feat / energy_feat: Phone对齐时长&能量
        whisper_feat: Whisper特征 [B, seq_len, whisper_dim]（可选）
        """
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn >= 0)
        
        # 简化：直接拼接GOP、dur、energy特征
        aux_feat = torch.cat([dur_feat, energy_feat], dim=-1)
        aux_feat = self.aux_norm(aux_feat)
        aux_aligned = self.aux_drop(aux_feat)
        
        # Whisper辅助特征
        if self.whisper_dim > 0:
            if whisper_feat is None or whisper_feat.size(-1) == 0:
                whisper_feat = torch.zeros(B, seq_len, self.whisper_dim, device=x.device, dtype=x.dtype)
            else:
                whisper_feat = whisper_feat[:, :seq_len, :]
        else:
            whisper_feat = None
        
        fused_feat = torch.cat([x, aux_aligned], dim=-1)  # [B, seq_len, fused_input_dim]
        x = self.fuse_linear(fused_feat)  # [B, seq_len, embed_dim]
        
        phn_one_hot = torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        phn_embed = self.hipama.phn_proj(phn_one_hot)
        x = x + phn_embed
        
        # Phone层：GLMFBlock + LN+FF
        x_zip = self.phone_zip(x)
        x_post = self.phone_zip_post_ff(self.phone_zip_post_norm(x_zip))
        x_zip_out = x_zip + x_post  # [B, seq_len, embed_dim]
        
        x_conv_original = self.phone_base_conv(x_zip_out.transpose(1, 2)).transpose(1, 2)
        x_conv_depthwise = self.phone_conv_module(x_zip_out)
        phone_feature = self.phone_ln(x_conv_original + x_conv_depthwise + x_zip_out)
        
        # 7. Phone score
        p = self.hipama.mlp_head_phn(phone_feature).reshape(B, seq_len, 1)
        
        word_input = phone_feature
        w1_rep = self.hipama.rep_w1(word_input)
        w2_rep = self.hipama.rep_w2(word_input)
        w3_rep = self.hipama.rep_w3(word_input)
        
        w_stack = torch.stack([w1_rep, w2_rep, w3_rep], dim=2)  # [B, seq_len, 3, embed_dim]
        B, T, A, D = w_stack.shape
        w_stack_flat = w_stack.view(B * T, A, D)
        
        w_stack_updated = self.aspect_gat(w_stack_flat)
        w_stack_updated = w_stack_updated.view(B, T, A, D)  # [B, T, 3, D]
        
        # 使用Linear融合替代mean：拼接3个aspect后通过Linear投影
        w_stack_concat = w_stack_updated.view(B, T, A * D)  # [B, T, 3*D]
        word_rep_phone = self.aspect_fuse(w_stack_concat)  # [B, T, D]
        if (
            self.use_word_relative_pos
            and word_ids is not None
            and self.word_pos_proj is not None
        ):
            rel_pos = compute_word_relative_positions(word_ids)
            rel_pos = rel_pos.to(word_rep_phone.device, dtype=word_rep_phone.dtype)
            word_rep_phone = word_rep_phone + self.word_pos_proj(rel_pos)
        if word_ids is not None:
            agg_word_rep, word_valid_mask = self.opap.aggregate(word_ids, word_rep_phone)
        else:
            agg_word_rep = word_rep_phone
            word_valid_mask = None
        w_zip = self.word_zip(agg_word_rep)
        w_zip_post = self.word_zip_ff(self.word_zip_norm(w_zip))
        w_zip_out = w_zip + w_zip_post
        # 先 BiLSTM 建立局部顺序/韵律趋势，再 MHA 做全局整合
        word_lstm_out, _ = self.word_lstm(w_zip_out)  # [B, W, D]
        word_ctx = w_zip_out + torch.sigmoid(self.word_lstm_gate) * self.word_lstm_dropout(word_lstm_out)
        # MHA 做全局配对/对齐
        if word_valid_mask is not None:
            attn_out, _ = self.word_attn(
                word_ctx, word_ctx, word_ctx, key_padding_mask=~word_valid_mask
            )
        else:
            attn_out, _ = self.word_attn(word_ctx, word_ctx, word_ctx)
        word_ctx = self.word_attn_norm(word_ctx + attn_out)
        word_ctx = word_ctx + self.word_attn_ff(word_ctx)
        word_pred = self.word_head(word_ctx)
        if word_valid_mask is not None:
            word_pred = word_pred * word_valid_mask.unsqueeze(-1).type_as(word_pred)
        # 将word级别的预测扩展回phone级别（以匹配参考实现的格式）
        if word_ids is not None:
            word_pred_phone = self.opap.expand(word_ids, word_pred)
        else:
            # 如果没有word_ids，则使用phone级别的特征直接预测（fallback）
            word_pred_phone = self.word_head(word_rep_phone)
        w1 = word_pred_phone[:, :, 0:1]
        w2 = word_pred_phone[:, :, 1:2]
        w3 = word_pred_phone[:, :, 2:3]
        
        # 句子级输入：优先使用词级特征均值，若无有效词则回退到phone均值
        utt_emb_phone = phone_feature.mean(dim=1)
        if word_valid_mask is not None:
            word_counts = word_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            word_sentence = (word_ctx * word_valid_mask.unsqueeze(-1).type_as(word_ctx)).sum(dim=1) / word_counts
            fallback_mask = (word_valid_mask.sum(dim=1, keepdim=True) == 0).type_as(word_sentence)
            utt_emb = fallback_mask * utt_emb_phone + (1 - fallback_mask) * word_sentence
        else:
            utt_emb = utt_emb_phone
        if self.whisper_dim > 0 and whisper_feat is not None:
            whisper_sentence = self.whisper_sent_proj(whisper_feat).mean(dim=1)
            utt_emb = self.sentence_input_merge(torch.cat([utt_emb, whisper_sentence], dim=-1))
        utt_emb = self.sentence_ln(utt_emb)
        
        branch_out, fused_out, gate_weights, final_out = self.gated_head(utt_emb)
        # fused_out: [B, 5] = [ACC, COM, FLU, PROC, TOTAL] 的门控多分支融合结果
        u1 = fused_out[:, 0:1]   # ACC
        u2 = fused_out[:, 1:2]   # COM
        u3 = fused_out[:, 2:3]   # FLU
        u4 = fused_out[:, 3:4]   # PROC
        u5 = fused_out[:, 4:5]   # TOTAL
        
        return u1, u2, u3, u4, u5, p, w1, w2, w3, gate_weights
