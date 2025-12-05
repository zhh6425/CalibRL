# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch
import random
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F


def get_score_mean_std(id2score):
    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:               
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    return id2mean, id2std


def get_score(
    score, id2mean, id2std, epsilon,
    norm_adv_by_std_in_grpo=False):
    if norm_adv_by_std_in_grpo:
        return (score - id2mean) / (id2std + epsilon)
    else:
        return score - id2mean


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_mix_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    pref_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    mixup_policy: str = 'random',
    mixup_ratio: float = 1.0,
    pass_ratio_low: float = 0.0,
    pass_ratio_high: float = 1.0,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length). 
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
            
    * New:
        mixup_policy:
            "pass":   sample by pass ratio
            "random": random sample
            "std":    sample by score std

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # entropys = entropys.detach()
    scores = token_level_rewards.sum(dim=-1)
    pref_log_prob = torch.zeros_like(old_log_prob)

    # record the score for on/on+off policy
    id2score = defaultdict(list)
    id2pass = defaultdict(list)
    id2random = defaultdict(bool)
    id2prob = defaultdict(torch.Tensor)
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if not pref_mask[i].any():     # only on-policy sample
                id2score[index[i]].append(scores[i])
            if pref_mask[i].any() and not index[i] in id2prob:
                id2prob[index[i]] = old_log_prob[i]
                    
        for id in np.unique(index):
            # record random for each group
            id2random[id] = random.random() <= mixup_ratio
            # record pass ratio for each group
            id2pass[id] = sum(1 for s in id2score[id] if s > 0.1) / len(id2score[id])  
                                 
        id2mean, id2std = get_score_mean_std(id2score)
        
        keep_indices = []
        pass_sample_mask = torch.zeros((bsz, 1), dtype=torch.bool)
        for i in range(bsz):
            # pref_log_prob
            if index[i] in id2prob:
                pref_log_prob[i] = id2prob[index[i]]
    
            pass_sample_mask[i] = scores[i] > 0.1
            if mixup_policy == 'pass':
                in_pass_range = id2pass[index[i]] >= pass_ratio_low and id2pass[index[i]] <= pass_ratio_high
                if in_pass_range:
                    keep_indices.append(i)
                else:
                    if not pref_mask[i].any():  # keep on-policy data
                        keep_indices.append(i)   
            elif mixup_policy == 'random':
                if id2random[index[i]]:
                    keep_indices.append(i)
                else:
                    if not pref_mask[i].any():  # keep on-policy data
                        keep_indices.append(i)   
            scores[i] = get_score(
                scores[i], id2mean[index[i]], id2std[index[i]],
                epsilon, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo, 
                )
            
        scores = scores.unsqueeze(-1) * response_mask
        
    return {
        "advantages": scores, 
        "returns": scores,
        "keep_indices": keep_indices,
        "pref_log_prob": pref_log_prob,
        "pass_sample_mask": pass_sample_mask
        }


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":                # DAPO
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":      # stronger like drgrpo
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":     # GRPO
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":  # Dr.GRPO
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_mix_policy_loss(
    old_log_prob,        # (bs, seq_len)
    log_prob,            # (bs, seq_len)
    pref_log_prob,       # (bs, seq_len)
    advantages,          # (bs, seq_len)
    response_mask,       # (bs, seq_len) binary mask for valid tokens
    pref_mask,           # (bs, seq_len) binary mask marking golden tokens
    pass_sample_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    mix_policy_loss="standard",
    loss_agg_mode="token-mean"      # "token-mean" or other modes
):
    """
    Compute GRPO policy loss.

    Returns a dict with:
        pg_loss, ppo_kl, pg_clipfrac, pg_clipfrac_lower,
    """

    negative_approx_kl = log_prob - old_log_prob  # (bs, seq_len)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses = pg_losses1
    
    # clip
    low = cliprange if cliprange_low is None else cliprange_low
    high = cliprange if cliprange_high is None else cliprange_high
    
    pg_clipfrac = torch.tensor(0.0)
    pg_clipfrac_lower = torch.tensor(0.0)
    
    if mix_policy_loss in ("standard", "luffy", "rlplus", "expert-guided"):
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - low, 1 + high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

        pg_losses3 = -advantages * clip_ratio_c
        on_clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

        pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, clip_pg_losses1)

    if mix_policy_loss == "luffy": # split the on/off-policy loss
        # off-policy loss
        off_ratio = torch.exp(log_prob)
        off_ratio = off_ratio / (off_ratio + 0.1)
        off_pg_losses1 = - advantages * off_ratio
        # overall pg_loss
        pref_mask = pref_mask.float()
        pg_losses = off_pg_losses1 * pref_mask + pg_losses * (1 - pref_mask)
    
    if mix_policy_loss == "rlplus": # split the on/off-policy loss
        # off-policy loss
        off_ratio = 2*torch.exp(log_prob)/(torch.exp(old_log_prob) + 0.5*(torch.exp(old_log_prob) + 1)) 
        off_pg_losses1 = - advantages * off_ratio * (1 - torch.exp(log_prob.detach()))**(0.5)
        # overall pg_loss
        pref_mask = pref_mask.float()
        pg_losses = off_pg_losses1 * pref_mask + pg_losses * (1 - pref_mask)

    if mix_policy_loss == "expert-guided":
        # mask pg_losses
        pref_mask = pref_mask.any(dim=1)
        pg_losses = pg_losses[~pref_mask]

        logits = log_prob - pref_log_prob

        guided_terms = -(pass_sample_mask.float() * 2.0 - 1.0) * logits
        guided_losses = F.leaky_relu(guided_terms, negative_slope=0.5, inplace=True)

        weight_with_adv = True
        if weight_with_adv:
            guided_losses = torch.abs(advantages) * guided_losses
        guided_losses = guided_losses[~pref_mask]

        response_mask = response_mask[~pref_mask]
        
    # else standard
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        )
    
    return_dict = {
        "pg_loss":           pg_loss,
        "ppo_kl":            ppo_kl,
        "pg_clipfrac":       pg_clipfrac,
        "pg_clipfrac_lower": pg_clipfrac_lower,
    }

    if mix_policy_loss == "expert-guided":
        guided_loss = agg_loss(
            loss_mat=guided_losses, 
            loss_mask=response_mask, 
            loss_agg_mode=loss_agg_mode,
            )
        return_dict.update({
            "guided_loss": guided_loss,
            })
    
    return return_dict


