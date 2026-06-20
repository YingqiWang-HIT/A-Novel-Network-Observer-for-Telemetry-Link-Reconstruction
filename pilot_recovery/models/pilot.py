# -*- coding: utf-8 -*-
"""PILOT model: anomaly-retentive hybrid networked observer."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PILOTMixedSpectrumCell(nn.Module):
    """Stable asynchronous mixed-spectrum state flow for irregular telemetry events."""

    def __init__(
        self,
        n_channels: int,
        input_dim: int,
        state_dim: int = 48,
        n_osc: int = 4,
        trend_dim: int = 12,
        hidden: int = 128,
        rho_min: float = 0.02,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.n_osc = n_osc
        self.osc_dim = 2 * n_osc
        self.trend_dim = min(trend_dim, max(1, state_dim - self.osc_dim - 1))
        self.slow_dim = state_dim - self.osc_dim - self.trend_dim
        if self.slow_dim < 1:
            raise ValueError("state_dim must be larger than 2*n_osc + trend_dim")
        self.time_dim = max(0, input_dim - 3 * n_channels)
        self.rho_min = rho_min

        node_in_dim = 3 + self.time_dim
        self.drive = nn.Sequential(nn.Linear(node_in_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, state_dim))
        self.param = nn.Sequential(
            nn.Linear(node_in_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 2 * n_osc + 2 + state_dim)
        )
        self.jump = nn.Sequential(nn.Linear(node_in_dim + 1, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, state_dim))
        self.jump_gain = nn.Sequential(
            nn.Linear(node_in_dim + 1, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, state_dim), nn.Sigmoid()
        )
        self.readout = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        base = torch.linspace(0.25, 1.25, n_osc, dtype=torch.float32)
        self.log_base_omega = nn.Parameter(torch.log(base))

    def _node_features(self, x_t: torch.Tensor):
        N = self.n_channels
        v = x_t[:, :N]
        m = x_t[:, N:2 * N]
        d = x_t[:, 2 * N:3 * N]
        if self.time_dim > 0:
            tf = x_t[:, 3 * N:].unsqueeze(1).expand(-1, N, -1)
            node = torch.cat([v.unsqueeze(-1), m.unsqueeze(-1), d.unsqueeze(-1), tf], dim=-1)
        else:
            node = torch.stack([v, m, d], dim=-1)
        return node, v, m, d

    def propagate(self, h: torch.Tensor, node: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        p = self.param(node)
        rho_raw = p[..., :self.n_osc]
        omg_raw = p[..., self.n_osc:2 * self.n_osc]
        eta_raw = p[..., 2 * self.n_osc:2 * self.n_osc + 1]
        eps_raw = p[..., 2 * self.n_osc + 1:2 * self.n_osc + 2]
        gate_raw = p[..., 2 * self.n_osc + 2:]

        dt = (1.0 + 8.0 * delta).unsqueeze(-1)
        rho = self.rho_min + 0.12 * F.softplus(rho_raw)
        omega0 = torch.exp(self.log_base_omega).view(1, 1, self.n_osc)
        omega = omega0 * (1.0 + 0.35 * torch.tanh(omg_raw))

        osc = h[..., :self.osc_dim].view(B, N, self.n_osc, 2)
        decay = torch.exp(-rho.unsqueeze(-1) * dt.unsqueeze(-1)).clamp(0.02, 1.0)
        angle = omega.unsqueeze(-1) * dt.unsqueeze(-1)
        c, s = torch.cos(angle), torch.sin(angle)
        x0, x1 = osc[..., 0:1], osc[..., 1:2]
        rot0 = c * x0 - s * x1
        rot1 = s * x0 + c * x1
        osc_next = (decay * torch.cat([rot0, rot1], dim=-1)).reshape(B, N, self.osc_dim)

        trend = h[..., self.osc_dim:self.osc_dim + self.trend_dim]
        slow = h[..., self.osc_dim + self.trend_dim:]
        eta = 0.01 + 0.10 * F.softplus(eta_raw)
        eps = 0.002 + 0.015 * torch.sigmoid(eps_raw)
        trend_next = trend * torch.exp(-eta * dt).clamp(0.02, 1.0)
        slow_next = slow * torch.exp(-eps * dt).clamp(0.80, 1.0)
        h_flow = torch.cat([osc_next, trend_next, slow_next], dim=-1)

        drive = torch.tanh(self.drive(node))
        input_gate = torch.sigmoid(gate_raw) * (1.0 - torch.exp(-0.25 * dt))
        return h_flow + input_gate * drive

    def correct(self, h: torch.Tensor, node: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        y_prior = self.readout(h).squeeze(-1)
        innov = (value - y_prior).unsqueeze(-1)
        jump_in = torch.cat([node, innov], dim=-1)
        dz = torch.tanh(self.jump(jump_in))
        gain = self.jump_gain(jump_in) * mask.unsqueeze(-1)
        h_post = h + gain * dz
        y_post = self.readout(h_post).squeeze(-1)
        return h_post, y_prior, y_post, innov.squeeze(-1)


class PILOTReliabilityHypergraph(nn.Module):
    """Reliability-conditioned local/cross/global hypergraph coupling."""

    def __init__(self, A_local, A_cross, A_global, state_dim: int, hidden: int = 128, dropout: float = 0.08):
        super().__init__()
        self.register_buffer("A_local", torch.tensor(A_local, dtype=torch.float32))
        self.register_buffer("A_cross", torch.tensor(A_cross, dtype=torch.float32))
        self.register_buffer("A_global", torch.tensor(A_global, dtype=torch.float32))
        self.msg_local = nn.Linear(state_dim, state_dim)
        self.msg_cross = nn.Linear(state_dim, state_dim)
        self.msg_global = nn.Linear(state_dim, state_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(state_dim * 4 + 1), nn.Linear(state_dim * 4 + 1, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 3)
        )
        self.strength = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(state_dim)

    def _rel_message(self, h: torch.Tensor, reliability: torch.Tensor, A: torch.Tensor, linear: nn.Module):
        src = h * reliability.unsqueeze(-1)
        num = torch.einsum("ij,bjd->bid", A, src)
        den = torch.einsum("ij,bj->bi", A, reliability).unsqueeze(-1).clamp_min(1e-4)
        return linear(num / den)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor):
        reliability = (mask * torch.exp(-3.0 * delta)).clamp(0.0, 1.0)
        ml = F.silu(self._rel_message(h, reliability, self.A_local, self.msg_local))
        mc = F.silu(self._rel_message(h, reliability, self.A_cross, self.msg_cross))
        mg = F.silu(self._rel_message(h, reliability, self.A_global, self.msg_global))
        alpha = torch.softmax(self.gate(torch.cat([h, ml, mc, mg, reliability.unsqueeze(-1)], dim=-1)), dim=-1)
        msg = alpha[..., 0:1] * ml + alpha[..., 1:2] * mc + alpha[..., 2:3] * mg
        lam = 0.35 * torch.sigmoid(self.strength)
        h_new = h + lam * (msg - h)
        return self.norm(h_new), lam.detach()


class PILOTProjectedSeparator(nn.Module):
    """Finite-window projected innovation separator with sparse persistent support."""

    def __init__(self, n_channels: int, pred_len: int, hidden: int = 128, admm_steps: int = 5, dropout: float = 0.08):
        super().__init__()
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.admm_steps = admm_steps
        self.log_tau = nn.Parameter(torch.tensor(-2.0))
        feat_dim = 8
        self.support_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.amp_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, pred_len))
        self.directed_mix = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, pred_len))

    @staticmethod
    def _soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

    def forward(self, innovation: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor):
        B, L, N = innovation.shape
        obs_count = mask.sum(dim=1).clamp_min(1.0)
        low = (innovation * mask).sum(dim=1, keepdim=True) / obs_count.unsqueeze(1)
        projected = (innovation - low) * mask
        a = projected
        tau = F.softplus(self.log_tau) + 1e-4
        for _ in range(self.admm_steps):
            left = torch.cat([a[:, :1], a[:, :-1]], dim=1)
            right = torch.cat([a[:, 1:], a[:, -1:]], dim=1)
            smooth = 0.50 * a + 0.25 * left + 0.25 * right
            a = self._soft_threshold(smooth, tau) * mask

        abs_a = torch.abs(a)
        mean_abs = abs_a.sum(dim=1) / obs_count
        max_abs = abs_a.max(dim=1).values
        last_abs = abs_a[:, -min(6, L):].mean(dim=1)
        std_abs = torch.sqrt(((abs_a - mean_abs.unsqueeze(1)) ** 2 * mask).sum(dim=1) / obs_count + 1e-6)
        persist = (abs_a > tau).float().sum(dim=1) / obs_count
        obs_rate = mask.mean(dim=1)
        last_delta = delta[:, -1]
        sign_mean = torch.tanh((a.sum(dim=1) / obs_count) * 2.0)
        feat = torch.stack([mean_abs, max_abs, last_abs, std_abs, persist, obs_rate, last_delta, sign_mean], dim=-1)

        support_prob = torch.sigmoid(self.support_head(feat).squeeze(-1))
        amp = self.amp_head(feat).permute(0, 2, 1)
        directed = 0.25 * self.directed_mix(feat).permute(0, 2, 1)
        correction = (amp + directed) * support_prob.unsqueeze(1)
        return correction, support_prob, a


class PILOTObserver(nn.Module):
    """Predictive Identifiability-aware Link Observer for telemetry recovery."""

    def __init__(
        self,
        A,
        A_local,
        A_cross,
        A_global,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        lags: Tuple[int, ...] = (0, 1, 3, 6, 12),
        mode: str = "full",
        state_dim: int = 48,
        hidden: int = 128,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.mode = mode
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.flow = PILOTMixedSpectrumCell(n_channels, input_dim, state_dim=state_dim, hidden=hidden, dropout=dropout)
        self.hypergraph = PILOTReliabilityHypergraph(A_local, A_cross, A_global, state_dim, hidden=hidden, dropout=dropout)
        self.separator = PILOTProjectedSeparator(n_channels, pred_len, hidden=hidden, admm_steps=5, dropout=dropout)
        self.temporal_pool = nn.GRU(state_dim, state_dim, num_layers=1, batch_first=True)
        self.nominal_head = nn.Sequential(
            nn.LayerNorm(state_dim * 3 + 3), nn.Linear(state_dim * 3 + 3, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, pred_len)
        )
        self.output_gate = nn.Sequential(nn.LayerNorm(state_dim + 3), nn.Linear(state_dim + 3, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(state_dim + 5), nn.Linear(state_dim + 5, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, pred_len)
        )
        self.calib_scale = nn.Parameter(torch.ones(1, pred_len, n_channels))
        self.calib_bias = nn.Parameter(torch.zeros(1, pred_len, n_channels))

    def _encode_with_coupling(self, x: torch.Tensor):
        B, L, _ = x.shape
        N = self.n_channels
        h = torch.zeros(B, N, self.state_dim, dtype=x.dtype, device=x.device)
        h_seq, y_prior_seq, y_post_seq, innov_seq, coupling_vals = [], [], [], [], []
        for t in range(L):
            node, v, m, d = self.flow._node_features(x[:, t])
            h = self.flow.propagate(h, node, d)
            if self.mode != "no_graph":
                h, lam = self.hypergraph(h, m, d)
                coupling_vals.append(lam)
            h, y_prior, y_post, innov = self.flow.correct(h, node, v, m)
            h_seq.append(h)
            y_prior_seq.append(y_prior)
            y_post_seq.append(y_post)
            innov_seq.append(innov * m)
        coupling_strength = torch.stack(coupling_vals).mean() if coupling_vals else torch.tensor(0.0, device=x.device)
        return torch.stack(h_seq, dim=1), torch.stack(y_prior_seq, dim=1), torch.stack(y_post_seq, dim=1), torch.stack(innov_seq, dim=1), coupling_strength

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        N = self.n_channels
        values = x[:, :, :N]
        mask = x[:, :, N:2 * N]
        delta = x[:, :, 2 * N:3 * N]

        h_seq, y_prior, y_post, innovation, coupling_strength = self._encode_with_coupling(x)
        h_last = h_seq[:, -1]
        z = h_seq.permute(0, 2, 1, 3).reshape(B * N, L, self.state_dim)
        _, g_last = self.temporal_pool(z)
        g_last = g_last[-1].reshape(B, N, self.state_dim)

        neigh = torch.einsum("ij,bjd->bid", self.A, h_last)
        last_node = torch.stack([values[:, -1], mask[:, -1], delta[:, -1]], dim=-1)
        head_in = torch.cat([h_last, g_last, neigh, last_node], dim=-1)
        nominal = self.nominal_head(head_in).permute(0, 2, 1)

        if self.mode == "no_separator":
            anomaly_corr = torch.zeros_like(nominal)
            support_prob = torch.zeros(B, N, dtype=x.dtype, device=x.device)
            projected_anomaly = torch.zeros_like(innovation)
        else:
            anomaly_corr, support_prob, projected_anomaly = self.separator(innovation, mask, delta)
            if self.mode == "temporal":
                anomaly_corr = torch.zeros_like(anomaly_corr)

        gate = self.output_gate(torch.cat([h_last, last_node], dim=-1)).permute(0, 2, 1)
        if self.mode in ["temporal", "no_separator"]:
            gate = torch.zeros_like(gate)
        pred = nominal + gate * anomaly_corr

        obs_rate = mask.mean(dim=1)
        neigh_obs = torch.einsum("ij,bj->bi", self.A, obs_rate)
        kappa_proxy = (obs_rate + neigh_obs).clamp_min(1e-3)
        unc_feat = torch.cat([
            h_last,
            obs_rate.unsqueeze(-1),
            delta[:, -1].unsqueeze(-1),
            support_prob.unsqueeze(-1),
            kappa_proxy.unsqueeze(-1),
            torch.abs(anomaly_corr).mean(dim=1).unsqueeze(-1),
        ], dim=-1)
        logvar = self.uncertainty_head(unc_feat).permute(0, 2, 1).clamp(-7.0, 5.0)
        uncertainty = F.softplus(logvar) + 1e-4 + (1.0 / (kappa_proxy.unsqueeze(1) + 1e-3)) * 1e-3
        pred = pred * torch.clamp(self.calib_scale, 0.90, 1.10) + torch.clamp(self.calib_bias, -0.20, 0.20)

        return pred, {
            "support_prob": support_prob,
            "anomaly_score": support_prob,
            "projected_anomaly": projected_anomaly,
            "anomaly_correction": anomaly_corr,
            "uncertainty": uncertainty,
            "logvar": logvar,
            "coupling_strength": coupling_strength,
            "history_reconstruction": y_post,
        }
