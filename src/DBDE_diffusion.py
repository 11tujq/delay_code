# src/tianshou/dbde_diffusion.py

from typing import Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """标准的 diffusion 时间步嵌入，用在 eps 网络里."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 的时间步整数（long/int64），范围 [0, T-1]
        Returns:
            (B, dim) 的时间嵌入
        """
        if t.dtype not in (torch.float32, torch.float64):
            t = t.float()
        half_dim = self.dim // 2
        device = t.device

        # 经典的 sin/cos 位置编码
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]          # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EpsNet(nn.Module):
    """
    预测噪声 epsilon_theta(s_t^n, M_t, n) 的 MLP 网络。

    输入:
        x_noisy: 当前扩散步的带噪状态 s_t^n, 形状 (B, state_dim)
        cond:   条件 M_t（延迟信息编码），形状 (B, cond_dim)
        t:      时间步, 形状 (B,)

    输出:
        eps_pred: 噪声预测，形状 (B, state_dim)
    """
    def __init__(
        self,
        state_dim: int,
        cond_dim: int,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()

        # 时间步 -> 隐空间的 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        in_dim = state_dim + cond_dim + hidden_dim
        layers = []
        last_dim = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, state_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_noisy: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, state_dim)
            cond:   (B, cond_dim)
            t:      (B,)
        """
        temb = self.time_mlp(t)  # (B, hidden_dim)
        h = torch.cat([x_noisy, cond, temb], dim=-1)
        eps = self.mlp(h)
        return eps


class DBDEDiffusion(nn.Module):
    """
    Diffusion-based discrepancy estimator (DBDE).

    对应论文中的 p_θ(s_t | M_t)：
    - 训练:   给定真实无延迟状态 s_t 和延迟信息 M_t，最小化噪声预测损失 L_diff.
    - 采样:   给定 M_t，从 p_θ(s_t | M_t) 中采样若干 s_t.

    你可以把 M_t 理解为「延迟观测 + 历史动作」经过 concat 之后的向量。
    """
    def __init__(
        self,
        state_dim: int,                 # 真实状态 s_t 的维度
        cond_dim: int,                  # 延迟信息 M_t 的维度
        num_steps: int = 50,            # 扩散步数 N
        beta_start: float = 1e-4,       # beta schedule 起点
        beta_end: float = 2e-2,         # beta schedule 终点
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        time_emb_dim: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps

        # 线性 beta schedule
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        # 一些常用 buffer，和公式对应:
        # q(s^n | s^0) = N(s^n; sqrt(alpha_bar_n)s^0, (1-alpha_bar_n)I)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        # 后验方差，用于反向采样
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # 噪声预测网络 epsilon_θ
        self.eps_model = EpsNet(
            state_dim=state_dim,
            cond_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

    # ------------------------------------------------------------------
    # q(s_t^n | s_t) 的 closed-form 采样: 前向加噪
    # ------------------------------------------------------------------
    def q_sample(
        self,
        s0: torch.Tensor,              # (B, state_dim)
        t: torch.Tensor,               # (B,)
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        s_t^n = sqrt(alpha_bar_n) * s_t + sqrt(1 - alpha_bar_n) * eps
        """
        if noise is None:
            noise = torch.randn_like(s0)

        batch_size = s0.shape[0]
        device = s0.device
        t = t.to(device)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        return sqrt_alpha_bar * s0 + sqrt_one_minus_alpha_bar * noise

    # ------------------------------------------------------------------
    # 训练损失 L_diff: 噪声预测损失
    # ------------------------------------------------------------------
    def p_losses(
        self,
        s0: torch.Tensor,              # (B, state_dim)
        cond: torch.Tensor,            # (B, cond_dim)
        t: torch.Tensor,               # (B,)
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        单步噪声预测损失:
            L = E[ || eps - eps_θ(s_t^n, M_t, n) ||^2 ].
        """
        if noise is None:
            noise = torch.randn_like(s0)

        x_noisy = self.q_sample(s0, t, noise)
        eps_pred = self.eps_model(x_noisy, cond, t.float())
        loss = F.mse_loss(eps_pred, noise, reduction="mean")
        return loss

    def training_loss(
        self,
        s0: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        对一个 batch 计算 L_diff，用在优化时调用。

        Args:
            s0:   (B, state_dim) 真实无延迟状态 s_t
            cond: (B, cond_dim)  延迟信息 M_t

        Returns:
            dict 包含:
                "loss": 标量 L_diff
                "t":    当前采样的时间步 (B,)
        """
        batch_size = s0.shape[0]
        device = s0.device

        # n ~ Uniform({0,...,N-1})
        t = torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        loss = self.p_losses(s0, cond, t)
        return {"loss": loss, "t": t}

    # ------------------------------------------------------------------
    # 反向过程 p_θ(s^{n-1} | s^n, M_t): 单步采样
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,               # (B, state_dim) 当前步 s^n
        cond: torch.Tensor,            # (B, cond_dim)
        t: torch.Tensor,               # (B,) 当前步 n
    ) -> torch.Tensor:
        """
        从 p_θ(s^{n-1} | s^n, M_t) 采样一步。
        """
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1)
        posterior_var = self.posterior_variance[t].view(-1, 1)

        eps_theta = self.eps_model(x, cond, t.float())
        # s^{n-1} = 1/sqrt(alpha_n) * (s^n - beta_n / sqrt(1-alpha_bar_n) * eps_θ)
        model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus_alpha_bar * eps_theta)

        # 最后一步不再加噪
        if (t == 0).all():
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

    # ------------------------------------------------------------------
    # 给定 M_t，从 p_θ(s_t | M_t) 中采样
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,            # (B, cond_dim)
        num_samples_per_cond: int = 1,
    ) -> torch.Tensor:
        """
        给定条件 M_t，采样 s_t ~ p_θ(s_t | M_t)。

        Args:
            cond: (B, cond_dim)
            num_samples_per_cond: 每个条件生成多少条样本

        Returns:
            samples: (B * num_samples_per_cond, state_dim)
                     这里是把 batch 扩展了，你可以再 reshape 成
                     (B, num_samples_per_cond, state_dim) 使用。
        """
        device = self.device
        cond = cond.to(device)
        B = cond.shape[0]

        if num_samples_per_cond > 1:
            cond = cond.repeat_interleave(num_samples_per_cond, dim=0)

        # 初始 s^N 从标准高斯开始
        x = torch.randn(B * num_samples_per_cond, self.state_dim, device=device)

        # 反向扩散: n = N-1, ..., 0
        for step in reversed(range(self.num_steps)):
            t = torch.full(
                (x.shape[0],),
                step,
                device=device,
                dtype=torch.long,
            )
            x = self.p_sample(x, cond, t)

        return x
'''
# 比如你在 runner.py 里导入
from tianshou.dbde_diffusion import DBDEDiffusion

# 初始化的时候，根据 delayed 信息的维度构造
state_dim = env.observation_space.shape[0]      # 真实无延迟 state 维度
cond_dim  = some_M_t_dim                        # 你自己拼 delayed obs + 历史 action 后的维度

self.dbde = DBDEDiffusion(
    state_dim=state_dim,
    cond_dim=cond_dim,
    num_steps=self.global_cfg.actor_input.obs_pred.num_steps,
    hidden_dim=self.global_cfg.actor_input.obs_pred.feat_dim,
    device=self.cfg.device,
)

# 训练时:
#   s0: 真实无延迟状态 (比如 batch.oobs_nodelay)
#   M_t: 延迟信息 (比如 batch.M_t_flat)
loss_dict = self.dbde.training_loss(s0=batch.oobs_nodelay, cond=batch.M_t_flat)
combined_loss = combined_loss + loss_dict["loss"] * self.global_cfg.actor_input.obs_pred.diff_loss_weight

# 推理时: 给定 M_t, 采样若干个 s_t
samples = self.dbde.sample(cond=M_t_flat, num_samples_per_cond=K)    # (B*K, state_dim)
# 然后你可以 reshape 成 (B, K, state_dim) 用来做 Q 不确定性加权
samples = samples.view(batch_size, K, state_dim)

'''