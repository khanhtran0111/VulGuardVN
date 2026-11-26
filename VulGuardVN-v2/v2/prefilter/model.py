"""Multi-view prefilter (plan §6.1): token CNN, AST-like CNN, UniXcoder projection, 24 numeric features, plus
the cross-fitted fine-tuned UniXcoder logit (view `finetune`, added 2026-11-23).
Outputs logits for the fusion head (s_fus) and the auxiliary heads p_sem / p_gra, as in v1."""
from __future__ import annotations

import torch
from torch import nn

ALL_VIEWS = ("token", "ast", "semantic", "numeric", "finetune")


class SeqCNN(nn.Module):
    def __init__(self, vocab: int, dim: int, filters: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.dropout = nn.Dropout1d(dropout)
        self.conv1 = nn.Conv1d(dim, filters, 5, padding=2)
        self.conv2 = nn.Conv1d(filters, filters, 3, padding=1)
        self.out_dim = filters

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(1)                      # [B,1,L]
        x = self.embedding(ids).transpose(1, 2)            # [B,D,L]
        x = self.dropout(x)
        x = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        x = x.masked_fill(~mask, -1e4)
        pooled = x.max(dim=2).values
        return torch.where(pooled < -1e3, torch.zeros_like(pooled), pooled)


class MultiViewPrefilter(nn.Module):
    def __init__(self, token_vocab: int, ast_vocab: int, semantic_dim: int, numeric_dim: int, cfg: dict,
                 views: tuple[str, ...] = ALL_VIEWS) -> None:
        super().__init__()
        self.views = tuple(v for v in ALL_VIEWS if v in views)
        p, proj, dense, drop = cfg, int(cfg["projection_dim"]), int(cfg["dense_units"]), float(cfg["dropout"])
        fusion_dim = 0
        graph_dim = 0
        if "token" in self.views:
            self.token = SeqCNN(token_vocab, int(p["token_embedding_dim"]), int(p["token_filters"]), drop)
            fusion_dim += self.token.out_dim
        if "ast" in self.views:
            self.ast = SeqCNN(ast_vocab, int(p["ast_embedding_dim"]), int(p["ast_filters"]), drop)
            fusion_dim += self.ast.out_dim
            graph_dim += self.ast.out_dim
        if "semantic" in self.views:
            self.sem_proj = nn.Sequential(nn.Linear(semantic_dim, proj), nn.ReLU(), nn.Dropout(drop))
            self.sem_head = nn.Sequential(nn.Linear(proj, max(32, proj // 2)), nn.ReLU(), nn.Linear(max(32, proj // 2), 1))
            fusion_dim += proj + 1
        if "numeric" in self.views:
            fusion_dim += numeric_dim
            graph_dim += numeric_dim
        if "finetune" in self.views:
            fusion_dim += 2                                   # standardised logit + its sigmoid
        self.has_graph = graph_dim > 0
        if self.has_graph:
            self.gra_proj = nn.Sequential(nn.Linear(graph_dim, proj), nn.ReLU(), nn.Dropout(drop))
            self.gra_head = nn.Sequential(nn.Linear(proj, max(32, proj // 2)), nn.ReLU(), nn.Linear(max(32, proj // 2), 1))
            fusion_dim += 1
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, dense), nn.ReLU(), nn.Dropout(drop),
                                    nn.Linear(dense, max(64, dense // 2)), nn.ReLU(), nn.Linear(max(64, dense // 2), 1))

    def forward(self, tokens, ast, semantic, numeric, finetune=None) -> dict[str, torch.Tensor]:
        parts, graph_parts, out = [], [], {}
        if "token" in self.views:
            parts.append(self.token(tokens))
        if "ast" in self.views:
            a = self.ast(ast)
            parts.append(a)
            graph_parts.append(a)
        if "semantic" in self.views:
            s = self.sem_proj(semantic)
            sem_logit = self.sem_head(s).squeeze(1)
            parts += [s, torch.sigmoid(sem_logit).unsqueeze(1)]
            out["sem"] = sem_logit
        if "numeric" in self.views:
            parts.append(numeric)
            graph_parts.append(numeric)
        if "finetune" in self.views:
            z = finetune.unsqueeze(1)
            parts += [z, torch.sigmoid(z)]
        if self.has_graph:
            g = self.gra_proj(torch.cat(graph_parts, dim=1))
            gra_logit = self.gra_head(g).squeeze(1)
            parts.append(torch.sigmoid(gra_logit).unsqueeze(1))
            out["gra"] = gra_logit
        out["fus"] = self.fusion(torch.cat(parts, dim=1)).squeeze(1)
        return out
