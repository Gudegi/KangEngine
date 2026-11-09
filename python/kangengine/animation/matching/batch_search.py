"""Device-resident, exhaustive motion matching for independent query batches."""

from typing import NamedTuple

import numpy as np
import torch

from .database import MotionMatchingDatabase


class MotionMatchBatch(NamedTuple):
    """One match per query; all tensors have shape [N].

    No eligible clip gives IDs of -1, NaN times, and infinite costs.
    Controllers leave those environments unchanged.
    """

    # Rows across all clips' search candidates, not MotionLibrary frame offsets.
    row_ids: torch.Tensor
    motion_ids: torch.Tensor
    # Frame indices within each selected motion.
    frame_ids: torch.Tensor
    times: torch.Tensor
    costs: torch.Tensor


class MotionMatchingBatchSearch:
    """Compare batched queries with every candidate frame on CPU or CUDA.

    All environments share the search data; chunks limit working memory.
    Matrix distances are fast; direct distances reduce rounding errors for
    nearly identical features. Equal computed costs select the first row.
    Treat tensors as read-only; rebuild to change features or normalization.
    """

    def __init__(
        self,
        database: MotionMatchingDatabase,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        distance_mode: str = "matrix",
        query_chunk_size: int = 1024,
        frame_chunk_size: int = 4096,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("batch search requires float32 or float64")
        if distance_mode not in ("matrix", "direct"):
            raise ValueError("distance_mode must be 'matrix' or 'direct'")
        if query_chunk_size <= 0 or frame_chunk_size <= 0:
            raise ValueError("search chunk sizes must be positive")
        self.database = database
        self.features = torch.tensor(database.features, dtype=dtype, device=device)
        self.device = self.features.device
        self.mean = torch.tensor(database.mean, dtype=dtype, device=device)
        self.scale = torch.tensor(database.scale, dtype=dtype, device=device)
        self.weights = torch.tensor(database.weights, dtype=dtype, device=device)
        self.motion_ids = torch.tensor(database.motion_indices, device=device)
        self.frame_ids = torch.tensor(database.frame_indices, device=device)
        self.fps = torch.tensor(
            [motion.fps() for motion in database.motions], dtype=dtype, device=device
        )
        counts = [clip.candidate_count for clip in database._clips]
        self._counts = torch.tensor(counts, device=device)
        self._offsets = torch.tensor(
            np.cumsum([0, *counts[:-1]]), dtype=torch.long, device=device
        )
        self._query_chunk_size = query_chunk_size
        self._frame_chunk_size = frame_chunk_size
        self._distance_mode = distance_mode
        self._squared_norms = self.features.square().sum(-1)
        self._normalization_factor = self.weights / self.scale
        self._row_ids = torch.arange(len(self.features), device=self.device)
        self._times = self.frame_ids / self.fps[self.motion_ids]

    def normalize(self, raw_queries: torch.Tensor) -> torch.Tensor:
        """Normalize raw features [..., D] using the database's feature layout."""
        return (raw_queries - self.mean) * self._normalization_factor

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,
        *,
        current_motion_ids: torch.Tensor | None = None,
        current_times: torch.Tensor | None = None,
        continuation_bias: float = 0.0,
        allowed_motion_mask: torch.Tensor | None = None,
    ) -> MotionMatchBatch:
        """Search normalized queries [N, D] without host/device transfers.

        Args:
            queries: Finite features on the search device, with its dtype.
            current_motion_ids: Valid clip IDs [N], paired with current_times.
            current_times: Seconds within each current clip [N].
            continuation_bias: Nonnegative cost reduction for the candidate nearest
                current_times; favors continuing the current clip.
            allowed_motion_mask: Boolean [M] or [N, M] mask on the search device.
                None allows every clip. M is the number of clips.
        """
        if (current_motion_ids is None) != (current_times is None):
            raise ValueError("supply current_motion_ids and current_times together")
        if continuation_bias < 0:
            raise ValueError("continuation_bias must be nonnegative")

        # Keep the working distance matrix bounded independently of DB size.
        count = queries.shape[0]
        rows = torch.full((count,), -1, dtype=torch.long, device=self.device)
        costs = queries.new_full((count,), torch.inf)
        shared_disallowed = None
        if allowed_motion_mask is not None and allowed_motion_mask.ndim == 1:
            shared_disallowed = ~allowed_motion_mask[self.motion_ids]
        for start in range(0, count, self._query_chunk_size):
            end = min(start + self._query_chunk_size, count)
            query = queries[start:end]
            if self._distance_mode == "matrix":
                query_norms = query.square().sum(-1, keepdim=True)
            continuation = None
            if current_motion_ids is not None and continuation_bias > 0:
                ids = current_motion_ids[start:end]
                frames = torch.floor(
                    current_times[start:end] * self.fps[ids] + 0.5
                ).long()
                continuation = torch.where(
                    (frames >= 0) & (frames < self._counts[ids]),
                    self._offsets[ids] + frames,
                    -1,
                )
            allowed = allowed_motion_mask
            if allowed is not None and allowed.ndim == 2:
                allowed = allowed[start:end]

            for first in range(0, len(self.features), self._frame_chunk_size):
                last = min(first + self._frame_chunk_size, len(self.features))
                if self._distance_mode == "matrix":
                    distance = (
                        query_norms
                        + self._squared_norms[first:last]
                        - 2 * (query @ self.features[first:last].T)
                    ).clamp_min_(0)
                else:
                    # Direct subtraction avoids cancellation for near-zero costs.
                    distance = torch.cdist(
                        query,
                        self.features[first:last],
                        compute_mode="donot_use_mm_for_euclid_dist",
                    ).square()
                if continuation is not None:
                    row_ids = self._row_ids[first:last]
                    distance -= continuation_bias * (continuation[:, None] == row_ids)
                if shared_disallowed is not None:
                    distance.masked_fill_(shared_disallowed[first:last], torch.inf)
                elif allowed is not None:
                    distance.masked_fill_(
                        ~allowed[..., self.motion_ids[first:last]], torch.inf
                    )
                value, local_ids = distance.min(dim=-1)
                better = value < costs[start:end]
                rows[start:end] = torch.where(
                    better, first + local_ids, rows[start:end]
                )
                costs[start:end] = torch.minimum(costs[start:end], value)

        valid = rows >= 0
        safe_rows = rows.clamp_min(0)
        ids = self.motion_ids[safe_rows]
        frames = self.frame_ids[safe_rows]
        return MotionMatchBatch(
            rows,
            torch.where(valid, ids, -1),
            torch.where(valid, frames, -1),
            torch.where(valid, self._times[safe_rows], torch.nan),
            costs,
        )
