"""WalkerNet 的数据集实现。

WalkerDataset 只读取已经由 CDO 重网格后的 ``data_1x1`` 文件，不负责原始
NetCDF 的经纬度映射。重网格请先运行 ``scripts/data/remap_to_1x1.sh``。

返回的样本遵守 ``src.interfaces`` 中的最小共享接口：

    x:            (L, 4, 180, 360), float32
    y:            (1, 4, 180, 360), float32
    y_rollout:    (K, 4, 180, 360), float32，可选，用于多步滚动训练
    target_month: int in [1, 12]
    target_months:(K,), int64，可选，逐 lead 目标月份
    valid_mask:   (4, 180, 360), bool

设计原则：
- 预处理和训练读取分离：CDO remap 是一次性脚本，Dataset 只读结果。
- 支持单 source 和多 source 混合训练；一个样本内部只来自同一个 source。
- 模型输入不能包含 NaN/Inf：缺测区域用 valid_mask 标记，张量中填 0。
- 归一化统计只从训练年份计算，验证/测试复用训练统计。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass
import re

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

try:
    from .interfaces import FILL_VALUE, GRID_H, GRID_W, VARIABLES, WalkerSample
except ImportError:  # pragma: no cover - useful when running this file directly
    from interfaces import FILL_VALUE, GRID_H, GRID_W, VARIABLES, WalkerSample


REMAPPED_FILENAMES = {
    "tos": "tos_1x1.nc",
    "zos": "zos_1x1.nc",
    "tauu": "tauu_1x1.nc",
    "tauv": "tauv_1x1.nc",
}

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SourceSpec:
    """一个数据源的最小描述。

    name 用于日志、缓存后缀和调试；path 指向包含四个 ``*_1x1.nc`` 的目录。
    """

    name: str
    path: Path


class WalkerDataset(Dataset):
    """WalkerNet 单步预测数据集。

    每个样本使用历史 L 个月预测下 1 个月：

        x = data[t - L : t]
        y = data[t : t + 1]

    这里的 t 是预测目标月的时间索引。
    """

    # 进程内缓存，key 是 data_path + data_cache_path，value 是已加载好的数组和坐标。
    _data_cache: dict[tuple[Path, str], dict[str, Any]] = {}

    def __init__(
        self,
        data_path: str | Path | None,
        config: dict[str, Any],
        split: str = "train",
        norm_stats: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # ---- 基本配置检查 ----
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.config = config
        self.data_config = config.get("data", config)
        self.split = split
        self.L = int(self.data_config["L"])
        self.target_steps = int(self.data_config.get("target_steps", 1))
        self.norm = str(self.data_config.get("norm", "zscore")).lower()
        self.norm_scope = str(self.data_config.get("norm_scope", "pooled")).lower()
        self.sources = self._resolve_sources(data_path, self.data_config)
        self.source_names = tuple(source.name for source in self.sources)

        if self.L < 1:
            raise ValueError(f"Input window L must be >= 1, got {self.L}")
        if self.target_steps < 1:
            raise ValueError(f"target_steps must be >= 1, got {self.target_steps}")
        if self.norm_scope not in {"pooled", "source"}:
            raise ValueError(f"norm_scope must be 'pooled' or 'source', got {self.norm_scope!r}")

        # Dataset 和模型必须使用 interfaces.py 中固定的变量顺序。
        expected_variables = tuple(self.data_config.get("variables", VARIABLES))
        if expected_variables != VARIABLES:
            raise ValueError(f"config variables must be {VARIABLES}, got {expected_variables}")

        h = int(self.data_config.get("H", GRID_H))
        w = int(self.data_config.get("W", GRID_W))
        if (h, w) != (GRID_H, GRID_W):
            raise ValueError(f"config grid must be {(GRID_H, GRID_W)}, got {(h, w)}")

        # ---- 读取共享数据缓存 ----
        self.source_payloads: list[dict[str, Any]] = []
        for source in self.sources:
            source_config = self._source_data_config(self.data_config, source.name)
            print(f"[dataset] load source {source.name}: {source.path}", flush=True)
            self.source_payloads.append(self._load_or_get_cache(source.path, source_config))
            print(f"[dataset] source {source.name} ready", flush=True)

        # 为了兼容已有评测代码，单 source 时保留这些便捷属性。
        first_payload = self.source_payloads[0]
        self.data_path = self.sources[0].path
        self.data = first_payload["data"]
        self.years = first_payload["years"]
        self.months = first_payload["months"]
        self.lat = first_payload["lat"]
        self.lon = first_payload["lon"]
        self.valid_mask = first_payload["valid_mask"]

        # sample_indices 保存 (source_idx, target_t)，target_t 是目标月索引。
        self.sample_indices = self._build_sample_indices(split)
        if len(self.sample_indices) == 0:
            years = self.data_config[f"{split}_years"]
            raise ValueError(f"No samples found for split={split!r}, years={years}, L={self.L}")

        # train Dataset 默认自己计算归一化统计；val/test 应传入 train.norm_stats。
        # 如果 config.data.norm_stats_path 存在，则优先复用磁盘缓存，DDP 多卡时尤其重要。
        if norm_stats is None:
            stats_path = self._norm_stats_path()
            if stats_path is not None and stats_path.exists():
                self.norm_stats = self._load_norm_stats(stats_path)
            else:
                self.norm_stats = self._compute_norm_stats()
                if stats_path is not None:
                    self._save_norm_stats(stats_path, self.norm_stats)
        else:
            self.norm_stats = norm_stats

        self._mean = self.norm_stats.get("mean")
        self._std = self.norm_stats.get("std")
        self._min = self.norm_stats.get("min")
        self._max = self.norm_stats.get("max")

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def __getitem__(self, idx: int) -> WalkerSample:
        source_idx, target_t = self.sample_indices[idx]
        source_idx = int(source_idx)
        target_t = int(target_t)
        source_name = self.source_names[source_idx]
        payload = self.source_payloads[source_idx]
        data = payload["data"]
        months = payload["months"]
        valid_mask = payload["valid_mask"]

        # 取历史窗口和目标月。copy=True 是为了后续 NaN 填充/归一化不会影响缓存。
        x_np = np.array(data[target_t - self.L : target_t], copy=True)
        y_rollout_np = np.array(data[target_t : target_t + self.target_steps], copy=True)

        x = torch.from_numpy(x_np).float()
        y_rollout = torch.from_numpy(y_rollout_np).float()

        x = self._normalize_tensor(x, source_idx)
        y_rollout = self._normalize_tensor(y_rollout, source_idx)
        y = y_rollout[:1]

        # 模型输入不允许有 NaN/Inf。无效位置由 valid_mask 告诉 loss。
        x = torch.nan_to_num(x, nan=FILL_VALUE, posinf=FILL_VALUE, neginf=FILL_VALUE)
        y_rollout = torch.nan_to_num(y_rollout, nan=FILL_VALUE, posinf=FILL_VALUE, neginf=FILL_VALUE)
        y = y_rollout[:1]

        sample: WalkerSample = {
            "x": x,
            "y": y,
            "target_month": int(months[target_t]),
            "valid_mask": valid_mask,
            "time_index": target_t,
            "source_index": source_idx,
            "source_id": source_name,
        }
        if self.target_steps > 1:
            sample["y_rollout"] = y_rollout
            sample["target_months"] = torch.from_numpy(
                np.array(months[target_t : target_t + self.target_steps], dtype=np.int64)
            )
        return sample

    def denormalize(self, y: torch.Tensor, source_index: Any = None) -> torch.Tensor:
        """把归一化空间中的张量还原到物理量单位。

        输入张量只要求最后三个维度是 ``(4, H, W)``，例如：
        ``(B, 1, 4, H, W)`` 或 ``(1, 4, H, W)``。
        当 ``norm_scope=source`` 时，source_index 必须是单个 source 编号，
        或与 batch 第一维对应的 ``(B,)`` 编号张量。
        """
        if self.norm == "none":
            return y
        if self.norm == "zscore":
            mean = self._view_stats(self._mean, y, source_index)
            std = self._view_stats(self._std, y, source_index)
            return y * std + mean
        if self.norm == "minmax":
            min_value = self._view_stats(self._min, y, source_index)
            max_value = self._view_stats(self._max, y, source_index)
            return y * (max_value - min_value) + min_value
        raise ValueError(f"Unsupported normalization method: {self.norm}")

    @classmethod
    def prepare_data_cache(cls, data_path: str | Path | None, config: dict[str, Any]) -> None:
        """预先生成完整数据缓存；DDP 下应由 rank 0 调用一次。

        该缓存不改变数据内容，只把四个 NetCDF 预先堆成连续的 ``.npy`` 文件，
        之后各 rank 可以用 mmap 快速打开同一份数组。
        """
        data_config = config.get("data", config)
        for source in cls._resolve_sources(data_path, data_config):
            source_config = cls._source_data_config(data_config, source.name)
            cache_paths = cls._data_cache_paths(source_config)
            if cache_paths is None or cls._data_cache_exists(cache_paths):
                continue

            print(f"[dataset] prepare cache for source {source.name}: {source.path}", flush=True)
            payload = cls._load_data_from_netcdf(source.path)
            cls._save_data_cache(cache_paths, payload)
            print(f"[dataset] cache ready for source {source.name}", flush=True)

    @staticmethod
    def _resolve_sources(data_path: str | Path | None, data_config: dict[str, Any]) -> list[SourceSpec]:
        """解析单 source 或多 source 配置。

        多 source 配置示例：

            sources:
              - name: CESM2
                path: data/cesm2
              - name: EC-Earth3
                path: data/ec-earth3
        """
        raw_sources = data_config.get("sources")
        if not raw_sources:
            raw_path = data_path or data_config["path"]
            path = Path(raw_path).expanduser().resolve()
            name = str(data_config.get("source_name") or path.name)
            return [SourceSpec(name=name, path=path)]

        sources: list[SourceSpec] = []
        for idx, item in enumerate(raw_sources):
            if isinstance(item, (str, Path)):
                path = Path(item).expanduser().resolve()
                name = path.name
            elif isinstance(item, dict):
                if "path" not in item:
                    raise ValueError(f"data.sources[{idx}] must contain path")
                path = Path(item["path"]).expanduser().resolve()
                name = str(item.get("name") or path.name)
            else:
                raise TypeError(f"data.sources[{idx}] must be a path string or mapping, got {type(item)!r}")

            if not name:
                raise ValueError(f"data.sources[{idx}] has empty name")
            sources.append(SourceSpec(name=name, path=path))
        return sources

    @staticmethod
    def _source_data_config(data_config: dict[str, Any], source_name: str) -> dict[str, Any]:
        """给每个 source 派生独立数据缓存路径，避免多个 source 覆盖同一组 .npy。"""
        source_config = dict(data_config)
        if data_config.get("sources") and data_config.get("data_cache_path"):
            prefix = Path(str(data_config["data_cache_path"])).expanduser()
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
            source_config["data_cache_path"] = str(prefix.parent / f"{prefix.name}_{safe_name}")
        return source_config

    @classmethod
    def _load_or_get_cache(cls, data_path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
        """读取数据目录；如果当前进程已经读过，就直接复用缓存对象。"""
        cache_paths = cls._data_cache_paths(data_config)
        cache_key = (data_path, str(cache_paths["prefix"]) if cache_paths is not None else "")
        if cache_key not in cls._data_cache:
            cls._data_cache[cache_key] = cls._load_data(data_path, data_config)
        return cls._data_cache[cache_key]

    @classmethod
    def _load_data(cls, data_path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
        """优先读取 .npy 数据缓存；没有缓存时回退到 NetCDF。"""
        cache_paths = cls._data_cache_paths(data_config)
        if cache_paths is not None and cls._data_cache_exists(cache_paths):
            print(f"[dataset] use data cache: {cache_paths['prefix']}", flush=True)
            return cls._load_data_cache(cache_paths)

        print(f"[dataset] build data cache from NetCDF: {data_path}", flush=True)
        payload = cls._load_data_from_netcdf(data_path)
        if cache_paths is not None:
            cls._save_data_cache(cache_paths, payload)
            print(f"[dataset] saved data cache: {cache_paths['prefix']}", flush=True)
        return payload

    @staticmethod
    def _load_data_from_netcdf(data_path: Path) -> dict[str, Any]:
        """读取四个 remap 后的 NetCDF 文件并堆成 (T, 4, H, W)。"""
        if not data_path.exists():
            raise FileNotFoundError(
                f"Remapped data directory not found: {data_path}\n"
                "Run: wsl -d Ubuntu-24.04 -- bash "
                "/path/to/WalkerNet/scripts/data/remap_to_1x1.sh"
            )

        arrays: list[np.ndarray] = []
        reference_lat: np.ndarray | None = None
        reference_lon: np.ndarray | None = None
        reference_years: np.ndarray | None = None
        reference_months: np.ndarray | None = None

        for variable in VARIABLES:
            path = data_path / REMAPPED_FILENAMES[variable]
            if not path.exists():
                raise FileNotFoundError(f"Missing remapped file for {variable}: {path}")

            print(f"[dataset] read {path}", flush=True)
            with xr.open_dataset(path, decode_times=False) as ds:
                if variable not in ds:
                    raise ValueError(f"{path} does not contain variable {variable!r}")

                # CDO 输出变量维度应为 (time, lat, lon)。这里显式 transpose，
                # 防止不同 NetCDF writer 导致维度顺序显示不一致。
                arr = ds[variable].transpose("time", "lat", "lon").values.astype(np.float32)
                if arr.shape[1:] != (GRID_H, GRID_W):
                    raise ValueError(f"{path}: expected spatial shape {(GRID_H, GRID_W)}, got {arr.shape[1:]}")
                arr[~np.isfinite(arr)] = np.nan

                lat = ds["lat"].values.astype(np.float64)
                lon = ds["lon"].values.astype(np.float64)

            years, months = WalkerDataset._read_year_month(path)

            # 第一个变量作为坐标/时间参考，后续变量必须完全对齐。
            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon
                reference_years = years
                reference_months = months
            else:
                if not np.allclose(lat, reference_lat):
                    raise ValueError(f"{path}: latitude coordinate does not match previous variables")
                if not np.allclose(lon, reference_lon):
                    raise ValueError(f"{path}: longitude coordinate does not match previous variables")
                if not np.array_equal(years, reference_years) or not np.array_equal(months, reference_months):
                    raise ValueError(f"{path}: year/month coordinate does not match previous variables")

            arrays.append(arr)

        # stack 后 data 形状为 (T, 4, H, W)，变量顺序由 VARIABLES 决定。
        data = np.stack(arrays, axis=1)

        # valid_mask 是空间 mask，不随时间变化。这里采用 any(axis=0)：
        # 只要某变量某格点在历史上至少有一次有效，就认为该格点可参与训练。
        # 具体 loss 仍可以进一步按 y 是否有效做更细粒度处理。
        valid_mask_np = np.isfinite(data).any(axis=0)
        valid_mask = torch.from_numpy(valid_mask_np.astype(np.bool_))

        return {
            "data": data,
            "years": reference_years,
            "months": reference_months,
            "lat": reference_lat,
            "lon": reference_lon,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _data_cache_paths(data_config: dict[str, Any]) -> dict[str, Path] | None:
        """根据 data_cache_path 生成一组缓存文件路径。"""
        value = data_config.get("data_cache_path")
        if not value:
            return None

        prefix = Path(value).expanduser()
        parent = prefix.parent
        name = prefix.name
        return {
            "prefix": prefix,
            "data": parent / f"{name}_data.npy",
            "years": parent / f"{name}_years.npy",
            "months": parent / f"{name}_months.npy",
            "lat": parent / f"{name}_lat.npy",
            "lon": parent / f"{name}_lon.npy",
            "valid_mask": parent / f"{name}_valid_mask.npy",
            "meta": parent / f"{name}_meta.pt",
        }

    @staticmethod
    def _data_cache_exists(paths: dict[str, Path]) -> bool:
        """检查完整数据缓存是否已经可用。"""
        required = ("data", "years", "months", "lat", "lon", "valid_mask", "meta")
        return all(paths[key].exists() for key in required)

    @staticmethod
    def _load_data_cache(paths: dict[str, Path]) -> dict[str, Any]:
        """用 mmap 读取已经堆好的完整数据缓存。"""
        meta = torch.load(paths["meta"], map_location="cpu")
        expected = {
            "variables": VARIABLES,
            "grid": (GRID_H, GRID_W),
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Data cache meta mismatch: {key}={meta.get(key)!r}, expected {value!r}")

        data = np.load(paths["data"], mmap_mode="r")
        years = np.load(paths["years"])
        months = np.load(paths["months"])
        lat = np.load(paths["lat"])
        lon = np.load(paths["lon"])
        valid_mask_np = np.load(paths["valid_mask"])

        if tuple(data.shape[1:]) != (len(VARIABLES), GRID_H, GRID_W):
            raise ValueError(f"Data cache shape mismatch: got {data.shape}")

        return {
            "data": data,
            "years": years,
            "months": months,
            "lat": lat,
            "lon": lon,
            "valid_mask": torch.from_numpy(valid_mask_np.astype(np.bool_, copy=False)),
        }

    @staticmethod
    def _save_data_cache(paths: dict[str, Path], payload: dict[str, Any]) -> None:
        """把完整数据缓存写到磁盘，供多卡训练 mmap 复用。"""
        paths["prefix"].parent.mkdir(parents=True, exist_ok=True)

        np.save(paths["data"], payload["data"])
        np.save(paths["years"], payload["years"])
        np.save(paths["months"], payload["months"])
        np.save(paths["lat"], payload["lat"])
        np.save(paths["lon"], payload["lon"])
        np.save(paths["valid_mask"], payload["valid_mask"].cpu().numpy())
        torch.save(
            {
                "variables": VARIABLES,
                "grid": (GRID_H, GRID_W),
                "shape": tuple(payload["data"].shape),
            },
            paths["meta"],
        )

    @staticmethod
    def _read_year_month(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """读取 NetCDF 的 time 坐标，返回每个时间步对应的 year/month。"""
        try:
            with xr.open_dataset(path, decode_times=True, use_cftime=True) as ds:
                times = ds["time"].values
                years = np.array([int(t.year) for t in times], dtype=np.int32)
                months = np.array([int(t.month) for t in times], dtype=np.int8)
                return years, months
        except Exception:
            from netCDF4 import num2date

            with xr.open_dataset(path, decode_times=False) as ds:
                time = ds["time"]
                units = time.attrs["units"]
                calendar = time.attrs.get("calendar", "standard")
                dates = num2date(time.values, units=units, calendar=calendar, only_use_cftime_datetimes=True)
                years = np.array([int(t.year) for t in dates], dtype=np.int32)
                months = np.array([int(t.month) for t in dates], dtype=np.int8)
                return years, months

    def _build_sample_indices(self, split: str) -> np.ndarray:
        """根据 split 年份范围构造所有可用 ``(source_idx, target_t)``。"""
        pieces = []
        for source_idx, payload in enumerate(self.source_payloads):
            target_indices = self._build_source_sample_indices(payload["years"], split)
            if target_indices.size == 0:
                continue
            source_column = np.full_like(target_indices, source_idx)
            pieces.append(np.stack([source_column, target_indices], axis=1))
        if not pieces:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(pieces, axis=0).astype(np.int64)

    def _build_source_sample_indices(self, years: np.ndarray, split: str) -> np.ndarray:
        """根据单个 source 的年份范围构造所有可用目标月 t。"""
        start_year, end_year = self.data_config[f"{split}_years"]
        target_in_split = (years >= int(start_year)) & (years <= int(end_year))

        # t 必须至少 >= L，否则没有足够历史窗口。
        # 多步训练时还要保证所有未来目标月都仍在当前 split 内，避免 val/test 泄漏。
        enough_history = np.arange(len(years)) >= self.L
        last_target = np.arange(len(years)) + self.target_steps - 1
        enough_future = last_target < len(years)
        future_in_split = np.zeros(len(years), dtype=bool)
        valid_future_positions = np.where(enough_future)[0]
        future_years = years[last_target[valid_future_positions]]
        future_in_split[valid_future_positions] = future_years <= int(end_year)
        return np.where(target_in_split & enough_history & enough_future & future_in_split)[0].astype(np.int64)

    def _compute_norm_stats(self) -> dict[str, torch.Tensor]:
        """只用训练年份计算全体或逐 source 的归一化统计量。"""
        if self.norm == "none":
            return {}

        train_start, train_end = self.data_config["train_years"]
        has_training_month = any(
            np.any((payload["years"] >= int(train_start)) & (payload["years"] <= int(train_end)))
            for payload in self.source_payloads
        )
        if not has_training_month:
            raise ValueError(f"No training months found for train_years={self.data_config['train_years']}")

        # pooled 兼容已训练模型，shape=(4,)；source 为每个物理模式独立统计，
        # shape=(S, 4)，避免不同模式的气候均值和振幅互相污染。
        payload_groups = (
            [[payload] for payload in self.source_payloads]
            if self.norm_scope == "source"
            else [self.source_payloads]
        )

        stats: dict[str, torch.Tensor] = {}
        if self.norm == "zscore":
            mean = np.empty((len(payload_groups), len(VARIABLES)), dtype=np.float32)
            std = np.empty_like(mean)
            for group_idx, payloads in enumerate(payload_groups):
                for variable_idx, variable in enumerate(VARIABLES):
                    total = 0.0
                    total_square = 0.0
                    count = 0
                    for payload in payloads:
                        train_mask = (payload["years"] >= int(train_start)) & (
                            payload["years"] <= int(train_end)
                        )
                        values = np.asarray(payload["data"][train_mask, variable_idx], dtype=np.float64)
                        finite = np.isfinite(values)
                        if not np.any(finite):
                            continue
                        valid_values = values[finite]
                        total += float(valid_values.sum(dtype=np.float64))
                        total_square += float((valid_values * valid_values).sum(dtype=np.float64))
                        count += int(valid_values.size)
                    if count == 0:
                        label = self.source_names[group_idx] if self.norm_scope == "source" else "pooled"
                        raise ValueError(f"No finite training values found for source={label}, variable={variable}")
                    mean_value = total / count
                    variance = max(total_square / count - mean_value * mean_value, 0.0)
                    mean[group_idx, variable_idx] = mean_value
                    std[group_idx, variable_idx] = float(np.sqrt(variance))
                    if not np.isfinite(std[group_idx, variable_idx]) or std[group_idx, variable_idx] == 0:
                        std[group_idx, variable_idx] = 1.0
            if self.norm_scope == "pooled":
                mean = mean[0]
                std = std[0]
            stats["mean"] = torch.from_numpy(mean)
            stats["std"] = torch.from_numpy(std)
            return stats

        if self.norm == "minmax":
            min_value = np.empty((len(payload_groups), len(VARIABLES)), dtype=np.float32)
            max_value = np.empty_like(min_value)
            for group_idx, payloads in enumerate(payload_groups):
                for variable_idx, variable in enumerate(VARIABLES):
                    group_mins = []
                    group_maxs = []
                    for payload in payloads:
                        train_mask = (payload["years"] >= int(train_start)) & (
                            payload["years"] <= int(train_end)
                        )
                        values = payload["data"][train_mask, variable_idx]
                        if np.isfinite(values).any():
                            group_mins.append(np.nanmin(values))
                            group_maxs.append(np.nanmax(values))
                    if not group_mins:
                        label = self.source_names[group_idx] if self.norm_scope == "source" else "pooled"
                        raise ValueError(f"No finite training values found for source={label}, variable={variable}")
                    min_value[group_idx, variable_idx] = np.nanmin(np.asarray(group_mins, dtype=np.float32))
                    max_value[group_idx, variable_idx] = np.nanmax(np.asarray(group_maxs, dtype=np.float32))
                    span = max_value[group_idx, variable_idx] - min_value[group_idx, variable_idx]
                    if not np.isfinite(span) or span == 0:
                        max_value[group_idx, variable_idx] = min_value[group_idx, variable_idx] + 1.0
            if self.norm_scope == "pooled":
                min_value = min_value[0]
                max_value = max_value[0]
            stats["min"] = torch.from_numpy(min_value)
            stats["max"] = torch.from_numpy(max_value)
            return stats

        raise ValueError(f"Unsupported normalization method: {self.norm}")

    def _norm_stats_path(self) -> Path | None:
        """返回归一化统计缓存路径；未配置则返回 None。"""
        value = self.data_config.get("norm_stats_path")
        if not value:
            return None
        return Path(value).expanduser()

    def _load_norm_stats(self, path: Path) -> dict[str, torch.Tensor]:
        """从磁盘读取归一化统计，并做轻量一致性检查。"""
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "stats" not in payload:
            raise ValueError(f"Invalid norm stats file: {path}")

        meta = dict(payload.get("meta", {}))
        expected_meta = {
            "norm": self.norm,
            "variables": VARIABLES,
            "train_years": tuple(self.data_config["train_years"]),
            "source_names": self.source_names,
        }
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"Norm stats meta mismatch for {path}: {key}={meta.get(key)!r}, expected {expected!r}"
                )
        cached_scope = str(meta.get("norm_scope", "pooled"))
        if cached_scope != self.norm_scope:
            raise ValueError(
                f"Norm stats meta mismatch for {path}: norm_scope={cached_scope!r}, "
                f"expected {self.norm_scope!r}"
            )

        stats = payload["stats"]
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid norm stats payload: {path}")
        result = {key: value.detach().cpu() for key, value in stats.items()}
        expected_shape = (len(self.sources), len(VARIABLES)) if self.norm_scope == "source" else (len(VARIABLES),)
        for name, value in result.items():
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Norm stats shape mismatch for {path}: {name}={tuple(value.shape)}, expected {expected_shape}"
                )
        return result

    def _save_norm_stats(self, path: Path, stats: dict[str, torch.Tensor]) -> None:
        """把训练集归一化统计写入磁盘，供验证集和 DDP 其它 rank 复用。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "norm": self.norm,
                "variables": VARIABLES,
                "train_years": tuple(self.data_config["train_years"]),
                "source_names": self.source_names,
                "norm_scope": self.norm_scope,
            },
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
        }
        torch.save(payload, path)

    def _normalize_tensor(self, tensor: torch.Tensor, source_index: Any = None) -> torch.Tensor:
        """按变量维度广播归一化统计量。"""
        if self.norm == "none":
            return tensor
        if self.norm == "zscore":
            mean = self._view_stats(self._mean, tensor, source_index)
            std = self._view_stats(self._std, tensor, source_index)
            return (tensor - mean) / std
        if self.norm == "minmax":
            min_value = self._view_stats(self._min, tensor, source_index)
            max_value = self._view_stats(self._max, tensor, source_index)
            return (tensor - min_value) / (max_value - min_value)
        raise ValueError(f"Unsupported normalization method: {self.norm}")

    @staticmethod
    def _view_stats(stats: torch.Tensor | None, target: torch.Tensor, source_index: Any = None) -> torch.Tensor:
        """选择对应 source，并 reshape 成可广播到 target 的形状。"""
        if stats is None:
            raise ValueError("Normalization stats are missing")

        stats = stats.to(device=target.device, dtype=target.dtype)
        if stats.ndim == 2:
            if source_index is None:
                if stats.shape[0] != 1:
                    raise ValueError("source_index is required when norm_scope='source'")
                stats = stats[0]
            else:
                indices = torch.as_tensor(source_index, dtype=torch.long, device=target.device)
                if indices.ndim == 0:
                    stats = stats[indices]
                elif indices.ndim == 1:
                    if target.ndim < 4 or indices.numel() != target.shape[0]:
                        raise ValueError(
                            f"Batched source_index shape {tuple(indices.shape)} does not match target {tuple(target.shape)}"
                        )
                    selected = stats.index_select(0, indices)
                    shape = [target.shape[0]] + [1] * (target.ndim - 1)
                    shape[-3] = len(VARIABLES)
                    return selected.view(*shape)
                else:
                    raise ValueError(f"source_index must be scalar or 1D, got shape {tuple(indices.shape)}")

        if stats.ndim != 1 or stats.shape[0] != len(VARIABLES):
            raise ValueError(f"Invalid normalization stats shape: {tuple(stats.shape)}")
        shape = [1] * target.ndim
        shape[-3] = len(VARIABLES)
        return stats.view(*shape)
