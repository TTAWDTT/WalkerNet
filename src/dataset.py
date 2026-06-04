"""WalkerNet 的数据集实现。

WalkerDataset 只读取已经由 CDO 重网格后的 ``data_1x1`` 文件，不负责原始
NetCDF 的经纬度映射。重网格请先运行 ``scripts/remap_to_1x1.sh``。

返回的样本遵守 ``src.interfaces`` 中的最小共享接口：

    x:            (L, 4, 180, 360), float32
    y:            (1, 4, 180, 360), float32
    target_month: int in [1, 12]
    valid_mask:   (4, 180, 360), bool

设计原则：
- 预处理和训练读取分离：CDO remap 是一次性脚本，Dataset 只读结果。
- 模型输入不能包含 NaN/Inf：缺测区域用 valid_mask 标记，张量中填 0。
- 归一化统计只从训练年份计算，验证/测试复用训练统计。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        self.norm = str(self.data_config.get("norm", "zscore")).lower()

        if self.L < 1:
            raise ValueError(f"Input window L must be >= 1, got {self.L}")

        # Dataset 和模型必须使用 interfaces.py 中固定的变量顺序。
        expected_variables = tuple(self.data_config.get("variables", VARIABLES))
        if expected_variables != VARIABLES:
            raise ValueError(f"config variables must be {VARIABLES}, got {expected_variables}")

        h = int(self.data_config.get("H", GRID_H))
        w = int(self.data_config.get("W", GRID_W))
        if (h, w) != (GRID_H, GRID_W):
            raise ValueError(f"config grid must be {(GRID_H, GRID_W)}, got {(h, w)}")

        # ---- 读取共享数据缓存 ----
        self.data_path = Path(data_path or self.data_config["path"]).expanduser().resolve()
        cache = self._load_or_get_cache(self.data_path, self.data_config)

        self.data = cache["data"]
        self.years = cache["years"]
        self.months = cache["months"]
        self.lat = cache["lat"]
        self.lon = cache["lon"]
        self.valid_mask = cache["valid_mask"]

        # sample_indices 保存的是目标月 t，而不是输入窗口起点。
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
        target_t = int(self.sample_indices[idx])

        # 取历史窗口和目标月。copy=True 是为了后续 NaN 填充/归一化不会影响缓存。
        x_np = np.array(self.data[target_t - self.L : target_t], copy=True)
        y_np = np.array(self.data[target_t : target_t + 1], copy=True)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        x = self._normalize_tensor(x)
        y = self._normalize_tensor(y)

        # 模型输入不允许有 NaN/Inf。无效位置由 valid_mask 告诉 loss。
        x = torch.nan_to_num(x, nan=FILL_VALUE, posinf=FILL_VALUE, neginf=FILL_VALUE)
        y = torch.nan_to_num(y, nan=FILL_VALUE, posinf=FILL_VALUE, neginf=FILL_VALUE)

        return {
            "x": x,
            "y": y,
            "target_month": int(self.months[target_t]),
            "valid_mask": self.valid_mask,
            "time_index": target_t,
        }

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """把归一化空间中的张量还原到物理量单位。

        输入张量只要求最后三个维度是 ``(4, H, W)``，例如：
        ``(B, 1, 4, H, W)`` 或 ``(1, 4, H, W)``。
        """
        if self.norm == "none":
            return y
        if self.norm == "zscore":
            mean = self._view_stats(self._mean, y)
            std = self._view_stats(self._std, y)
            return y * std + mean
        if self.norm == "minmax":
            min_value = self._view_stats(self._min, y)
            max_value = self._view_stats(self._max, y)
            return y * (max_value - min_value) + min_value
        raise ValueError(f"Unsupported normalization method: {self.norm}")

    @classmethod
    def prepare_data_cache(cls, data_path: str | Path | None, config: dict[str, Any]) -> None:
        """预先生成完整数据缓存；DDP 下应由 rank 0 调用一次。

        该缓存不改变数据内容，只把四个 NetCDF 预先堆成连续的 ``.npy`` 文件，
        之后各 rank 可以用 mmap 快速打开同一份数组。
        """
        data_config = config.get("data", config)
        cache_paths = cls._data_cache_paths(data_config)
        if cache_paths is None or cls._data_cache_exists(cache_paths):
            return

        resolved_data_path = Path(data_path or data_config["path"]).expanduser().resolve()
        payload = cls._load_data_from_netcdf(resolved_data_path)
        cls._save_data_cache(cache_paths, payload)

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
            return cls._load_data_cache(cache_paths)

        payload = cls._load_data_from_netcdf(data_path)
        if cache_paths is not None:
            cls._save_data_cache(cache_paths, payload)
        return payload

    @staticmethod
    def _load_data_from_netcdf(data_path: Path) -> dict[str, Any]:
        """读取四个 remap 后的 NetCDF 文件并堆成 (T, 4, H, W)。"""
        if not data_path.exists():
            raise FileNotFoundError(
                f"Remapped data directory not found: {data_path}\n"
                "Run: wsl -d Ubuntu-24.04 -- bash /mnt/d/Github/WalkerNet/scripts/remap_to_1x1.sh"
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
        """根据 split 年份范围构造所有可用目标月 t。"""
        start_year, end_year = self.data_config[f"{split}_years"]
        target_in_split = (self.years >= int(start_year)) & (self.years <= int(end_year))

        # t 必须至少 >= L，否则没有足够历史窗口。
        enough_history = np.arange(len(self.years)) >= self.L
        return np.where(target_in_split & enough_history)[0].astype(np.int64)

    def _compute_norm_stats(self) -> dict[str, torch.Tensor]:
        """只用训练年份计算归一化统计量。"""
        if self.norm == "none":
            return {}

        train_start, train_end = self.data_config["train_years"]
        train_mask = (self.years >= int(train_start)) & (self.years <= int(train_end))
        if not np.any(train_mask):
            raise ValueError(f"No training months found for train_years={self.data_config['train_years']}")

        stats: dict[str, torch.Tensor] = {}
        if self.norm == "zscore":
            mean = np.empty((len(VARIABLES),), dtype=np.float32)
            std = np.empty((len(VARIABLES),), dtype=np.float32)
            for idx in range(len(VARIABLES)):
                # nanmean/nanstd 会忽略陆地或缺测点。
                values = self.data[train_mask, idx]
                mean[idx] = np.nanmean(values, dtype=np.float64)
                std[idx] = np.nanstd(values, dtype=np.float64)
                if not np.isfinite(std[idx]) or std[idx] == 0:
                    std[idx] = 1.0
            stats["mean"] = torch.from_numpy(mean)
            stats["std"] = torch.from_numpy(std)
            return stats

        if self.norm == "minmax":
            min_value = np.empty((len(VARIABLES),), dtype=np.float32)
            max_value = np.empty((len(VARIABLES),), dtype=np.float32)
            for idx in range(len(VARIABLES)):
                # minmax 同样忽略 NaN，只统计有效物理值范围。
                values = self.data[train_mask, idx]
                min_value[idx] = np.nanmin(values)
                max_value[idx] = np.nanmax(values)
                if not np.isfinite(max_value[idx] - min_value[idx]) or max_value[idx] == min_value[idx]:
                    max_value[idx] = min_value[idx] + 1.0
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
        }
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"Norm stats meta mismatch for {path}: {key}={meta.get(key)!r}, expected {expected!r}"
                )

        stats = payload["stats"]
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid norm stats payload: {path}")
        return {key: value.detach().cpu() for key, value in stats.items()}

    def _save_norm_stats(self, path: Path, stats: dict[str, torch.Tensor]) -> None:
        """把训练集归一化统计写入磁盘，供验证集和 DDP 其它 rank 复用。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "norm": self.norm,
                "variables": VARIABLES,
                "train_years": tuple(self.data_config["train_years"]),
            },
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
        }
        torch.save(payload, path)

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """按变量维度广播归一化统计量。"""
        if self.norm == "none":
            return tensor
        if self.norm == "zscore":
            mean = self._view_stats(self._mean, tensor)
            std = self._view_stats(self._std, tensor)
            return (tensor - mean) / std
        if self.norm == "minmax":
            min_value = self._view_stats(self._min, tensor)
            max_value = self._view_stats(self._max, tensor)
            return (tensor - min_value) / (max_value - min_value)
        raise ValueError(f"Unsupported normalization method: {self.norm}")

    @staticmethod
    def _view_stats(stats: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        """把 (4,) 的统计量 reshape 成可广播到 target 的形状。"""
        if stats is None:
            raise ValueError("Normalization stats are missing")
        shape = [1] * target.ndim
        shape[-3] = len(VARIABLES)
        return stats.to(device=target.device, dtype=target.dtype).view(*shape)
