"""SSP remap 输入时间轴检查测试。"""

from pathlib import Path

import pytest

from scripts.data.remap_cmip6_ssp_to_1x1 import VariableJob, validate_input_timeline


def test_validate_input_timeline_accepts_complete_split_files():
    job = VariableJob(
        scenario="ssp245",
        source_id="CESM2",
        variable="tos",
        files=(
            Path("tos_Omon_CESM2_ssp245_r1i1p1f1_gn_201501-206412.nc"),
            Path("tos_Omon_CESM2_ssp245_r1i1p1f1_gn_206501-210012.nc"),
        ),
    )
    validate_input_timeline(job)


def test_validate_input_timeline_rejects_missing_year():
    job = VariableJob(
        scenario="ssp245",
        source_id="EC-Earth3",
        variable="tauu",
        files=(
            Path("tauu_Amon_EC-Earth3_ssp245_r10i1p1f1_gr_201501-202012.nc"),
            Path("tauu_Amon_EC-Earth3_ssp245_r10i1p1f1_gr_202201-210012.nc"),
        ),
    )
    with pytest.raises(ValueError, match="gap or overlap"):
        validate_input_timeline(job)
