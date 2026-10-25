import logging
import os
from typing import Literal, Optional, Union

import spatialdata as sd
from anndata import AnnData
from spatialdata_io import xenium


def read_Xenium(
    file_path: str,
    zarr_prefix: Optional[str] = None,
    return_type: Literal["sdata", "anndata"] = "sdata",
) -> Union[sd.SpatialData, AnnData]:
    if return_type not in ("sdata", "anndata"):
        raise ValueError("return_type must be either 'sdata' or 'anndata'.")

    zarr_path = os.path.join(file_path, f"{zarr_prefix}.zarr") if zarr_prefix else None

    if zarr_path and os.path.exists(zarr_path):
        logging.info(f"Loading cached Zarr: {zarr_path}")
        sdata = sd.read_zarr(zarr_path)
    else:
        logging.info(f"Reading raw Xenium data: {file_path}")
        sdata = xenium(file_path)

        if zarr_path:
            logging.info(f"Creating Zarr cache at: {zarr_path}")
            sdata.write(zarr_path)

    if return_type == "sdata":
        return sdata

    if "table" in sdata.tables:
        return sdata.tables["table"].copy()

    raise KeyError("The 'table' element was not found in the SpatialData object.")
