from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import shared_memory

# ==============================================================================
# Abstract Base Class for 2D Snapshot Data (use at some point)
# ==============================================================================

class Snapshot2DBase(ABC):
    """
    Abstract base class for 2D simulation snapshots.
    """

    # Required abstract methods / attributes
    @property
    @abstractmethod
    def bbox(self):
        pass

    @property
    @abstractmethod
    def cells_per_block_x(self):
        pass

    @property
    @abstractmethod
    def cells_per_block_y(self):
        pass

    @property
    @abstractmethod
    def simtime(self):
        pass

    @property
    @abstractmethod
    def td_vars(self):
        pass

    @abstractmethod
    def cellCoords(self, blockID, with_edges=False):
        pass

    @abstractmethod
    def cellVolumes(self, blockID):
        pass

    # --- Shared memory methods implemented here ---
    @classmethod
    def from_shm(cls, shm_registry_entry):
        """
        Reconstruct a Snapshot2D-like object from shared memory buffers
        created by create_shm_for_snapshot_generic().
        Keeps SharedMemory handles alive to prevent premature release.
        """
        self = object.__new__(cls)  # bypass __init__

        def parse_dtype(dtype_meta):
            if isinstance(dtype_meta, str):
                try:
                    return np.dtype(eval(dtype_meta))
                except Exception:
                    return np.dtype(dtype_meta)
            else:
                return np.dtype(dtype_meta)

        # Store SharedMemory handles to prevent GC
        self._shm_handles = []

        # Iterate over all attributes in the SHM metadata
        for attr_name, attr_meta in shm_registry_entry.items():
            if attr_name == "td_vars":
                continue  # handle separately

            if isinstance(attr_meta, dict) and "name" in attr_meta:
                shm = shared_memory.SharedMemory(name=attr_meta["name"])
                self._shm_handles.append(shm)
                arr = np.ndarray(attr_meta["shape"], parse_dtype(attr_meta["dtype"]), buffer=shm.buf)
                setattr(self, attr_name, arr)
            else:
                # scalar or metadata-only
                setattr(self, attr_name, attr_meta.get("value", None))

        # Handle td_vars separately
        self.td_vars = {}
        td_vars_meta = shm_registry_entry.get("td_vars", {})
        for key, meta in td_vars_meta.items():
            shm = shared_memory.SharedMemory(name=meta["name"])
            self._shm_handles.append(shm)
            self.td_vars[key] = np.ndarray(meta["shape"], parse_dtype(meta["dtype"]), buffer=shm.buf)

        return self

    def close_shm(self):
        """Close all attached shared memory blocks."""
        if hasattr(self, "_shm_handles"):
            for shm in self._shm_handles:
                try:
                    shm.close()
                except Exception:
                    pass
            self._shm_handles = []



# Child class
class Snapshot2DFLASH(Snapshot2DBase):
    """
    Concrete implementation of Snapshot2DBase for FLASH 2D outputs.
    """

    def __init__(self, path, keys, only_leafs=True):
        # Implement reading HDF5 snapshot and initialize attributes
        self.bbox = ...  # numpy array
        self.cells_per_block_x = ...
        self.cells_per_block_y = ...
        self.simtime = ...
        self.td_vars = {...}  # dict of variables

    def cellCoords(self, blockID, with_edges=False):
        # implement method
        ...

    # Implement all other abstract methods from Snapshot2DBase...