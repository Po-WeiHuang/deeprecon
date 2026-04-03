import glob                    
import fsspec                  
import numpy as np            
import uproot                  
import torch                  
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Iterable, Union, Hashable, Any
import os 
import random
class UprootMultiFileDataset(IterableDataset):
    def __init__(
        self,
        file_paths: str | Iterable[str],
        tree_name: str,
        expressions: Iterable[str] | None = None,
        cut: str | None = None,
        seed: int = 42,
        buffer_size: int = 10000,
        cache: bool = False,
        debug: bool = False,
    ) -> None:
        if expressions is None:
            expressions = set()
        if isinstance(file_paths, str):
            self.file_paths = glob.glob(file_paths)
        else:
            self.file_paths = file_paths
        self.tree_name = tree_name
        self.expressions = set(expressions)
        self.cut = cut
        self.seed = seed
        self.buffer_size = buffer_size
        self.generator = None
        self.debug = debug
        self.cache = cache
        self.file_paths = list(self.file_paths)
        random.shuffle(self.file_paths)
        if self.debug:
            self.debug_print("Initializing")

        self.events_yielded = 0

        self.length = None
        if self.cache:
            self.condor_scratch = os.environ.get("_CONDOR_SCRATCH_DIR", ".")
            self.cache_path = os.path.join(self.condor_scratch, "fsspec_cache")
            # Create the specific cache folder if it doesn't exist
            os.makedirs(self.cache_path, exist_ok=True)

    def __len__(self) -> int:
        if self.length is None:
            for path in self.file_paths:
                with uproot.open(path)[self.tree_name] as tree:
                    self.length += tree.num_entries
        return self.length

    def debug_print(self, msg: str) -> None:
        if self.debug:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                n_workers = 1
            else:
                worker_id = worker_info.id
                n_workers = worker_info.num_workers
            print(f"Worker {worker_id + 1} of {n_workers}: {msg}")

    def __iter__(self):
        file_paths = (
            glob.glob(self.file_paths)
            if isinstance(self.file_paths, str)
            else self.file_paths
        )
        if len(file_paths) == 0:
            raise ValueError("No files found!")
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers

        if self.debug:
            print(f"Worker {worker_id + 1} of {n_workers} starting.")

        self.seed += worker_id

        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)

        file_slice = slice(
            len(file_paths) * worker_id // n_workers,
            len(file_paths) * (worker_id + 1) // n_workers,
        )

        file_paths = file_paths[file_slice]
        if len(file_paths) == 0:
            raise ValueError("No files assigned to this worker!")

        shuffled_file_indices = self.generator.permutation(len(file_paths))

        buffer = []
        self.n_entries = 0

        if self.cache:
            fs = fsspec.filesystem(
                "simplecache", target_protocol="file", cache_storage=self.cache_path
            )
            open_context = fs.open
        else:
            open_context = open

        for file_index in shuffled_file_indices:
            file = file_paths[file_index]

            with open_context(file, mode="rb") as f:
                with uproot.open(f) as ntuple:
                    arrays = ntuple[self.tree_name].arrays(self.expressions)

            empty = True
            for entry in arrays:
                empty = False
                entry = (entry, file)
                if len(buffer) < self.buffer_size:
                    buffer.append(entry)
                    continue
                elif len(buffer) > self.buffer_size:
                    raise ValueError(
                        f"Buffer size exceeded! Size is {len(buffer)} but should be {self.buffer_size}."
                    )

                buffer_i = self.generator.choice(self.buffer_size)
                if len(buffer) < 1:
                    raise ValueError("Buffer is empty!")
                if len(buffer) != self.buffer_size:
                    raise ValueError(
                        f"Buffer is not full!. Size is {len(buffer)} but should be {self.buffer_size}."
                    )

                self.debug_print(f"Yielding event {self.n_entries}")

                self.n_entries += 1
                #print("buffer[buffer_i] ",buffer[buffer_i])
                yield buffer[buffer_i]

                buffer[buffer_i] = entry

            if empty:
                raise ValueError(f"No entries found in file {file}!")

        self.debug_print("Flushing buffer")
        # Flush out the rest of the buffer
        permutations = self.generator.permutation(len(buffer))
        for buffer_i in permutations:
            yield buffer[buffer_i]

