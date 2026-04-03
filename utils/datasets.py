import torch
import numpy as np
import uproot
from torch.utils.data import DataLoader
from typing import Iterator, Hashable
#from uprootdataset import UprootMultiFileDataset
from utils.uprootdataset import UprootMultiFileDataset

class PosEnergyRecoDataset(UprootMultiFileDataset):
    def __init__(
        self,
        truth_expressions: list = [],
        seed=74819,
        time_jitter: float = 2.0,
        *args,
        **kwargs,
    ):
        # 1. Define the branches you want to extract
        self.hit_expressions = ["hitids", "hittimes"]
        self.reco_expressions = ["posx", "posy", "posz", "posz_av", "posr_av", "fitValid", "evIndex"]
        
        # 2. Ensure base class loads these branches
        all_needed = set(self.hit_expressions + self.reco_expressions + truth_expressions)
        kwargs.setdefault('expressions', list(all_needed))
        
        super().__init__(*args, **kwargs)

        self.seed = seed
        self.truth_expressions = truth_expressions
        # add jitter time to the mc truth 
        self.time_jitter = time_jitter

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        # Your base class yields (Record, filename)
        for item in super().__iter__():
            # Step 1: Unpack the record from the tuple
            record = item[0] 

            # Step 2: SNO+ Bronze Cuts (Manual Filtering)
            # Only process if fit is valid and it's the primary trigger
            #if record["fitValid"] != 1 or record["evIndex"] != 0:
            if record["evIndex"] != 0:
                continue
            # rule out events if having invalid hittime 
            hit_times_np = record["hittimes"].to_numpy().astype(np.float32)
            if(len(hit_times_np) == 0 or np.any(hit_times_np) < 0):
                continue

            # Step 3: Extract Vector Data (Variable Length)
            # Use to_numpy() to escape Awkward Record types
            hit_ids_np = record["hitids"].to_numpy().astype(np.int64)

            

            # Step 4: Convert everything to Tensors (No Padding)
            inputs = {
                "hit_ids": torch.from_numpy(hit_ids_np),
                "hit_times": torch.from_numpy(hit_times_np),
                "n_hits": torch.tensor(len(hit_ids_np), dtype=torch.long),
                # Scalar Reco values
                "scintfitposition": torch.tensor([float(record["posx"]),float(record["posy"]),float(record["posz"])], dtype=torch.float32),
                "posr_av": torch.tensor(float(record["posr_av"]), dtype=torch.float32),
            }

            # Step 5: Unpack Truth Expressions
            mctime = float(record["mctime1"])
            
            # --- ADD NOISE ---
            if self.time_jitter > 0:
                # np.random.normal(mean, std)
                mctime += np.random.normal(0, self.time_jitter)
            truth = {
                "position": torch.tensor([
                    float(record["mcPosx"]), 
                    float(record["mcPosy"]), 
                    float(record["mcPosz"])
                ], dtype=torch.float32),
                "energy": torch.tensor(float(record["mcEdepQuenched"]), dtype=torch.float32),
                "evtime": torch.tensor(mctime, dtype=torch.float32)
            }

            # Final yield: Two clean dictionaries of PyTorch Tensors
            #print(inputs, truth)
            yield inputs, truth

def collate_variable(batch):
    """
    batch is [(inputs_1, truth_1), (inputs_2, truth_2), ...]
    Standard PyTorch DataLoader
    """

    first_inputs, first_truth = batch[0]
    
    final_inputs = {}
    for key in first_inputs.keys():
        if key in ["hit_ids", "hit_times"]:
            # Keep variable length hits as a LIST of tensors
            final_inputs[key] = [sample[0][key] for sample in batch]
        else:
            # Scalars can be safely stacked into a batch tensor
            final_inputs[key] = torch.stack([sample[0][key] for sample in batch])
            
    final_truth = {
        key: torch.stack([sample[1][key] for sample in batch]) 
        for key in first_truth.keys()
    }
    #print(final_inputs, final_truth)
    return final_inputs, final_truth

def collate_varlen(batch):
    """
    batch is a list of tuples: [(inputs_dict, truth_dict), ...]
    """
    # 1. Initialize lists to hold the raw tensors for each key
    id_list = []
    time_list = []
    truth_list = []
    lengths = []

    for inputs, truth in batch:
        # We keep them separate for now
        id_list.append(inputs['hit_ids'])
        time_list.append(inputs['hit_times'])
        
        truth_list.append(truth)
        lengths.append(inputs['hit_ids'].size(0))

    # 2. Concatenate the hits into "Long Tensors" (Total_Hits,)
    # This is the 'Packed' or 'Flattened' format for Varlen Attention
    x_packed = {
        "hit_ids": torch.cat(id_list, dim=0).long(),    # Long for Embedding
        "hit_times": torch.cat(time_list, dim=0).float() # Float for Math
    }
    
    # 3. Handle Metadata (Varlen sequence markers)
    lengths_t = torch.tensor(lengths, dtype=torch.int32)
    max_len = int(lengths_t.max().item())
    cu_seq = torch.zeros(len(lengths_t) + 1, dtype=torch.int32)
    cu_seq[1:] = torch.cumsum(lengths_t, dim=0)

    # 4. Stack Truth (Dict of Batched Tensors)
    final_truth = {
        key: torch.stack([t[key] for t in truth_list]) 
        for key in truth_list[0].keys()
    }
    #print("x_packed ",x_packed)
    return x_packed, final_truth, cu_seq, max_len

def main():
    # Configuration
    data_path = "/data/snoplus2/weiiiii/electron_bismsb_0dot5_10/Ntuple/Bronze/37121*.ntuple.root"
    truth_vars = ["mcPosx", "mcPosy", "mcPosz", "mcPosr", "mcEdepQuenched","mctime1"]

    # 1. Initialize Dataset
    dataset = PosEnergyRecoDataset(
        file_paths=data_path,
        tree_name="output",
        truth_expressions=truth_vars,
        buffer_size=50,
        debug=False
    )

    # 2. Setup DataLoader with the variable collator
    # Note: batch_size=4 to show that it handles different event lengths
    # 4 events at one batch
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=1, 
        collate_fn=collate_varlen
    )

    print(f"{'Batch':<6} | {'Event lengths in batch':<30} | {'Avg Energy'}")
    print("-" * 65)
    """
    # 3. Iteration Loop
    for i, (inputs, truth) in enumerate(dataloader):
        if i >= 5: break # Only show first 5 batches

        # In variable length, inputs['hit_ids'] is a LIST of 4 tensors
        lengths = [len(h) for h in inputs["hit_ids"]]
        avg_energy = torch.mean(truth["mcEdepQuenched"]).item()

        print(f"{i:<6} | {str(lengths):<30} | {avg_energy:<8.2f} MeV")
        
        # Example: How you would access the first event in the batch
        first_event_z = inputs["posz"][0].item()
        # print(f"  -> First event in batch Reco Z: {first_event_z:.1f}")

    print("\nProcessing complete.")
    """
    for i, (x_packed, truth, cu_seq, max_len) in enumerate(dataloader):
        if i >= 5: break 

        # x_packed is now a single large matrix of all hits in the 125 events
        total_hits = x_packed["hit_times"]
        avg_energy = torch.mean(truth["mcEdepQuenched"]).item()
        print(f"{i:<6} | {str(len(total_hits)):<30} | {avg_energy:<8.2f} MeV")
        #print(f"Batch {i}")
        #print(f"  Total Hits: {total_hits}")
        #print(f"  Max Event Length: {max_len}")
        #print(f"  Avg Energy: {avg_energy:.2f} MeV")
        #print(f"  cu_seq Map: {cu_seq[:5]}... (first 5 indices)")

if __name__ == "__main__":
    main()