from pathlib import Path
from functools import wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from audiolm_pytorch.audiolm_pytorch import batch_unique_consecutive

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]

# dataset functions

class SemanticDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.max_semantic_length = int(max_length / seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        # replace the '.wav' extension with '.semantic.pt'
        
        semantic_file = Path(str(file).replace('.wav', '.semantic.pt'))
        data = torch.load(semantic_file)
        
        # take a random crop of max_semantic_length
        
        start_index = torch.randint(0, max(1, data.size(1) - self.max_semantic_length), (1, ))
        data = data[:, start_index:(start_index + self.max_semantic_length)]
        
        # unique the semantic output
        
        data = batch_unique_consecutive(data)
        semantic_token_length = data.size(1)

        # pad or curtail
        max_length = self.max_semantic_length
        
        if exists(max_length):
            if semantic_token_length > max_length:
                max_start = semantic_token_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]
            else:
                data = F.pad(data, (0, max_length - semantic_token_length), 'constant')

        data = rearrange(data, '1 ... -> ...')
        
        return data
    

class SemanticPhonemeDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.max_downsampled_length = int(max_length / seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        semantic_file = file.with_suffix('.semantic.pt')
        semantic_token_ids = torch.load(semantic_file)
        
        phonemes_file = file.with_suffix('.phonemes.pt')
        phoneme_token_ids = torch.load(phonemes_file).unsqueeze(0)
        
        print(f"semantic_token_ids: {semantic_token_ids.size()}, phoneme_token_ids: {phoneme_token_ids.size()}")
        assert semantic_token_ids.size(1) == phoneme_token_ids.size(1), f'semantic and phoneme token ids must have the same length: {semantic_token_ids.size(1)}, {phoneme_token_ids.size(1)}'
        
        # take a random crop of max_downsampled_length
        max_semantic_length = min(self.max_downsampled_length, semantic_token_ids.size(1))
        max_phoneme_length = min(self.max_downsampled_length, phoneme_token_ids.size(1))
        
        if semantic_token_ids.size(1) > max_semantic_length:
            start = torch.randint(0, semantic_token_ids.size(1), (1, ))
            semantic_token_ids = semantic_token_ids[:, start:(start + max_semantic_length)]
            phoneme_token_ids = phoneme_token_ids[:, start:(start + max_phoneme_length)]
        
        # unique the token ids
        
        semantic_token_ids = batch_unique_consecutive(semantic_token_ids)
        phoneme_token_ids = batch_unique_consecutive(phoneme_token_ids)
        
        semantic_token_length = semantic_token_ids.size(1)

        # pad or curtail
        
        max_length = self.max_downsampled_length
        
        if exists(max_length):
            if semantic_token_length > max_length:
                max_start = semantic_token_length - max_length
                start = torch.randint(0, max_start, (1, ))
                semantic_token_ids = semantic_token_ids[:, start:start + max_length]
                phoneme_token_ids = phoneme_token_ids[:, start:start + max_length]
            else:
                semantic_token_ids = F.pad(semantic_token_ids, (0, max_length - semantic_token_length), 'constant')
                phoneme_token_ids = F.pad(phoneme_token_ids, (0, max_length - semantic_token_length), 'constant')

        semantic_token_ids = rearrange(semantic_token_ids, '1 ... -> ...')
        phoneme_token_ids = rearrange(phoneme_token_ids, '1 ... -> ...')
        
        return semantic_token_ids, phoneme_token_ids

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)
    
def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
