from typing import Mapping
import numpy as np
import torch
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

restype_num = len(restypes)  # := 20.

unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']

restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(
        sequence: str,
        mapping: Mapping[str, int],
        map_unknown_to_x: bool = False) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.
    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain
        amino acid 'X', an error will be thrown. If False, any amino acid not in
        the mapping will throw an error.
    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
      the sequence.
    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError('The mapping must have values from 0 to num_unique_aas-1 '
                         'without any gaps. Got: %s' % sorted(mapping.values()))

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping['X'])
            else:
                raise ValueError(f'Invalid character in the sequence: {aa_type}')
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'data.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化列表用于存储特征和标签
sequences = []
labels = []

# 解析文件内容
for line in lines:
    if line.startswith('>'):
        # 处理标签
        _, label = line.strip().split('|')
        labels.append(int(label))
    else:
        # 处理序列数据
        sequences.append(line.strip())

for i in sequences:
    seq_tensor = sequence_to_onehot(i, restype_order_with_x)

print()