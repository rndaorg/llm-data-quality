WORLD_GROUP = list(range(8))

DATA_PARALLEL_GROUP_A = [0, 1, 2, 3]
DATA_PARALLEL_GROUP_B = [4, 5, 6, 7]

TENSOR_PARALLEL_GROUPS = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]

def get_groups(rank):

    groups = []

    if rank in DATA_PARALLEL_GROUP_A:
        groups.append("DP_GROUP_A")

    if rank in DATA_PARALLEL_GROUP_B:
        groups.append("DP_GROUP_B")

    for idx, group in enumerate(TENSOR_PARALLEL_GROUPS):
        if rank in group:
            groups.append(f"TP_GROUP_{idx}")

    return groups

print(
    f"Rank {rank} belongs to: {get_groups(rank)}"
)
