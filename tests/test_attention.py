import torch
from src.models.attention import LatentQueryAttention

L = 5
F = 3
d = 8
m = 4
batch_size = 4
num_tokens = L * F

attention = LatentQueryAttention(embed_dim=d, num_queries=m)

tokens = torch.randn(batch_size, num_tokens, d)
print(f"\nInput tokens shape: {tokens.shape}")

Z, A = attention(tokens)

print(f"\nExpected Z shape: ({batch_size}, {m}, {d})")
print(f"Actual Z shape:   {Z.shape}")

print(f"\nExpected A shape: ({batch_size}, {m}, {num_tokens})")
print(f"Actual A shape:   {A.shape}")

assert Z.shape == (batch_size, m, d), "Z shape mismatch"
assert A.shape == (batch_size, m, num_tokens), "A shape mismatch"
print("\nShape assertions passed")

print("\nAttention weight row sums (each should be 1.0):")
print(f"  batch=0, query=0: {A[0, 0].sum().item():.6f}")
print(f"  batch=0, query=1: {A[0, 1].sum().item():.6f}")
print(f"  batch=1, query=0: {A[1, 0].sum().item():.6f}")

print(
    f"\nAre query 0 and query 1 attention maps different: {not torch.allclose(A[0, 0], A[0, 1])}"
)
print(
    f"Are batch 0 and batch 1 attention maps different: {not torch.allclose(A[0, 0], A[1, 0])}"
)

print(f"\nAttention weight min: {A.min().item():.6f}")
print(f"Attention weight max: {A.max().item():.6f}")
print(f"Attention weight mean: {A.mean().item():.6f}")
print(f"Uniform baseline (1/num_tokens): {1 / num_tokens:.6f}")
