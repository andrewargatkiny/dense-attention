import torch
import pytest

from src.positional_embeddings import RoPE, RelPEBase


class ReferenceRoPE(RelPEBase):
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, num_heads=None, emb_fraction=1.0):
        super(ReferenceRoPE, self).__init__()
        """Reference RoPE implementation which handles merged head and 
        embedding (num_h+dim) dimensions correctly.

        Parameters
        ----------
        seq_len : int
            Maximum length of input sequence. RoPE cache will have this length 
            in the sequence dimension.     
        n_elem : int
            Embedding dimension of one head
        base : int
            RoPE \Theta base. Default is 10000.     
        num_heads : int, optional
            Number of heads. Defaults to None. If supplied, cached RoPE buffers 
            take form of `bs (1), seqlen, n_elem * num_heads`, else 
            `bs (1), headdim (1), seqlen, n_elem`.
        emb_fraction: float
            Fraction of the embedding dimension to which RoPE should be 
            applied. Default is 1.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2) / n_elem))
        # theta = 1.0 / (base ** (torch.ones(n_elem // 2) / n_elem))

        if emb_fraction != 1.0:
            n_elem_rope = int(n_elem * emb_fraction)
            # RoPE dimension should be divisible by 2.
            n_elem_rope = n_elem_rope - n_elem_rope % 2
            theta_rope = 1.0 / (base ** (torch.arange(0, n_elem_rope, 2)
                                         / n_elem_rope))
            theta = torch.ones(size=(n_elem // 2,))
            theta[:n_elem_rope // 2] = theta_rope

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len)

        # Calculate the product of position index and $\theta_i$
        angles = torch.outer(seq_idx, theta).repeat(1, 2).float()
        self.rotate_half = self.rotate_half_classic
        if num_heads is not None and num_heads > 1:
            angles = torch.outer(seq_idx, theta).repeat_interleave(2, 1).float()
            self.rotate_half = self.rotate_half_fused_dims
        cache_cos = torch.cos(angles).unsqueeze(0)
        cache_sin = torch.sin(angles).unsqueeze(0)
        if num_heads is None:
            cache_cos = cache_cos.unsqueeze(1)
            cache_sin = cache_sin.unsqueeze(1)
            # cache: bs (1), headdim (1), seqlen, embed dim
        else:
            cache_cos = cache_cos.repeat(1, 1, num_heads)
            cache_sin = cache_sin.repeat(1, 1, num_heads)
            # cache: bs (1), seqlen, embed dim * num heads
        self.register_buffer("cache_cos", cache_cos, persistent=False)
        self.register_buffer("cache_sin", cache_sin, persistent=False)

    @staticmethod
    def rotate_half_classic(x):
        """Rotates half the hidden dims of the input.
        From https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L84
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def rotate_half_fused_dims(x):
        """A version of the rotate half function for the case where head and
        embedding dimensions are not decoupled.
        """
        # x = (x0, x1, x2, x3, ...), y = (-x1, x0, -x3, x2, ...)
        y = torch.empty_like(x)
        y[..., 0::2] = - x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]
        return y

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        # truncate to support variable sizes
        seq_len = x.size(-2)
        cache_cos = self.cache_cos[..., :seq_len, :]
        cache_sin = self.cache_sin[..., :seq_len, :]
        pdtype = x.dtype
        # First half of the tensor [x1, x2] takes form x1 * cos - x2 * sin
        # Second half is x1 * cos + x2 * sin
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        """Applies RoPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows * window_size, num_heads * head_dim) or
        (bs, num_heads, num_windows * window_size, head_dim), depending on init."""
        cache_cos = self.cache_cos[..., :window_size, :].repeat(1, num_windows, 1)
        cache_sin = self.cache_sin[..., :window_size, :].repeat(1, num_windows, 1)
        pdtype = x.dtype
        # First half of the tensor [x1, x2] takes form x1 * cos - x2 * sin
        # Second half is x1 * cos + x2 * sin
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

    def apply_local_relpe2(self, x: torch.Tensor, window_size, num_windows):
        """Applies RoPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows, num_heads, window_size, head_dim) or
        (bs, num_windows, window_size, num_heads * head_dim),
        and that RoPE cache is respectively
        (bs (1), num_windows (1), num_heads (1), window_size, head_dim)
        or (bs (1), num_windows (1), window_size, num_heads * head_dim).

        This method has the same effect as `apply_relpe`, because by definition
        seq_len = window_size and additional `num_windows` dim is broadcasted
        either way. It will be deprecated in a future release.
        """
        cache_cos = self.cache_cos.unsqueeze(0)[..., :window_size, :]
        cache_sin = self.cache_sin.unsqueeze(0)[..., :window_size, :]
        pdtype = x.dtype
        # First half of the tensor [x1, x2] takes form x1 * cos - x2 * sin
        # Second half is x1 * cos + x2 * sin
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)


@pytest.mark.parametrize(
    "seq_len, window_size", [(32, 8), (21, 7), (33, 11)]
)
@pytest.mark.parametrize("head_dim", [2, 16, 32, 64])
@pytest.mark.parametrize("num_heads", [1, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_all_methods_one_func(seq_len, head_dim, num_heads, window_size, dtype):
    """
    Compares different RoPE methods in one function:
    Test cases:
      - apply_relpe (merged num_h+dim) == apply_relpe (reference)
        if num_h == 1 or dim == 2
      - apply_relpe (merged num_h+dim) == apply_relpe (separate num_h+dim)
      - apply_local_relpe (merged num_h+dim) == apply_local_relpe (reference)
        if num_h == 1 or dim == 2
      - apply_local_relpe (merged num_h+dim) != apply_relpe (merged num_h+dim)
        if window_size != seq_len
      - apply_local_relpe (merged num_h+dim) == apply_relpe (merged num_h+dim)
        if window_size == seq_len
      - apply_local_relpe (merged num_h+dim) == apply_local_relpe (sep. num_h+dim)
      - apply_local_relpe2 (merged num_h+dim) == apply_local_relpe2 (sep. num_h+dim)
      - apply_local_relpe2 (merged num_h+dim) == apply_local_relpe (merged num_h+dim)

    By transitivity, success of these test cases implies that more equalities
    hold, e.g.,
    apply_local_relpe2 (sep. num_h+dim) = apply_local_relpe (merged num_h+dim).
    """
    if head_dim % 2 != 0:
        pytest.skip("RoPE requires even head_dim")
    assert seq_len % window_size == 0, "window_size must divide seq_len"
    num_windows = seq_len // window_size
    # Constants: batch_size, relative and absolute tolerances
    B = 3
    RTOL = 1e-5
    ATOL = 1e-5
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiation of RoPE classes for merged and separated num_h+dim
    # dimensions cases and of the reference class.
    rope_h_merged = RoPE(
        seq_len=seq_len, n_elem=head_dim, base=10000, sep_head_dim=False,
        num_heads=num_heads,
    ).to(device)
    rope_h_sep = RoPE(
        seq_len=seq_len, n_elem=head_dim, base=10000, sep_head_dim=True,
        num_heads=num_heads,
    ).to(device)
    ref_rope = ReferenceRoPE(
        seq_len=seq_len, n_elem=head_dim, base=10000, num_heads=num_heads,
    ).to(device)

    # Instantiation of different shapes of one random input tensor.
    x_raw = torch.randn(B, seq_len, num_heads, head_dim,
                        device=device, dtype=dtype)
    x = x_raw.view(B, seq_len, num_heads * head_dim)
    x_w_sep = x_raw.view(B, num_windows, window_size, num_heads * head_dim)
    x_h_sep = x_raw.permute(0, 2, 1, 3)
    # x_h_sep: B, num_heads, seq_len, head_dim
    x_h_sep_w_sep = x_h_sep.reshape(B, num_heads, num_windows,
                                    window_size, head_dim)


    # apply_relpe
    out_rope = rope_h_merged.apply_relpe(x)
    out_ref = ref_rope.apply_relpe(x)
    out_rope_sep = rope_h_sep.apply_relpe(x_h_sep)
    assert out_rope.shape == out_ref.shape == (B, seq_len, num_heads * head_dim)
    # Merged and reference merged implementations should be equal for single
    # head or head dimension of 2 elements.
    if num_heads == 1 or head_dim == 2:
        assert torch.allclose(out_rope, out_ref, rtol=RTOL, atol=ATOL)
    assert out_rope_sep.shape == (B, num_heads, seq_len, head_dim)
    out_rope_sep = (out_rope_sep.permute(0, 2, 1, 3)
                    .reshape(B, seq_len, num_heads * head_dim))
    # Two ways of RoPE calculation (merged and separated heads) should be equal.
    assert torch.allclose(out_rope, out_rope_sep, rtol=RTOL, atol=ATOL)

    # apply_local_relpe
    out_rope_loc = rope_h_merged.apply_local_relpe(x, window_size=window_size,
                                                   num_windows=num_windows)
    out_ref_loc = ref_rope.apply_local_relpe(x, window_size=window_size,
                                             num_windows=num_windows)
    assert out_rope_loc.shape == out_ref_loc.shape == (B, seq_len, num_heads * head_dim)
    # Merged and reference merged implementations should be equal for single
    # head or head dimension of 2 elements.
    if num_heads == 1 or head_dim == 2:
        assert torch.allclose(out_rope_loc, out_ref_loc, rtol=RTOL, atol=ATOL)
    # Together, the two clauses below test that
    # (window_size == seq_len) <=> (local RoPE == global RoPE).
    # 1. (window_size != seq_len) => (local RoPE != global RoPE)
    assert window_size == seq_len or not torch.allclose(out_rope_loc, out_rope,
                                                        rtol=RTOL, atol=ATOL)
    # 2. (window_size == seq_len) => (local RoPE == global RoPE)
    assert torch.allclose(out_rope_loc[:, :window_size, ...],
                          out_rope[:, :window_size, ...], rtol=RTOL, atol=ATOL)

    out_rope_sep_loc = rope_h_sep.apply_local_relpe(
        x_h_sep, window_size=window_size, num_windows=num_windows
    )
    assert out_rope_sep_loc.shape == (B, num_heads, seq_len, head_dim)
    out_rope_sep_loc = (out_rope_sep_loc.permute(0, 2, 1, 3)
                        .reshape(B, seq_len, num_heads * head_dim))
    # Two ways of local RoPE calculation (merged and separated heads) should
    # be equal.
    assert torch.allclose(out_rope_loc, out_rope_sep_loc, rtol=RTOL, atol=ATOL)


    # apply_local_relpe2
    out_rope_loc2 = rope_h_merged.apply_local_relpe2(
        x_w_sep, window_size=window_size, num_windows=num_windows
    )
    assert out_rope_loc2.shape == (
        B, num_windows, window_size, num_heads * head_dim)
    out_rope_sep_loc2 = rope_h_sep.apply_local_relpe2(
        x_h_sep_w_sep, window_size=window_size, num_windows=num_windows
    )
    assert out_rope_sep_loc2.shape == (
        B, num_heads, num_windows, window_size, head_dim)
    out_rope_sep_loc2 = (
        out_rope_sep_loc2.permute(0, 2, 3, 1, 4)
        .reshape(B, num_windows, window_size, num_heads * head_dim)
    )
    # Separate and merged heads methods should yield equal results
    assert torch.allclose(out_rope_loc2, out_rope_sep_loc2, rtol=RTOL, atol=ATOL)
    out_rope_loc2 = out_rope_loc2.view(B, seq_len, num_heads * head_dim)
    # `apply_local_relpe` and `apply_local_relpe2` functions should yield the
    # same results.
    assert torch.allclose(out_rope_loc2, out_rope_loc, rtol=RTOL, atol=ATOL)

