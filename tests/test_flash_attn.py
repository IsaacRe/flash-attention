import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import flash_attn_wrapper  # noqa: F401

from vllm_flash_attn.bert_padding import pad_input, unpad_input
from vllm_flash_attn.flash_attn_interface import _get_block_size_n, convert_kvc_S_to_attn

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False, key_leftpad=None
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
    check_s=None,
    check_lse=None,
    seqlens=None,
    kvc_S=None,
    converted_kvc_S=None,
    kvc_S_copy=None,
    check_lse_copy=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias

    if check_s is not None:
        assert check_s.dtype == torch.float
        # print(scores)
        # print(check_s.shape, scores.shape)

        # print((check_s - scores)[check_s != float('-inf')].abs().max())
        # print((check_s.exp() - scores.exp())[check_s != float('-inf')].abs().max())

        assert torch.allclose(check_s.type(scores.dtype), scores, atol=1e-2)
        assert torch.allclose(check_s.exp(), scores.exp(), atol=1e-1)
    if check_lse is not None:
        lse_ref = torch.logsumexp(scores, dim=-1)
        lse_out = torch.ones_like(lse_ref) * float('-inf')
        start = 0
        for i, l in enumerate(seqlens[1:]):
            curr_l = l - start
            lse_ref_ = lse_ref[i,:,:curr_l]
            lse_ = check_lse[:,start:l]
            lse_out[i,:,:curr_l] = lse_
            # print(lse_ref_)
            # print(lse_)
            assert torch.allclose(lse_ref_, lse_, atol=1e-2)
            start = l
        # print((lse_out - lse_ref)[lse_out != float('-inf')].abs().max())
        # print((lse_out.exp() - lse_ref.exp())[lse_out != float('-inf')].abs().max())
        assert torch.allclose(lse_out[lse_out != float('-inf')], lse_ref[lse_out != float('-inf')], atol=1e-2)
        assert torch.allclose(lse_out[lse_out != float('-inf')].exp(), lse_ref[lse_out != float('-inf')].exp(), atol=1e-1)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if (check_s is not None) and check_lse is not None:
        kagg_window = 32
        check_attention = check_s.exp() / lse_out.exp()[...,None]
        assert torch.allclose(check_attention[lse_out != float('-inf')], attention[lse_out != float('-inf')], atol=1e-3)
        assert (converted_kvc_S == convert_kvc_S_to_attn(kvc_S_copy, check_lse_copy, kagg_window, seqlens, d)).all()
        if converted_kvc_S is not None:
            assert kvc_S is not None
            assert (kvc_S == kvc_S_copy).all()
            assert (check_lse == check_lse_copy).all()
            s_ = kvc_S_copy / math.sqrt(d)
            start = 0
            print('hello')
            # print(kvc_S_copy)
            for i, l in enumerate(seqlens[1:]):
                curr_l = l - start
                nq = min(kagg_window, curr_l)
                lse = check_lse_copy[:,l-nq:l]
                s_[start:l,:,-nq:] -= lse[None]

                if i == 0:
                    print(lse)
                    print(s_.shape, start, l, nq)
                    # print(s_)
                    print(s_[start:l,:,-nq:])
                    print(s_[start:l,:,-nq:].exp())

                # print(attention[i,:,curr_l-nq:curr_l,:curr_l].shape)
                attn_ref = attention[i,:,curr_l-nq:curr_l,:curr_l].transpose(0, 1).transpose(0, 2)

                # print(attn_ref.shape, s_[start:l,:,-nq:].shape)
                assert torch.allclose(attn_ref, s_[start:l,:,-nq:].exp(), atol=1e-3)
                try:
                    assert (s_[start:l,:,-nq:].exp() == converted_kvc_S[start:l,:,-nq:]).all()
                except:
                    print(torch.where(s_[start:l,:,-nq:].exp() != converted_kvc_S[start:l,:,-nq:]))
                    print(s_[start:l,:,-nq:].exp()[0,0,0])
                    print(converted_kvc_S[start:l,:,-nq:][0,0,0])
                    raise

                start = l
            s_[kvc_S == float('-inf')] = float('-inf')
            s_ = s_.exp()

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        reorder_ops=reorder_ops,
        key_leftpad=key_leftpad,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        reorder_ops=reorder_ops,
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    seqlens=None,
    check_lse=None,
    check_s=None,
    invert_block_norm=True,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    out_mask = None
    if check_s is not None:
        # print(scores.shape)
        # print(check_s.shape)
        # print(scores)
        mask1 = check_s == -float('inf')
        mask2 = scores == -float('inf')
        out_mask = mask1 | mask2
        check_s[mask1] = scores[mask1]
        check_s[mask2] = -float('inf')
        # print(check_s)
        # print(torch.isclose(scores, check_s, atol=1e-3))
        # print(torch.where(torch.isclose(scores, check_s, atol=1e-3)))
        # scores (computed from raw kv's) != check_s (computed from raw kv's in correspondance with S_dmask output)
        assert torch.allclose(scores, check_s, atol=1e-3)
        if not invert_block_norm:
            # print(torch.isclose(attn_unnorm, scores, atol=1e-3) | (mask1) | (mask2))
            assert torch.allclose(attn_unnorm[(~mask1) & (~mask2)], scores[(~mask1) & (~mask2)], atol=1e-3)

    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    lse__ = lse.clone()
    if check_lse is not None:
        start = 0
        for i, l in enumerate(seqlens[1:]):
            curr_l = l - start
            lse_ = lse[i,:,:curr_l]
            check_lse_ = check_lse[:,start:l]
            # print(lse_.shape)
            # print(check_lse_.shape)
            # print(lse_)
            # print(check_lse_)
            assert torch.allclose(lse_, check_lse_, atol=1e-3)
            lse[i,:,:curr_l] = check_lse_
            start = l
        assert (lse__ != lse).any()
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp((m if invert_block_norm else 0) - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype), out_mask


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()


@pytest.mark.parametrize("key_attn_agg_window", [32])
@pytest.mark.parametrize("kvpacked", [False])
# @pytest.mark.parametrize('kvpacked', [False])
@pytest.mark.parametrize("dtype", ([torch.float16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("mha_type", ["gqa"])
# @pytest.mark.parametrize('mha_type', ["mqa"])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [32, 128]) #, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 147),
        # (8, 8),
        # (40, 40),
        # (72, 72),
        (136, 136),
        # (128, 217),
        # (113, 211),
        # (108, 256),
        # (256, 512),
        # (512, 256),
        # (1024, 1024),
        # (1023, 1024),
        # (1024, 1023),
        # (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("softcap", [0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, softcap, key_attn_agg_window
):
    if key_attn_agg_window > 0:
        dropout_p = 0.0
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if softcap > 0.0 and dropout_p > 0.0:
        pytest.skip("Softcap and dropout not supported together")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    nheads = 2
    nheads_k = 1
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap

    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = query_padding_mask.clone()
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        ) = generate_qkv(q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True)
        out_unpad, sm_lse, S_dmask = torch.ops.vllm.flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        out_unpad, sm_lse, S_dmask = torch.ops.vllm.flash_attn_varlen_func(
            q_unpad.clone(),
            k_unpad.clone(),
            v_unpad.clone(),
            cu_seqlens_q.clone(),
            cu_seqlens_k.clone(),
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=key_attn_agg_window == 0,
            key_attn_agg_window=key_attn_agg_window,
        )
        _, sm_lse_, S_dmask_ = torch.ops.vllm.flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
            key_attn_agg_window=0,
        )

    # print(S_dmask.shape)
    # print(S_dmask_.shape)
    # print(sm_lse.shape)
    # print(sm_lse_.shape)

    scale = math.sqrt(d)

    k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k_rep.float()) / scale
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    scores_ = scores.clone()
    # print(scores_[0,0,-32,0])
    # print((q.float()[0,-32,0] * k_rep.float()[0,0,0]).sum() / scale)
    # print((q.float()[0,-32,0][None,None] @ k_rep.float()[0,0,0][None,:,None]) / scale)
    # # import pdb; pdb.set_trace()
    # print((q.float()[0,-32][:,None] @ k_rep.float()[0].transpose(0, 1).transpose(-1, -2)) / scale)
    # print((q.float()[0,-32:].transpose(0, 1) @ k_rep.float()[0].transpose(0, 1).transpose(-1, -2)) / scale)
    # print((q.float()[0,104][:,None] @ k_rep.float()[0].transpose(0, 1).transpose(-1, -2)) / scale)

    out_S_dmask = torch.ones((batch_size, nheads, seqlen_q, seqlen_k), device=device, dtype=torch.float) * -float('inf')
    start = 0
    for i, l in enumerate(cu_seqlens_q[1:]):
        seqlen = l - start
        padding = seqlen_q - seqlen
        nq = min(key_attn_agg_window, seqlen)
        out_ = q.float()[i,:seqlen].transpose(0, 1) @ k_rep.float()[i,:seqlen].transpose(0, 1).transpose(1, 2)
        # print(out_[0,-32,0] / scale)
        # print(out_[0,104,0] / scale)
        out_ = out_[:,-nq:]
        q_ = q_unpad[start:l].transpose(0, 1)
        # print(q_)
        # print(q.float()[i,:seqlen].transpose(0, 1))
        k_ = k_unpad[start:l].repeat(1, nheads // nheads_k, 1).transpose(0, 1).transpose(1, 2)
        # print(q_.shape, k_.shape)
        out_ref = (q_.type(torch.float) @ k_.type(torch.float)).type(torch.float)
        # print(out_ref.shape)
        out_ref = out_ref[:,-nq:]
        out = S_dmask[start:l,:,-nq:].transpose(0,2).transpose(0,1)
        # print(out_)
        # print(scores_[i,:,-nq:,:seqlen])
        # print(out_ref.shape)
        # print(out.shape)
        # print(out[0])
        # print(out_ref[0])
        mask = out != -float('inf')
        assert not out[mask].isnan().any()
        # print(out_S_dmask[i])
        # print((out - out_ref)[mask].abs().max())
        assert torch.allclose(out[mask], out_ref[mask], atol=1e-4)
        assert torch.allclose(out_ref[mask], out_[mask], atol=1e-4)
        assert torch.allclose(out[mask], out_[mask], atol=1e-4)

        # print(scores_.shape)
        assert torch.allclose(scores_[i,:,-nq-padding:-padding,:seqlen], out_ / scale, atol=1e-3)
        scores_[i,:,-nq-padding:-padding,:seqlen][mask] = out[mask] / scale

        start = l

    # print(scores)
    # scores_ = scores.clone()
    # scores_[out_S_dmask != float('-inf')] = out_S_dmask[out_S_dmask != float('-inf')]
    # print(scores_)

    # print(torch.isclose(scores, scores_, atol=1e-3))
    assert (scores != scores_).any()
    # print(torch.where(~torch.isclose(scores, scores_, atol=1e-3)))
    assert torch.allclose(scores, scores_, atol=1e-3), 'hi'

    # print(S_dmask[:,0].transpose(0,1))

    # k_ = k_unpad[:,0]
    # q_ = q_unpad[:,0]
    # print(k_unpad.shape)
    # out = (q_ @ k_.transpose(0, 1)).type(torch.float)
    # print(out[-key_attn_agg_window:])
    # print(out[-key_attn_agg_window:].shape)
    # assert S_dmask.dtype == out.dtype == torch.float

    # print(S_dmask[:,0].transpose(0,1) - out[-key_attn_agg_window:])
    # print(S_dmask[:,0].transpose(0,1).shape, out[-key_attn_agg_window:].shape)

    # print(torch.where(torch.isclose(S_dmask[:,0].transpose(0,1), out[-key_attn_agg_window:], atol=1e-2)))
    # print(cu_seqlens_k, cu_seqlens_q)
    S_dmask_copy = S_dmask.clone()
    sm_lse_copy = sm_lse.clone()
    converted_kvc_S = convert_kvc_S_to_attn(S_dmask, sm_lse, key_attn_agg_window, cu_seqlens_k, d)
    converted_kvc_S_copy = converted_kvc_S.clone()

    # kp1 = cu_seqlens_k[1]
    # qp1 = cu_seqlens_q[1]

    # m_block_size = 64
    # n_block_size = 64

    # print("###### SEQ 0, HEAD 0 ######")
    # agg_range = min(key_attn_agg_window, qp1)
    # a = S_dmask.clone()[:,0,:].transpose(0, 1)
    # b = S_dmask_.clone()[0,0,:,:]
    # print(a.shape, b.shape)
    # b = b[:,:qp1]
    # b = b[:qp1]  # remove empty padding
    # b = b[b.size(0)-agg_range:]  # get only queries in agg window
    # a = a[a.size(0)-agg_range:,:qp1]
    # print(a.shape, b.shape)
    # print(a)
    # print(b)
    # print(a[:m_block_size,:n_block_size])
    # print(b[:m_block_size,:n_block_size])
    # print(torch.where(a != b))
    # assert (a == b).all()

    # print("###### SEQ 1, HEAD 0 ######")
    # agg_range = min(key_attn_agg_window, qp1)
    # kp2 = cu_seqlens_k[2]
    # qp2 = cu_seqlens_q[2]
    # a = S_dmask.clone()[:,0,:].transpose(0, 1)
    # b = S_dmask_.clone()[1,0,:,:]
    # b = b[:,:qp2-qp1]
    # b = b[:qp2-qp1]
    # b = b[b.size(0)-agg_range:]
    # a = a[a.size(0)-agg_range:,qp1:qp2]
    # print(a.shape, b.shape)
    # print(a)
    # print(b)
    # print(a[:m_block_size,:n_block_size])
    # print(b[:m_block_size,:n_block_size])
    # assert (a == b).all()

    # print("###### SEQ 1, HEAD 1 ######")
    # a = S_dmask.clone()[:,1,:].transpose(0, 1)
    # b = S_dmask_.clone()[1,1,:,:]
    # b = b[:,:qp2-qp1]
    # b = b[:qp2-qp1]
    # b = b[b.size(0)-agg_range:]
    # a = a[a.size(0)-agg_range:,qp1:qp2]
    # print(a.shape, b.shape)
    # print(a)
    # print(b)
    # print(a[:m_block_size,:n_block_size])
    # print(b[:m_block_size,:n_block_size])
    # assert (a == b).all()

    # print("Checking aggregation-range attention and S_dmask equality")
    # start_seqlen = 0
    # for i, cum_seqlen in enumerate(cu_seqlens_q[1:]):
    #     curr_seqlen = cum_seqlen - start_seqlen
    #     agg_range = min(curr_seqlen, key_attn_agg_window)
    #     print(agg_range)
    #     a = S_dmask.clone().transpose(0, 2).transpose(0, 1)[:,S_dmask.size(2)-agg_range:,start_seqlen:cum_seqlen]
    #     b = S_dmask_.clone()[i,:,curr_seqlen-agg_range:curr_seqlen,:curr_seqlen]
    #     print(a.shape, b.shape)
    #     # assert a.shape == (2, 128, 136)
    #     print(a)
    #     print(b)
    #     assert (a == b).all()
    #     start_seqlen += cum_seqlen

    # print(torch.where(b == 0.2079))
    # print(torch.where(a == 0.2079))
    # print(torch.where(b == 0.3132))
    # print(torch.where(a == 0.3132))
    # print(torch.where((a != b)))
    # print(torch.where((a != b)[:,0])[0].cpu().numpy().tolist())
    # print(torch.where((a != b).any(dim=1))[0].cpu().numpy().tolist())
    # print(torch.where((a != 0)[-1])[0].cpu().numpy().tolist())

    out = output_pad_fn(out_unpad)
    # assert (S_dmask > 0).any()
    S_dmask_converted = convert_flash_attn_S_to_softmax(
        S_dmask_,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        d,
        False,
        causal=causal,
        window_size=window_size,
    )
    attn_unnorm = S_dmask_converted.abs()
    if kvpacked:
        kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
        k_rep, v_rep = kv_rep.unbind(dim=2)
    else:
        k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    attn, _ = normalize_flash_attn_S(
        attn_unnorm,
        q,
        k_rep,
        v_rep,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p > 0.0,
        causal=causal,
        window_size=window_size,
        # seqlens=cu_seqlens_q,
        # check_lse=sm_lse,
        # check_s=scores_,
    )
    # attn1, _ = normalize_flash_attn_S(
    #     attn_unnorm,
    #     q,
    #     k_rep,
    #     v_rep,
    #     query_padding_mask,
    #     key_padding_mask,
    #     attn_bias,
    #     dropout_p > 0.0,
    #     causal=causal,
    #     window_size=window_size,
    #     seqlens=cu_seqlens_q,
    # )
    # attn2, out_mask = normalize_flash_attn_S(
    #     scores_,
    #     q,
    #     k_rep,
    #     v_rep,
    #     query_padding_mask,
    #     key_padding_mask,
    #     attn_bias,
    #     dropout_p > 0.0,
    #     causal=causal,
    #     window_size=window_size,
    #     seqlens=cu_seqlens_q,
    #     check_lse=sm_lse,
    #     check_s=scores_,
    #     invert_block_norm=False,
    # )
    if dropout_p > 0.0:
        dropout_mask = S_dmask_converted >= 0
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        # print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            # check_s=scores_,
            # check_lse=sm_lse,
            # seqlens=cu_seqlens_q,
            # kvc_S=S_dmask,
            # converted_kvc_S=converted_kvc_S_copy,
            # kvc_S_copy=S_dmask_copy,
            # check_lse_copy=sm_lse_copy,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )

    assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert (attn1 - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert (attn2[~out_mask] - attn_ref[~out_mask]).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()

    print(attn_ref[0,0])
    print(converted_kvc_S[-8:,0].transpose(0, 1))

    print("Checking aggregation-range attention equality")
    start_seqlen = 0
    max_diff = 2 * (attn_pt - attn_ref).abs().max().item()
    for i, cum_seqlen in enumerate(cu_seqlens_q[1:]):
        curr_seqlen = cum_seqlen - start_seqlen
        agg_range = min(curr_seqlen, key_attn_agg_window)
        print(agg_range)
        a = converted_kvc_S.clone().transpose(0, 2).transpose(0, 1)[:,converted_kvc_S.size(2)-agg_range:,start_seqlen:cum_seqlen]
        b = attn_ref.clone()[i,:,curr_seqlen-agg_range:curr_seqlen,:curr_seqlen]

        print(a.shape, b.shape)
        print(a)
        print(b)
        print((a-b).abs())
        print(torch.where((a-b).abs() > max_diff))
        print(i, max_diff)
        assert (a - b).abs().max().item() <= max_diff
        start_seqlen = cum_seqlen

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()


@pytest.mark.parametrize("key_attn_agg_window", [128])
@pytest.mark.parametrize("kvpacked", [False])
# @pytest.mark.parametrize('kvpacked', [False])
@pytest.mark.parametrize("dtype", ([torch.float16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("mha_type", ["gqa"])
# @pytest.mark.parametrize('mha_type', ["mqa"])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [32, 128]) #, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 147),
        # (8, 8),
        (136, 136),
        # (128, 217),
        # (113, 211),
        # (108, 256),
        # (256, 512),
        # (512, 256),
        # (1024, 1024),
        # (1023, 1024),
        # (1024, 1023),
        # (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@pytest.mark.parametrize("dropout_p", [0.17])
@pytest.mark.parametrize("softcap", [0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_varlen_output_old(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, softcap, key_attn_agg_window
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if softcap > 0.0 and dropout_p > 0.0:
        pytest.skip("Softcap and dropout not supported together")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap

    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        ) = generate_qkv(q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True)
        out_unpad, sm_lse, S_dmask = torch.ops.vllm.flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        out_unpad, sm_lse, S_dmask = torch.ops.vllm.flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn, _ = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        if not alibi:
            assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.04)
