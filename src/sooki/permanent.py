import sooki
from functools import partial
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

for name, target in sooki.registrations().items():
    jax.ffi.register_ffi_target(name, target)

gpu = False
gpu_targets = {}
if hasattr(sooki, "gpu_ops"):
    try:
        gpu_targets = sooki.gpu_ops.foo()
        for name, target in gpu_targets.items():
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            gpu = True
    except (ImportError, AttributeError) as e:
        print(f"GPU support initialization failed: {e}")
        gpu = False
else:
    print("No GPU module found. Continuing with CPU support only.")


@partial(jax.custom_vjp)
def perm(A, rows, cols):
    n, m = A.shape
    if rows.ndim != 1 or cols.ndim != 1:
        raise ValueError("perm: rows and cols must be 1D arrays")
    if rows.dtype not in (jnp.uint32, jnp.uint64) or cols.dtype not in (
        jnp.uint32,
        jnp.uint64,
    ):
        raise ValueError("perm: rows and cols must be uint32 or uint64")
    total_in = int(rows.sum())
    total_out = int(cols.sum())
    if total_in != total_out:
        raise ValueError(f"perm: sum(rows)={total_in} must equal sum(cols)={total_out}")
    if A.dtype != jnp.complex128:
        raise ValueError("perm: A.dtype must be complex128")
    if total_in == 0 and total_out == 0:
        return 1.0

    out_type = jax.ShapeDtypeStruct((), A.dtype)

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            out_type,
            vmap_method="broadcast_all",
        )(A, rows, cols)

    return jax.lax.platform_dependent(cpu=impl("perm"), cuda=impl("dperm"))


def perm_fwd(A, rows, cols):
    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            (
                jax.ShapeDtypeStruct((), A.dtype),
                jax.ShapeDtypeStruct((), A.dtype),
            ),
            vmap_method="broadcast_all",
        )(A, rows, cols)

    y, res = jax.lax.platform_dependent(cpu=impl("perm_fwd"), cuda=impl("dperm_fwd"))
    return y, (res, A, rows, cols)


def perm_bwd(res, cot):
    res, A, rows, cols = res

    def impl(target_name):
        return lambda: (
            cot * jax.ffi.ffi_call(
                target_name,
                jax.ShapeDtypeStruct(A.shape, A.dtype),
                vmap_method="broadcast_all",
            )(res, A, rows, cols),
            None,
            None,
        )

    return jax.lax.platform_dependent(cpu=impl("perm_bwd"), cuda=impl("dperm_bwd"))


perm.defvjp(perm_fwd, perm_bwd)
