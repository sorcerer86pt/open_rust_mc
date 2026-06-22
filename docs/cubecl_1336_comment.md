I am hitting this on **NVIDIA / Vulkan (Windows 11)** too, with one difference worth noting: here it does not race silently like on AMD/RADV — it **hard-crashes at dispatch** with `STATUS_ACCESS_VIOLATION` (exit `0xC0000005`). I think it's the same cause (per-thread private `Array<T>` going wrong under register pressure), but with a worse symptom.

### Setup
- cubecl `=0.10.0` from crates.io, `wgpu` runtime, Vulkan backend
- NVIDIA GPU, Windows 11, f64 enabled (adapter has `SHADER_F64`)

### What happens
A kernel with a lot of private `Array<T>` state compiles fine — `CUBECL_DEBUG_LOG` prints a full WGSL shader ending in `[END_KERNEL_COMPILATION]` — but the process dies at the first dispatch:

```
exit code: 0xC0000005  (STATUS_ACCESS_VIOLATION)
```

No nagging or validation error first. The crash is in driver code during dispatch part.

### What I tested
I split each piece into its own kernel on the same machine:

| Kernel | Result |
|---|---|
| PCG RNG in `u64` + `Array<Atomic<u32>>.fetch_add` | works |
| one big geometry helper (`find_cell`, 9 `&mut Array` params, depth-4 loop) | works |
| two levels of `#[cube]` helper calls | works |
| a 27-parameter `#[cube]` helper, called once | works |
| 8 × `Array<f64>::new(64)` (4 KB private), write then sum | works (no race, no crash) |
| **full kernel: `find_cell` + `trace_step` + per-event loop, all those arrays live at once** | crashes at dispatch, even at 4 threads × 4 steps |

So on my machine raw byte count alone does not trigger it (the 4 KB kernel is fine), and neither does nesting or param count on their own. It only fails when a lot of mixed-type private `Array` state is live across a long control-flow body — which fits the reported suspected cause (SPIR-V moving `Function`-storage variables out of real thread-private storage under pressure): the bad address then faults on NVIDIA where it just races on RADV.

### Reproducer shape
The smallest case that still crashes for me is a recursive-geometry Monte Carlo simulation step: one `#[cube(launch)]` entry calling two `#[cube]` helpers that share nine `Array::new(4)` stacks plus a 16-entry region stack, all f64/i32:

```rust
#[cube]
fn cell_contains(/* … */) -> u32 {
    let mut stack = Array::<u32>::new(16usize);
    // postfix CSG stack machine
}

#[cube]
fn find_cell(/* … */, st_cell: &mut Array<i32>, st_offx: &mut Array<f64>, /* 9 stacks */) -> u32 {
    // depth-4 descent, calls cell_contains
}

#[cube]
fn trace_step(/* … */, st_cell: &mut Array<i32>, /* same 9 stacks */, out: &mut Array<f64>) {
    // nearest-surface scan, calls surf_dist/grid_dist
}

#[cube(launch)]
fn transport(/* buffers */, atomics: &mut Array<Atomic<u32>>) {
    let mut st_cell = Array::<i32>::new(4usize);
    let mut st_offx = Array::<f64>::new(4usize);
    // 7 more length-4 stacks, a length-2 u64 RNG, a length-3 scratch
    let depth = find_cell(/* … */, &mut st_cell, &mut st_offx, /* … */);
    // per-step loop: trace_step + collision sampling + atomics.fetch_add(…)
    // → crashes at dispatch; the WGSL it generates looks fine.
}
```

Glad to attach a trimmed standalone code example if that helps.

