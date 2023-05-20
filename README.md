# PTXDump: dump PTX from a CUDA binary

Dumps PTX from a binary dynamically linked against libcudart.

These days the PTX code embedded into CUDA binaries is compressed with an undocumented algorithm. A matching decompression method has been created by [cuda-fatbin-decompression](https://github.com/gpuocelot/cuda-fatbin-decompression) project.

## Building

```
mkdir build
cd build
cmake ..
make
```

## Testing

The `ptxdump_example` sample CUDA binary has been compiled with CUDA 11.7. The following command dumps the corresponding sample PTX as shown below:

```
LD_PRELOAD=./libptxdump.so ./ptxdump_example

.version 7.7
.target sm_35
.address_size 64

.visible .entry _Z6kernelPi(
.param .u64 _Z6kernelPi_param_0
)
{
.reg .b32 %r<4>;
.reg .b64 %rd<5>;

ld.param.u64 %rd1, [_Z6kernelPi_param_0];
cvta.to.global.u64 %rd2, %rd1;
mov.u32 %r1, %tid.x;
mul.wide.u32 %rd3, %r1, 4;
add.s64 %rd4, %rd2, %rd3;
ld.global.u32 %r2, [%rd4];
add.s32 %r3, %r2, 1;
st.global.u32 [%rd4], %r3;
ret;

}
```

