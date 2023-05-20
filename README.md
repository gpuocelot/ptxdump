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

Similarly, `ptxdump_example` has been compiled with CUDA 7.5. As the method works in the same way, we could assume it covers a large set of CUDA versions:

```
LD_PRELOAD=./libptxdump.so ./ptxdump_example








.version 4.3
.target sm_35
.address_size 64



.weak .func (.param .b32 func_retval0) cudaMalloc(
.param .b64 cudaMalloc_param_0,
.param .b64 cudaMalloc_param_1
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


.weak .func (.param .b32 func_retval0) cudaFuncGetAttributes(
.param .b64 cudaFuncGetAttributes_param_0,
.param .b64 cudaFuncGetAttributes_param_1
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


.weak .func (.param .b32 func_retval0) cudaDeviceGetAttribute(
.param .b64 cudaDeviceGetAttribute_param_0,
.param .b32 cudaDeviceGetAttribute_param_1,
.param .b32 cudaDeviceGetAttribute_param_2
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


.weak .func (.param .b32 func_retval0) cudaGetDevice(
.param .b64 cudaGetDevice_param_0
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


.weak .func (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


.weak .func (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
.reg .b32 %r<2>;


mov.u32 %r1, 30;
st.param.b32	[func_retval0+0], %r1;
ret;
}


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
