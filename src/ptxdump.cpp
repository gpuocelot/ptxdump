#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ptxdump.h"

extern "C" {
#include "fatbin-decompress.h"
} // extern "C"

void ptxdump(const void *fatCubin)
{
	assert(fatCubin != 0);

	if(*(int*)fatCubin == __cudaFatMAGIC) 
	{
		// This format has been used by very old versions of CUDA.
		// Perhaps we should remove this, as nobody should be interested
		// in it anymore.

		__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;
		assert(binary->ident != 0);
		std::cout << binary->ident << std::endl;
		assert(binary->ptx != 0);
		assert(binary->ptx->ptx != 0);
	  
		int i = 0;
		while(binary->ptx[i].ptx != 0)
		{
			assert(binary->ptx[i].gpuProfileName != 0);
			std::cout << binary->ptx[i].gpuProfileName << std::endl;
			assert(binary->ptx[i].ptx != 0);
			std::cout << binary->ptx[i].ptx << std::endl;
			i++;
		}
	}
	else if(*(unsigned*)fatCubin == __cudaFatMAGIC2)
	{
		auto binary = (__cudaFatCudaBinary2*) fatCubin;
		auto header = (fat_elf_header*) binary->fatbinData;

		bool seen_ptx = false;
		char* base = (char*)(header + 1);
		auto entry = (fat_text_header*)(base);
		for (long long unsigned int offset = 0; offset < header->header_size;
			entry = (fat_text_header*)(base + offset), offset += entry->header_size + entry->size) 
		{
			if (!(entry->kind & FATBIN_2_PTX))
				continue;

			std::vector<uint8_t> result(entry->decompressed_size);
			auto size_fact = decompress((uint8_t*)entry + entry->header_size, entry->compressed_size,
				result.data(), result.size());

			if (size_fact != entry->decompressed_size)
			{
				fprintf(stderr, "Decompressed PTX size does not match the header info: %zu != %zu\n",
					(size_t)size_fact, (size_t)entry->decompressed_size);
			}

			printf("%s\n", reinterpret_cast<const char*>(result.data()));

			seen_ptx = true;
		}

		if (!seen_ptx)
		{
			fprintf(stderr, "PTX entry is is not found in CUBIN, remember to include virtual architecture into NVCC compiler options\n");
			assert(seen_ptx);
			return;
		}
	}
	else
	{
		fprintf(stderr, "Unrecognized CUBIN magic: 0x%x\n", *(int*)fatCubin);
	}
}

