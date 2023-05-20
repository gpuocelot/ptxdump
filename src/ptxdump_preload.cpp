#include <cassert>
#include <cstdio>
#include <iostream>
#include <dlfcn.h>

#include "ptxdump.h"

extern "C"
void** __cudaRegisterFatBinary(void *fatCubin) 
{
	static const char* libfname = "/usr/local/cuda/lib64/libcudart.so";
	void *cudart = dlopen(libfname, RTLD_NOW);
	if (!cudart)
	{
		fprintf(stderr, "Cannot dlopen() \"%s\": \"%s\"\n", libfname, dlerror());
		assert(cudart);
	}

	static const char* fname = "__cudaRegisterFatBinary";
	auto f = (void** (*)(void *))dlsym(cudart, fname);
	if (!f)
	{
		fprintf(stderr, "Cannot dlsym() \"%s\": \"%s\"\n", fname, dlerror());
		dlclose(cudart);
		assert(f);
	}
	else
		dlclose(cudart);

	ptxdump(fatCubin);
	return f(fatCubin);
}
