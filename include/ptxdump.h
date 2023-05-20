#ifndef PTXDUMP_H
#define PTXDUMP_H

#define __cudaFatMAGIC     0x1ee55a01
#define __cudaFatMAGIC2    0x466243b1

#define COMPRESSED_PTX     0x0000000000001000LL

enum FatBin2EntryType {
	FATBIN_2_PTX = 0x1
};

typedef struct {
    char* gpuProfileName;            
    char* ptx;
} __cudaFatPtxEntry;

typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;

typedef struct __cudaFatDebugEntryRec {
    char* gpuProfileName;            
    char* debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int size;
} __cudaFatDebugEntry;

typedef struct {
    char* name;
} __cudaFatSymbol;

typedef struct __cudaFatElfEntryRec {
    char* gpuProfileName;            
    char* elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int size;
} __cudaFatElfEntry;

typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
    __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;

typedef struct __cudaFatCudaBinaryRec2 {
    int magic;
    int version;
    const unsigned long long* fatbinData;
    char* f;
} __cudaFatCudaBinary2;

void ptxdump(const void *fatCubin);

#endif // PTXDUMP_H

