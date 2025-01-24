/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void AllReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AllReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, op, rep, nranks, rank));
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

__global__ void reduce_kernel(float* recvbuf, float* chunkbuf1, float* chunkbuf2, size_t chunkSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < chunkSize) {
        recvbuf[idx] = chunkbuf1[idx] + chunkbuf2[idx];
    }
}

testResult_t AllReduceRunBuiltin(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
    return testSuccess;
}

testResult_t AllReduceRunRing(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    //NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);

    // Calculate type size manually (adjust for your ncclDataType_t)
    size_t typeSize;
    switch (type) {
        case ncclFloat: typeSize = sizeof(float); break;
        case ncclDouble: typeSize = sizeof(double); break;
        case ncclInt: typeSize = sizeof(int); break;
        case ncclInt64: typeSize = sizeof(int64_t); break;
        default: return testInternalError; // Unsupported type
    }

    // Divide data into chunks for ring communication
    size_t chunkSize = count / nranks;

    // Copy input buffer to output buffer
    //cudaMemcpyAsync(recvbuff, sendbuff, count * typeSize, cudaMemcpyDeviceToDevice, stream);

    int sendTo = (rank + 1) % nranks;
    int recvFrom = (rank - 1 + nranks) % nranks;

    // Step 1: Reduce-Scatter phase
    for (int step = 0; step < nranks - 1; ++step) {
        // Temporary buffer in recvbuff
        size_t recvOffset = ((rank - 1 - step + nranks) % nranks) * chunkSize;
        size_t sendOffset = ((rank - step + nranks) % nranks) * chunkSize;
        size_t tempBufOffset = ((rank - 2 - step + 2*nranks) % nranks) * chunkSize;

        if (rank==1) {
            printf("recvOffset %d, sendOffset %d, tempBufOffset %d\n", recvOffset, sendOffset, tempBufOffset);
        }
        // Send current chunk and receive the next chunk
        NCCLCHECK(ncclSend((char*)(step==0?sendbuff:recvbuff) + sendOffset * typeSize, chunkSize, type, sendTo, comm, stream));
        // recv to the place that is not used
        NCCLCHECK(ncclRecv((char*)recvbuff + tempBufOffset * typeSize, chunkSize, type, recvFrom, comm, stream));

        // Perform reduction (sum operation)
        reduce_kernel<<<(chunkSize + 255) / 256, 256, 0, stream>>>(
            (float*)recvbuff + recvOffset, (float*)sendbuff+recvOffset, (float*)recvbuff + tempBufOffset, chunkSize);
        //cudaStreamSynchronize(stream);
    }

    // Step 2: All-Gather phase
    for (int step = 0; step < nranks - 1; ++step) {
        size_t sendOffset = ((rank + 1 - step + nranks) % nranks) * chunkSize;
        size_t recvOffset = ((rank - step + nranks) % nranks) * chunkSize;

        // Send reduced chunk and receive the next chunk to gather full result
        NCCLCHECK(ncclSend((char*)recvbuff + sendOffset * typeSize, chunkSize, type, sendTo, comm, stream));
        NCCLCHECK(ncclRecv((char*)recvbuff + recvOffset * typeSize, chunkSize, type, recvFrom, comm, stream));
    }

    return testSuccess;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  AllReduceInitData,
  AllReduceGetBw,
  //AllReduceRunBuiltin
  AllReduceRunRing
};

void AllReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allReduceTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine allReduceEngine = {
  AllReduceGetBuffSize,
  AllReduceRunTest
};

#pragma weak ncclTestEngine=allReduceEngine
