#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <ctime>
#include <vector>

#include "parallelSort.h"

using namespace std;

void printArr(const char* arrName, int *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %d %d %d %d\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void printArr(const char* arrName, float *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %f %f %f %f\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void randomSample(float *data, size_t dataSize, float *sample, size_t sampleSize) {
  for (size_t i=0; i<sampleSize; i++) {
    sample[i] = data[rand()%dataSize];
  }
}

int last = 0;

int generateRandB(int rank) {
  srand(last + time(NULL) + rank);
  int mNum = rand() % 1000;
  int tNum = rand() % 1000;
  int dNum = rand() % 1000;
  
  last = mNum * 1000000 + tNum * 1000 + dNum;
  return last;
}

#define ROOT 0

void parallelSort(float *data, float *&sortedData, int procs, int procId, size_t dataSize, size_t &localSize) {
  // Implement parallel sort algorithm as described in assignment 3
  // handout. 
  // Input:
  //  data[]: input arrays of unsorted data, *distributed* across p processors
  //          note that data[] array on each process contains *a fraction* of all data
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //  procs: total number of processes
  //  procId: id of this process
  //  dataSize: aggregated size of *all* data[] arrays
  //  localSize: size of data[] array on *this* process (as input)
  //             size of sortedData[] array on *this* process (as output)
  //
  //
  // Step 1: Choosing Pivots to Define Buckets
  // Step 2: Bucketing Elements of the Input Array
  // Step 3: Redistributing Elements
  // Step 5: Final Local Sort
  // ***********************************************************************

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != procs || rank < 0 || rank >= procs)
    assert(false);

  if (size == 1) {
    sortedData = (float *)malloc(dataSize * sizeof(float));
    memcpy(sortedData, data, dataSize * sizeof(float)); 
    sort(&sortedData[0], sortedData + dataSize);
    return;
  }
  
  int numPivots = procs - 1;
  float pivots[numPivots]; 
  int numLocalSamples = 12 * log((double)dataSize);
  float localSamples[numLocalSamples];
  int S = procs * numLocalSamples;
  float samples[rank == ROOT ? S : 0];
  
  // Get local samples from local data
  for (int i = 0; i < numLocalSamples; i++) {
    int randomIndex = generateRandB(rank) % localSize;
    localSamples[i] = data[randomIndex];
  } 
  
  // root gathers local samples
  MPI_Gather(localSamples, numLocalSamples, MPI_FLOAT,
             samples, numLocalSamples, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  if (rank == ROOT) {
    // Sort the sample data array
    sort(&samples[0], samples + S);

    // pick out the pivots from samples
    for (int i = 0; i < numPivots; i++) {
      pivots[i] = samples[((i+1) * S) / procs];
    }
  }
  
  // send the pivots from root to all other processes
  MPI_Bcast(pivots, numPivots, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
 
  // partition the local data into buckets
  std::vector<float> buckets[procs];
  for (int i = 0; i < localSize; i++) {
    if (data[i] >= pivots[numPivots - 1]) {
      buckets[procs - 1].push_back(data[i]);
      continue;
    }
    for (int j = 0; j < numPivots; j++) {
      if (data[i] < pivots[j]) {
        buckets[j].push_back(data[i]);
        break;
      }
    }
  }
  
  // prepare to send local data to other buckets
  int outgoingCount[procs];
  int outDisplacement[procs];
  for (int i = 0; i < procs; i ++) {
    outgoingCount[i] = buckets[i].size();
    outDisplacement[i] = (i == 0) ? 0
        : outDisplacement[i-1] + outgoingCount[i-1];
    for (int j = 0; j < buckets[i].size(); j++) {
      data[outDisplacement[i] + j] = buckets[i][j];
    }
  }
   
  // get the number of elements other proceses is going to send  
  int incomingCount[procs];
  MPI_Alltoall(outgoingCount, 1, MPI_INT, 
               incomingCount, 1, MPI_INT, MPI_COMM_WORLD);
  
  // get the total number of receiving data
  int inDisplacement[procs];
  int M = 0; 
  for (int i = 0; i < procs; i++) {
    inDisplacement[i] = M;
    M += incomingCount[i];
  }
   
  // shuffle local data
  sortedData = (float *)malloc(M * sizeof(float));
  MPI_Alltoallv(data, outgoingCount, outDisplacement, MPI_FLOAT, 
                sortedData, incomingCount, inDisplacement,
                MPI_FLOAT, MPI_COMM_WORLD);

  // local sort
  sort(&sortedData[0], sortedData + M);
  localSize = M;
  return;
}

