usage

```bash
mkdir build
cd build
make -j
./merge_NND_streaming datasetname data_file query_file M ef_construction // if data from bvecs(num,dim,num*dim*uint_8) add parameter bvecs
```