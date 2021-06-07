# Optimization

## Operations
### Allocate host memory as pinned (default is pageable)
 It can be read or written with much higher bandwidth than pageable memory obtained with functions such as malloc()
```
cudaHostAlloc((void**)&h_data,data_size,cudaHostAllocDefault);
```
instead 
```
malloc
```

### Copying result from device to host asynchronously
```  
cudaMemcpyAsync(h_calc_classes + offset , d_calc_classes+offset, calc_classes_size, cudaMemcpyDeviceToHost);
```


## Result
|Version 1|Version 2|
|---------|---------|
|26.552 s  |26.474 s  |
|26.526 s  |26.406 s  |
|26.786 s |26.587 s  |