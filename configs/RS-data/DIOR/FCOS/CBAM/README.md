## CBAM at FCOS in bbox_head 
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | cls-Reduction | reg-reduction | cls-Bias | reg-Bias | Mem(MB) | box AP | Ours   | Baseline |
|----------|---------------|---------------|----------|----------|---------|--------|--------|----------|
| R-50     | 4             | 4             | True     | True     | 3422    | 70.5   | 70.8   | 63.1     |
| R-50     | 16            | 16            | True     | True     | 3421    | 70.9   | 70.8   | 63.1     |
| R-50     | 32            | 32            | True     | True     | 3421    | 70.3   | 70.8   | 63.1     |
| R-101    | 4             | 4             | True     | True     | 4731    | 71.0   | 72.7   | 63.1     |
| R-101    | 16            | 16            | True     | True     | 4731    | 71.1   | 72.7   | 63.1     |
| R-101    | 32            | 32            | True     | True     | 4731    | 71.7   | 72.7   | 63.1     |


## CBAM at FCOS in FPN 
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | Reduction | Bias | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|------|---------|--------|--------|----------|
| R-50     | 4         | True | 3368    | 70.8   | 70.8   | 63.1     |
| R-50     | 16        | True | 3367    | 71.3   | 70.8   | 63.1     |
| R-50     | 32        | True | 3367    | 71.7   | 70.8   | 63.1     |
| R-101    | 4         | True | 4683    | 72.4   | 72.7   | 65.1     |
| R-101    | 16        | True | 4683    | 73.1   | 72.7   | 65.1     |
| R-101    | 32        | True | 4683    | 72.2   | 72.7   | 65.1     |



## CBAM at FCOS in Backbone
Train Method = trainval,  Seed = 1,   Determinitic = False 


| Backbone | Reduction | Bias | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|------|---------|--------|--------|----------|
| R-50     | 4         | True | 4086    | 70.5   | 70.8   | 63.1     |
| R-50     | 16        | True | 3975    | 70.0   | 70.8   | 63.1     |
| R-50     | 32        | True | 3954    | 69.7   | 70.8   | 63.1     |
| R-101    | 4         | True | 6220    | 71.4   | 72.7   | 65.1     |
| R-101    | 16        | True | 6001    | 71.0   | 72.7   | 65.1     |
| R-101    | 32        | True | 5971    | 71.1   | 72.7   | 65.1     |
