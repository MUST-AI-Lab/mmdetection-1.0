## SE at Faster-RCNN after ROI align 
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | Reduction | Bias  | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|-------|---------|--------|--------|----------|
| R-50     | 4         | True  | 5052    | 71.1   | 70.8   | 63.1     |
| R-50     | 16        | True  | 5051    | 71.0   | 70.8   | 63.1     |
| R-50     | 32        | True  | 5051    | 70.9   | 70.8   | 63.1     |
| R-101    | 4         | True  | 6363    | 72.8   | 72.3   | 63.1     |
| R-101    | 16        | True  | 6360    | 72.9   | 72.3   | 63.1     |
| R-101    | 32        | True  | 6363    | 73.1   | 72.3   | 63.1     |



## SE at Faster-RCNN after FPN 
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | Reduction | Bias  | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|-------|---------|--------|--------|----------|
| R-50     | 4         | TRUE  | 5152    | 71.1   | 70.8   | 63.1     |
| R-50     | 16        | TRUE  | 5152    | 71.1   | 70.8   | 63.1     |
| R-50     | 32        | TRUE  | 5148    | 71.0   | 70.8   | 63.1     |
| R-101    | 4         | TRUE  | 6467    | 72.8   | 72.3   | 63.1     |
| R-101    | 16        | TRUE  | 6468    | 72.8   | 72.3   | 63.1     |
| R-101    | 32        | TRUE  | 6467    | 73.2   | 72.3   | 63.1     |

## SE at Faster-RCNN after ROI and FPN
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | ROI Reduction | ROIBias  | FPN Reduction | FPN Bias  | Mem(MB) | box AP | Ours   | Baseline |
|----------|---------------|----------|---------------|-----------|---------|--------|--------|----------|
| R-50     | 4             | TRUE     | 4             | TRUE      | 5052    | 71.3   | 70.8   | 63.1     |
| R-50     | 16            | TRUE     | 16            | TRUE      | 5051    | 71.2   | 70.8   | 63.1     |
| R-50     | 32            | TRUE     | 32            | TRUE      | 5051    | 70.7   | 70.8   | 63.1     |
| R-101    | 4             | TRUE     | 4             | TRUE      | 6466    | 72.6   | 70.8   | 63.1     |
| R-101    | 16            | TRUE     | 16            | TRUE      | 6462    | 72.4   | 70.8   | 63.1     |
| R-101    | 32            | TRUE     | 32            | TRUE      | 6468    | 72.6   | 70.8   | 63.1     |


## SE at Faster-RCNN at backbone 
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | Reduction | Bias  | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|-------|---------|--------|--------|----------|
| R-50     | 4         | TRUE  | 5509    | 70.0   | 70.8   | 63.1     |
| R-50     | 16        | TRUE  | 5509    | 70.3   | 70.8   | 63.1     |
| R-50     | 32        | TRUE  | 5381    | 70.6   | 70.8   | 63.1     |
| R-101    | 4         | TRUE  | 7298    | 72.1   | 72.3   | 63.1     |
| R-101    | 16        | TRUE  | 7085    | 72.3   | 72.3   | 63.1     |
| R-101    | 32        | TRUE  | 7049    | 72.5   | 72.3   | 63.1     |



## SE at Faster-RCNN at backbone (pretrained)
Train Method = trainval,  Seed = 1,   Determinitic = False 

| Backbone | Reduction | Bias  | Mem(MB) | box AP | Ours   | Baseline |
|----------|-----------|-------|---------|--------|--------|----------|
| R-50     | 4         | TRUE  | 5403    | 72.2   | 70.8   | 63.1     |
| R-50     | 16        | TRUE  | 5291    | 74.5   | 70.8   | 63.1     |
| R-50     | 32        | TRUE  | 5270    | 72.2   | 70.8   | 63.1     |
