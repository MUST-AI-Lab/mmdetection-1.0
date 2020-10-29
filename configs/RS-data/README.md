# Faster RCNN

### DIOR FPN Seed-1 

| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-50 |        -      |      -    |  5050  |  70.9  |   70.9   |
|   SENet   |Resnet-50 |        N      |    4      |  5150  |  71.1  |   70.9   |
|   SENet   |Resnet-50 |        Y      |    4      |  5048  |  70.8  |   70.9   |
|   SENet   |Resnet-50 |        N      |    16     |  5396  |  70.4  |   70.9   |
|   SENet   |Resnet-50 |        Y      |    16     |  5048  |  70.9  |   70.9   |
|   SENet   |Resnet-50 |        N      |    32     |  5152  |  70.8  |   70.9   |
|   CBAM    |Resnet-50 |        N      |    4      |  5263  |  71.2  |   70.9   |
|   CBAM    |Resnet-50 |        N      |    16     |  5260  |  71.3  |   70.9   |
|   CBAM    |Resnet-50 |        Y      |    16     |  5047  |  70.9  |   70.9   |
|   CBAM    |Resnet-50 |        N      |    32     |  5260  |  72.0  |   70.9   |
|   SKNet   |Resnet-50 |        N      |    16     |  5570  |  55.3  |   70.9   |
|   BAM     |Resnet-50 |        N      |    16     |  5288  |  71.5  |   70.9   |

| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-101|        -      |      -    |  6361  |  72.4  |   72.4   |
|   SENet   |Resnet-101|        N      |    4      |  6468  |  72.8  |   72.4   |
|   SENet   |Resnet-101|        Y      |    4      |  6362  |  72.8  |   72.4   |
|   SENet   |Resnet-101|        N      |    16     |  6467  |  72.8  |   72.4   |
|   SENet   |Resnet-101|        N      |    32     |  6467  |  73.1  |   72.4   |
|   CBAM    |Resnet-101|        N      |    4      |  6572  |  72.8  |   72.4   |
|   CBAM    |Resnet-101|        N      |    16     |  6574  |  72.8  |   72.4   |
|   CBAM    |Resnet-101|        N      |    32     |  6574  |  72.8  |   72.4   |
|   SKNet   |Resnet-101|        N      |    16     |  6882  |  57.3  |   72.4   |
|   BAM     |Resnet-101|        N      |    16     |  6598  |  73.0  |   72.4   |


### DOTA FPN seed-1 V-100(32GB)
| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-50 |        -      |      -    |  23031 |  59.0  |   59.0   |
|   SENet   |Resnet-50 |        N      |    16     |  23205 |  60.0  |   59.0   |
|   CBAM    |Resnet-50 |        N      |    16     |  23378 |  59.1  |   59.0   |

| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-101|        -      |      -    |  24955 |  60.3  |   60.3   |
|   SENet   |Resnet-101|        N      |    16     |  25126 |  60.5  |   60.3   |
|   CBAM    |Resnet-101|        N      |    16     |  25299 |  59.9  |   60.3   |

### HRRSD FPN seed-1

| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-50 |        -      |      -    |  946   |  74.2  |   74.2   |
|   SENet   |Resnet-50 |        N      |    4      |  960   |  74.4  |   74.2   |
|   SENet   |Resnet-50 |        N      |    16     |  959   |  74.8  |   74.2   |
|   SENet   |Resnet-50 |        N      |    32     |  959   |  74.7  |   74.2   |
|   CBAM    |Resnet-50 |        N      |    4      |  967   |  74.6  |   74.2   |
|   CBAM    |Resnet-50 |        N      |    16     |  969   |  74.4  |   74.2   |
|   CBAM    |Resnet-50 |        N      |    32     |  967   |  74.4  |   74.2   |
|  NonLocal |Resnet-50 |        N      |  Gaussian |  1440  |  74.1  |   74.2   |
|  NonLocal |Resnet-50 |        Y      |  Gaussian |  1066  |  74.1  |   74.2   |
|  NonLocal |Resnet-50 |        N      |Em-Gaussian|  1443  |  74.5  |   74.2   |
|  NonLocal |Resnet-50 |        Y      |Em-Gaussian|  1071  |  74.3  |   74.2   |
|  NonLocal |Resnet-50 |        N      | DotProduct|  1113  |  74.2  |   74.2   |
|  NonLocal |Resnet-50 |        Y      | DotProduct|  1071  |  74.2  |   74.2   |


| Attention | Backbone | Shared params | Redection | Memory | mAP    | Baseline |
|-----------|----------|---------------|-----------|--------|--------|----------|
|     -     |Resnet-101|        -      |      -    |  1345  |  74.2  |   74.2   |
|   SENet   |Resnet-101|        N      |    4      |  1358  |  74.7  |   74.2   |
|   SENet   |Resnet-101|        N      |    16     |  1358  |  74.3  |   74.2   |
|   SENet   |Resnet-101|        N      |    32     |  1358  |  74.5  |   74.2   |
|   CBAM    |Resnet-101|        N      |    4      |  1369  |  74.6  |   74.2   |
|   CBAM    |Resnet-101|        N      |    16     |  1372  |  74.7  |   74.2   |
|   CBAM    |Resnet-101|        N      |    32     |  1372  |  74.8  |   74.2   |
|  NonLocal |Resnet-101|        N      |  Gaussian |  1844  |  74.2  |   74.2   |
|  NonLocal |Resnet-101|        Y      |  Gaussian |  1464  |  74.1  |   74.2   |
|  NonLocal |Resnet-101|        N      |Em-Gaussian|  1841  |  74.3  |   74.2   |
|  NonLocal |Resnet-101|        T      |Em-Gaussian|  1471  |  74.6  |   74.2   |
|  NonLocal |Resnet-101|        N      | DotProduct|  1519  |  74.0  |   74.2   |
|  NonLocal |Resnet-101|        Y      | DotProduct|  1471  |  74.4  |   74.2   |
