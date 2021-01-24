
### FCOS - Resnet-50


| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  | 10239  |  57.8  |   57.8   |
|   SENet   | Backbone |     16    |   1  | 10773  |  56.2  |   57.8   |
|   SENet   |    FPN   |     16    |   1  | 10280  |  58.1  |   57.8   |
|   SENet   |   Bbox   |     16    |   1  | 10326  |  56.9  |   57.8   |
|   CBAM    | Backbone |     16    |   1  | 11273  |  55.2  |   57.8   |
|   CBAM    |    FPN   |     16    |   1  | 10323  |  57.4  |   57.8   |
|   CBAM    |   Bbox   |     16    |   1  | 10413  |  57.3  |   57.8   |



### FCOS - Resnet-101

| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  | 12159  |  58.4  |   58.4   |
|  SENet    | Backbone |     16    |   1  | 13273  |  56.2  |   58.4   |
|  SENet    |    FPN   |     16    |   1  | 12202  |  58.1  |   58.4   |
|  SENet    |   Bbox   |     16    |   1  | 12245  |  58.4  |   58.4   |
|   CBAM    | Backbone |     16    |   1  | 14320  |  55.9  |   58.4   |
|   CBAM    |    FPN   |     16    |   1  | 12246  |  58.2  |   58.4   |
|   CBAM    |    Bbox  |     16    |   1  | 12332  |  58.4  |   58.4   |
