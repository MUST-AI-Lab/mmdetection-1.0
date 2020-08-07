
### FCOS - Resnet-50


| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  |  705   |  74.8  |   74.8   |
|   SENet   | Backbone |     4     |   1  |  882   |  74.0  |   74.8   |
|   SENet   | Backbone |     16    |   1  |  772   |  73.7  |   74.8   |
|   SENet   | Backbone |     32    |   1  |  752   |  73.4  |   74.8   |
|   SENet   |    FPN   |     4     |   1  |  705   |  75.2  |   74.8   |
|   SENet   |    FPN   |     16    |   1  |  704   |  75.2  |   74.8   |
|   SENet   |    FPN   |     32    |   1  |  704   |  74.9  |   74.8   |
|   SENet   |   Bbox   |     4     |   1  |  707   |  74.0  |   74.8   |
|   SENet   |   Bbox   |     16    |   1  |  707   |  74.9  |   74.8   |
|   SENet   |   Bbox   |     32    |   1  |  707   |  75.1  |   74.8   |
|   CBAM    | Backbone |     4     |   1  |  919   |  72.2  |   74.8   |
|   CBAM    | Backbone |     16    |   1  |  804   |  72.0  |   74.8   |
|   CBAM    | Backbone |     32    |   1  |  752   |  73.5  |   74.8   |
|   CBAM    |    FPN   |     4     |   1  |  710   |  74.8  |   74.8   |
|   CBAM    |    FPN   |     16    |   1  |  710   |  75.4  |   74.8   |
|   CBAM    |    FPN   |     32    |   1  |  709   |  73.8  |   74.8   |
|   CBAM    |   Bbox   |     4     |   1  |  713   |  75.5  |   74.8   |
|   CBAM    |   Bbox   |     16    |   1  |  712   |  74.5  |   74.8   |
|   CBAM    |   Bbox   |     32    |   1  |  712   |  75.3  |   74.8   |



### FCOS - Resnet-101

| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  |  1102  |  74.7  |   74.7   |
|  SENet    | Backbone |     4     |   1  |  1450  |  74.5  |   74.7   |
|  SENet    | Backbone |     16    |   1  |  1232  |  74.8  |   74.7   |
|  SENet    | Backbone |     32    |   1  |  1198  |  74.4  |   74.7   |
|  SENet    |    FPN   |     4     |   1  |  1107  |  75.3  |   74.7   |
|  SENet    |    FPN   |     16    |   1  |  1106  |  75.0  |   74.7   |
|  SENet    |    FPN   |     32    |   1  |  1106  |  75.2  |   74.7   |
|  SENet    |   Bbox   |     4     |   1  |  1113  |  75.1  |   74.7   |
|  SENet    |   Bbox   |     16    |   1  |  1109  |  74.9  |   74.7   |
|  SENet    |   Bbox   |     32    |   1  |  1103  |  75.0  |   74.7   |
|   CBAM    | Backbone |     4     |   1  |  1515  |  72.6  |   74.7   |
|   CBAM    | Backbone |     16    |   1  |  1302  |  73.0  |   74.7   |
|   CBAM    | Backbone |     32    |   1  |  1195  |  74.2  |   74.7   |
|   CBAM    |    FPN   |     4     |   1  |  1111  |  75.6  |   74.7   |
|   CBAM    |    FPN   |     16    |   1  |  1110  |  75.5  |   74.7   |
|   CBAM    |    FPN   |     32    |   1  |  1110  |  74.6  |   74.7   |
|   CBAM    |    Bbox  |     4     |   1  |  1112  |  75.7  |   74.7   |
|   CBAM    |    Bbox  |     16    |   1  |  1112  |  75.4  |   74.7   |
|   CBAM    |    Bbox  |     32    |   1  |  1111  |  75.4  |   74.7   |
