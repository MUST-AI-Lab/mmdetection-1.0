
### Faster RCNN - Resnet-50


| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  | 23031  |  59.0  |   59.0   |
|   SENet   | Backbone |     16    |   1  | 23569  |  59.5  |   59.0   |
|   SENet   |    FPN   |     16    |   1  | 23205  |  60.0  |   59.0   |
|   SENet   |    ROI   |     16    |   1  | 23032  |  59.3  |   59.0   |
|   CBAM    | Backbone |     16    |   1  | 24069  |  58.4  |   59.0   |
|   CBAM    |    FPN   |     16    |   1  | 23378  |  59.1  |   59.0   |
|   CBAM    |    ROI   |     16    |   1  | 23031  |  59.0  |   59.0   |



### Faster RCNN - Resnet-101

| Attention | Position | Redection | Seed | Memory | mAP    | Baseline |
|-----------|----------|-----------|------|--------|--------|----------|
|     -     |     -    |     -     |   1  | 24955  |  60.3  |   60.3   |
|   SENet   | Backbone |     16    |   1  | 26069  |  59.8  |   60.3   |
|   SENet   |    FPN   |     16    |   1  | 25126  |  60.5  |   60.3   |
|   SENet   |    ROI   |     16    |   1  | -      |  -     |   60.3   |
|   CBAM    | Backbone |     16    |   1  | 27116  |  59.6  |   60.3   |
|   CBAM    |    FPN   |     16    |   1  | 25299  |  59.9  |   60.3   |
|   CBAM    |    ROI   |     16    |   1  | 24955  |  60.1  |   60.3   |

