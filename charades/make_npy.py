import numpy as np
# 试着以float32读入
feat = np.fromfile('FeatureData/i3d_rgb_lgi/feature.bin', dtype=np.float32)
print(feat.shape)