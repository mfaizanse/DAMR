from kinfu_cv import KinfuPlarr
import numpy as np


kfp = KinfuPlarr(640, 480, 4, 4, 4, 4, 4)

# print(kfp.getTestValue(7.5))
c = np.array([1,2,3,4]).tolist()
#
# print(type(b.tolist()))

b = [1,2,3]
print(type(b))
print(type(c))



print(kfp.getTestValue(c))