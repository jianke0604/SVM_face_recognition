import numpy as np
from collections import Counter

def concat_and_replace(arr1, arr2, arr3, arr4, arr5):
    # 将5个np array拼接成一个5*1000的array
    concatenated_array = np.vstack((arr1, arr2, arr3, arr4, arr5))

    # 统计每一列中出现次数最多的元素
    most_common_elements = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=concatenated_array)

    # 替换第0维为最常出现的元素
    # concatenated_array[0, :] = most_common_elements

    return most_common_elements

# 示例用法
arr1 = np.random.randint(0, 7, size=10)
print(arr1)
arr2 = np.random.randint(0, 7, size=10)
print(arr2)
arr3 = np.random.randint(0, 7, size=10)
print(arr3)
arr4 = np.random.randint(0, 7, size=10)
print(arr4)
arr5 = np.random.randint(0, 7, size=10)
print(arr5)
result_array = concat_and_replace(arr1, arr2, arr3, arr4, arr5)
print(result_array)
