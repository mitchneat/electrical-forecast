import numpy as np
import pandas as pd
import random
task1 = np.random.randint(0, 5*60*60, size = 10000000)
task2 = np.random.randint(0, 5*60*60, size = 10000000)
data = pd.DataFrame(
    {'task1_start': task1,
     'task2_start': task2,
    })
data['overlap'] = np.where(np.abs(data.task1_start - data.task2_start) <=3600, 1, 0)
print(data['overlap'].mean())