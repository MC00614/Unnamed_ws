# from sklearn.preprocessing import OneHotEncoder
# import numpy as np

# items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
# items = np.array(items).reshape(-1,1)

# oh_encoder = OneHotEncoder()
# oh_encoder.fit(items)
# oh_label = oh_encoder.transform(items)

# print(oh_label.toarray())

import pandas as pd

df = pd.DataFrame({'items':['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
pd.get_dummies(df)
print(df)