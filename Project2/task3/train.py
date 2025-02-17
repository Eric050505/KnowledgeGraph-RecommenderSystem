from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import util

validation_data = util.load_data("classification_validation_data.pkl")
validation_label = util.load_data("classification_validation_label.pkl")
validation_data = validation_data[:, 1:]
validation_label = validation_label[:, 1:].reshape(-1)

model = LogisticRegression()
selector = RFE(model, n_features_to_select=30)
X = validation_data
y = validation_label
selector = selector.fit(X, y)

selected_features = selector.support_
selected_features = selected_features.astype(int)
util.save_data("mask_code.pkl", selected_features)

print(util.load_data("mask_code.pkl"))
