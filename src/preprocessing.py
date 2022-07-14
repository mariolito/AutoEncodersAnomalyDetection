import os
import sys
dir_store = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.append(os.path.join(os.path.dirname(__file__),  ".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=dir_store)
from sklearn.preprocessing import MinMaxScaler


def process_data():

    data = store_handler.read("creditcard.csv")
    features = [i for i in data if i not in ['Class']]
    data["Time"] = data["Time"].diff()
    data = data[~data["Time"].isnull()].reset_index(drop=True)


    data['Time'] = MinMaxScaler().fit_transform(data['Time'].values.reshape(-1,1))

    df_train = data.iloc[list(data[data["Class"] == 0].sample(frac=0.8).index)]
    df_test = data.iloc[list(set(data.index) - set(df_train.index))]

    X_train = df_train[features].values
    X_test_Legit = df_test[df_test["Class"] == 0][features].values
    X_test_Fraud = df_test[df_test["Class"] == 1][features].values

    store_handler.store(X_train, "X_train.p")
    store_handler.store(X_test_Legit, "X_test_Legit.p")
    store_handler.store(X_test_Fraud, "X_test_Fraud.p")
    print("Data processed and stored")
    print("Length X_train: {}".format(len(X_train)))
    print("Length X_test_Legit: {}".format(len(X_test_Legit)))
    print("Length X_test_Fraud: {}".format(len(X_test_Fraud)))


if __name__ == '__main__':
    process_data()
