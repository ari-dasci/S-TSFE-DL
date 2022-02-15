import numpy as np
import pandas as pd

def readDataset(route, route_names):
    """
    This function reads the KDD Cup 99 dataset.

    Parameters
    ----------
    route : str
        Route of the dataset.
    route_names : str
        Route of the names of the variables.

    Returns
    -------
    data : array-like
        Array with the read data.
    attack_types : array-like
        Array with the possible attacks.
    attack_class : array-like
        Labels of the data.
    """
    data = pd.read_csv(route, sep=",")
    names_file = open(route_names, "r")
    variable_names = []
    attack_types = []
    first_line = True
    for line in names_file:
        if first_line:
            attack_types = line[:-1].split(",")
            first_line = False
        else:
            variable_names.append(line.split(":")[0])
    variable_names.append("attack_type")
    data.columns = variable_names
    data["attack_type"] = data["attack_type"].str.replace(r'.$', '')
    data = data.astype({"duration": "float",
                        "protocol_type": "str",
                        "service": "str",
                        "flag": "str",
                        "src_bytes": "float",
                        "dst_bytes": "float",
                        "land": "str",
                        "wrong_fragment": "float",
                        "urgent": "float",
                        "hot": "float",
                        "num_failed_logins": "float",
                        "logged_in": "str",
                        "num_compromised": "float",
                        "root_shell": "float",
                        "su_attempted": "float",
                        "num_root": "float",
                        "num_file_creations": "float",
                        "num_shells": "float",
                        "num_access_files": "float",
                        "num_outbound_cmds": "float",
                        "is_host_login": "str",
                        "is_guest_login": "str",
                        "count": "float",
                        "srv_count": "float",
                        "serror_rate": "float",
                        "srv_serror_rate": "float",
                        "rerror_rate": "float",
                        "srv_rerror_rate": "float",
                        "same_srv_rate": "float",
                        "diff_srv_rate": "float",
                        "srv_diff_host_rate": "float",
                        "dst_host_count": "float",
                        "dst_host_srv_count": "float",
                        "dst_host_same_srv_rate": "float",
                        "dst_host_diff_srv_rate": "float",
                        "dst_host_same_src_port_rate": "float",
                        "dst_host_srv_diff_host_rate": "float",
                        "dst_host_serror_rate": "float",
                        "dst_host_srv_serror_rate": "float",
                        "dst_host_rerror_rate": "float",
                        "dst_host_srv_rerror_rate": "float",
                    })
    data = oneHotEncode(data, ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login",])
    attack_class = np.array(data["attack_type"])
    data = data.drop("attack_type", axis=1)

    attack_types[-1] = attack_types[-1][:-1]
    return np.array(data), attack_types, attack_class

def oneHotEncode(data, columns):
    """
    This function performs one hot encoding over a set of columns.

    Parameters
    ----------
    data : array-like
        Data for performing one-hot encoding.
    columns : str list
        Names of the columns to perform one-hot encoding.

    Returns
    -------
    data : array-like
        Dataset modified with one-hot encoding.
    """
    for col in columns:
        one_hot = pd.get_dummies(data[col], prefix=col)
        data = data.drop(col, axis=1)
        data = data.join(one_hot)
    return data

def sliceDataset(data, batch_size, npred):
    """
    This function slices the dataset in batches of batch_size size to
    predict npred instances next.

    Parameters
    ----------
    data : array-like
        Data for to perform the slicing.
    batch_size : int
        Batch size of the slice to obtain.
    npred : int
        Number of instances to predict next to the batch.

    Returns
    -------
    Xs : array-like
        Data sliced.
    Ys : array-like
        Instances to predict from the data slice.
    """
    Xs = []
    Ys = []
    cont = 0
    while cont+batch_size+npred<len(data):
        Xs.append(data[cont:cont+batch_size].astype("float32"))
        Ys.append(data[cont+batch_size:cont+batch_size+npred].astype("float32"))
        cont+=npred
    return np.array(Xs), np.array(Ys)

def computeScore(y_true, y_pred):
    """
    This function computes the mean error obtained in a prediction.

    Parameters
    ----------
    y_true : array-like
        Real values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    score : array-like
        Error obtained from the prediction using the real values.
    """
    return np.sum(np.absolute(y_true-y_pred), axis=1)/y_true.shape[1]
