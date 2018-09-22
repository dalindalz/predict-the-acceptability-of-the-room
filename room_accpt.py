import pandas as pd
from Orange.data import Table, Domain
import random
from Orange.classification import SklTreeLearner
from Orange.evaluation import CrossValidation, scoring


# Read data from csv and return the data
def read_data_from_csv():
    dataset = pd.read_csv('assg1_room.csv')

    return dataset

# replace all na with mean of the feature
def replace_na_with_mean(column_name, data,mean):
    data[[column_name]] = data[[column_name]].fillna(mean)
    return data

# Read data from csv and return the data in table format in orange programming

def read_csv_table():
    data_table = Table.from_file('assign_clean.csv')
    return data_table

# Prepare class variables and feature variable

def prepro_class_variable(data_table):
    feature_variable = data_table[:,
                       ['Renting', 'Bills', 'Tenants', 'Rooms', 'Size', 'Furnished', 'Bathrooms']].domain.variables
    class_label_variable = data_table[:, ['accept']].domain.variables
    room_domain = Domain(feature_variable, class_label_variable)
    data_table = Table.from_table(domain=room_domain, source=data_table)
    return data_table

# Prepare training data

def prepare_train_test(data_table):
    random.shuffle(data_table)
    value = int(0.8 * len(data_table))
    train_dataset = data_table[:value]
    test_dataset = data_table[value:]
    return train_dataset, test_dataset

# Decision tree with orange programming

def build_decision_tree(max_leaf=None):
    tree_learner = SklTreeLearner(max_leaf_nodes=max_leaf)
    decision_tree = tree_learner(train_dataset)
    print(decision_tree)
    return tree_learner, decision_tree

# Predict accuarcy of the model

def predict_cal_acc(test_dataset, data_table, tree_learner):
    y_pred = decision_tree(test_dataset)
    eval_results = CrossValidation(data_table, [tree_learner], k=10)
    print("Accuracy: {:.3f}".format(scoring.CA(eval_results)[0]))
    print("AUC: {:.3f}".format(scoring.AUC(eval_results)[0]))


if __name__ == "__main__":
    data = read_data_from_csv()

    mean = data[['Tenants']].mean(skipna=True)
    data_clean = replace_na_with_mean('Tenants', data, mean)
    print(data[['Tenants']].isna().sum())
    data.to_csv('assign_clean.csv', encoding='utf-8', index=False, na_rep="N/A")

    data_table = read_csv_table()
    data_table = prepro_class_variable(data_table)
    train_dataset, test_dataset = prepare_train_test(data_table)

    tree_learner, decision_tree = build_decision_tree()

    predict_cal_acc(test_dataset, data_table, tree_learner)

    tree_learner, decision_tree = build_decision_tree(4)

    predict_cal_acc(test_dataset, data_table, tree_learner)
