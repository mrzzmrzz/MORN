from typing import List
import warnings
import random
import pandas as pd
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

warnings.filterwarnings("ignore")

# splitter function


def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold


# copy from xiong et al. attentivefp
def split(
    scaffolds_dict, smiles_tasks_df, tasks, frac, weights, sample_size, random_seed=0
):
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio

    # the minor proportion
    minor_ratio = 1 / weights[0][minor_class]
    optimal_count = frac * len(smiles_tasks_df)

    # count: current count of the  `return index list`
    # optimal_count: the optimal count for `return index list`
    # minor_count: the data in the `return index list`,
    # at the same time belong to minor class

    # only when the length of `return index list` in [0.9, 1.1] * optimal count
    # and the `minor count` in [0.9, 1.1] * optimal * minor_ratio, it will return
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) or (
        minor_count < minor_ratio * optimal_count * 0.9
        or minor_count > minor_ratio * optimal_count * 1.1
    ):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
        minor_count = len(
            smiles_tasks_df.iloc[index, :][smiles_tasks_df[tasks[0]] == minor_class]
        )
    #     print(random)
    return scaffold, index


def scaffold_randomized_spliting(
    smiles_tasks_df,
    tasks: List[str],
    random_seed=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    weights = []
    for i, task in enumerate(tasks):
        negative_df = smiles_tasks_df[smiles_tasks_df[task] == -1][["smiles", task]]
        positive_df = smiles_tasks_df[smiles_tasks_df[task] == 1][["smiles", task]]

        weights.append(
            [
                (positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0],
                (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0],
            ]
        )

    print("The dataset weights are", weights)
    print("generating scaffold......")

    scaffold_list = []
    all_scaffolds_dict = {}
    for index, smiles in enumerate(smiles_tasks_df["smiles"]):
        scaffold = generate_scaffold(smiles)
        scaffold_list.append(scaffold)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [index]
        else:
            all_scaffolds_dict[scaffold].append(index)

    samples_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(
        all_scaffolds_dict,
        smiles_tasks_df,
        tasks,
        frac_test,
        weights,
        samples_size,
        random_seed=random_seed,
    )

    training_scaffolds_dict = {
        x: all_scaffolds_dict[x]
        for x in all_scaffolds_dict.keys()
        if x not in test_scaffold
    }

    valid_scaffold, valid_index = split(
        training_scaffolds_dict,
        smiles_tasks_df,
        tasks,
        frac_valid,
        weights,
        samples_size,
        random_seed=random_seed,
    )

    training_scaffolds_dict = {
        x: training_scaffolds_dict[x]
        for x in training_scaffolds_dict.keys()
        if x not in valid_scaffold
    }
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele

    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

    trn_df = smiles_tasks_df.iloc[train_index, :]
    val_df = smiles_tasks_df.iloc[valid_index, :]
    test_df = smiles_tasks_df.iloc[test_index, :]

    return trn_df, val_df, test_df


def scaffold_split(
    smiles_tasks_df, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None
):
    smiles_list = smiles_tasks_df["smiles"]
    print("generating scaffold......")

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest, first sort the dictionary's
    # value list according the value from small -> large
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}

    # according to the tuple : (len(x[1]), x[1][0]) to sort the len reflects the
    # molecule size the x[1][0] reflects the smallest index in this molecular set
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []

    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    trn_df = smiles_tasks_df.iloc[train_idx, :]
    val_df = smiles_tasks_df.iloc[valid_idx, :]
    test_df = smiles_tasks_df.iloc[test_idx, :]

    return trn_df, val_df, test_df


def scaffold_random_split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0
):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    smiles_list = dataset["smiles"]
    rng = np.random.RandomState(seed)
    scaffolds = defaultdict(list)

    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_df = dataset.iloc[train_idx, :]
    valid_df = dataset.iloc[valid_idx, :]
    test_df = dataset.iloc[test_idx, :]

    return train_df, valid_df, test_df


def random_split(
    dataframe,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
    smiles_list=None,
):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    num_mols = len(dataframe)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[: int(frac_train * num_mols)]
    valid_idx = all_idx[
        int(frac_train * num_mols) : int(frac_valid * num_mols)
        + int(frac_train * num_mols)
    ]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols) :]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_df = dataframe.iloc[train_idx, :]
    valid_df = dataframe.iloc[valid_idx, :]
    test_df = dataframe.iloc[test_idx, :]

    return train_df, valid_df, test_df


def split_multi_label_containNan(df, tasks, seed):
    weights = []
    random_seed = seed
    for i, task in enumerate(tasks):
        # neg_df
        negative_df = df[df[task] == 0][["smiles", task]]
        # pos_df
        positive_df = df[df[task] == 1][["smiles", task]]

        negative_test = negative_df.sample(frac=1 / 10, random_state=random_seed)
        negative_valid = negative_df.drop(negative_test.index).sample(
            frac=1 / 9, random_state=random_seed
        )
        negative_train = negative_df.drop(negative_test.index).drop(
            negative_valid.index
        )

        positive_test = positive_df.sample(frac=1 / 10, random_state=random_seed)
        positive_valid = positive_df.drop(positive_test.index).sample(
            frac=1 / 9, random_state=random_seed
        )
        positive_train = positive_df.drop(positive_test.index).drop(
            positive_valid.index
        )

        weights.append(
            [
                (positive_train.shape[0] + negative_train.shape[0])
                / negative_train.shape[0],
                (positive_train.shape[0] + negative_train.shape[0])
                / positive_train.shape[0],
            ]
        )
        train_df_new = pd.concat([negative_train, positive_train])
        valid_df_new = pd.concat([negative_valid, positive_valid])
        test_df_new = pd.concat([negative_test, positive_test])

        if i == 0:
            train_df = train_df_new
            test_df = test_df_new
            valid_df = valid_df_new
        else:
            train_df = pd.merge(train_df, train_df_new, on="smiles", how="outer")
            test_df = pd.merge(test_df, test_df_new, on="smiles", how="outer")
            valid_df = pd.merge(valid_df, valid_df_new, on="smiles", how="outer")
    return train_df, valid_df, test_df, weights
