import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from variograd_utils import load_hdf5

# def load_predictions_arrays(filepaths, group, model_name, n_subjects, n_vertex, n_threads=10):
#     df_dict = load_sl_preds(filepaths, group, model_name, n_threads=n_threads)

#     preds_shape = [n_subjects, n_vertex]
#     first_item = next(iter(df_dict.values()))["pred"]
#     if first_item.ndim == 2: preds_shape.append(first_item.shape[1])

#     observed = np.zeros([n_subjects, n_vertex])
#     predicted = np.zeros(preds_shape)

#     for sl, df in df_dict.items():
#         observed[df["subject"], df["vertex"]] = df["true"]
#         predicted[df["subject"], df["vertex"]] = df["pred"]
#     return observed, predicted

def load_predictions_arrays(filepaths, group, model_name, n_subjects, n_vertex, n_threads=10):
    df_dict = load_sl_preds(filepaths, group, model_name, n_threads=n_threads)

    preds_shape = [n_subjects, n_vertex]
    first_item = next(iter(df_dict.values()))["pred"]
    if first_item.ndim == 2:
        preds_shape.append(100) #first_item.shape[1])
        
    observed = np.full([n_subjects, n_vertex], np.nan, dtype=float)
    predicted = np.full(preds_shape, np.nan, dtype=float)
    standard_dev = np.full([n_subjects, n_vertex], np.nan, dtype=float)

    for sl, df in df_dict.items():
        if first_item.ndim == 2: df["pred"] = df["pred"][:, :100]
        observed[df["subject"], df["vertex"]] = df["true"]
        predicted[df["subject"], df["vertex"]] = df["pred"]
        standard_dev[df["subject"], df["vertex"]] = df["sd"]
    return observed, predicted, standard_dev


def load_sl_preds(filepaths, group, model_name, n_threads=10):
    # create the thread pool
    with ThreadPoolExecutor(n_threads) as executor:
        # submit all tasks
        futures = [executor.submit(_load_sl_preds_df, p, group, model_name) for p in filepaths]
        # process all results
        results = {}
        for future in as_completed(futures):
            # open the file and load the data
            sl, df = future.result()
            results[sl] = df
        return results


def _load_sl_preds_df(filepath, group, model_name):
    data = load_hdf5(filepath)
    df = _preds_df(data, group, model_name)
    df["sd"] = pd.Series(df["true"]).groupby(df["subject"]).transform("std")
    df = {k: np.array(v) for k, v in df.items()}
    return data["searchlight_id"], df


def _preds_df(results_dict, group, model_name):
    df = {
        "subject": results_dict[group]["subject_indices"] - 1,
        "vertex": results_dict[group]["vertex_indices"] - 1,
        "true": np.squeeze(results_dict[group][model_name]["observed"]),
        "pred": np.squeeze(results_dict[group][model_name]["predicted"])
    }
    return df


def load_sl_scores(filepaths, group, model_name, n_threads=10):
    # create the thread pool
    with ThreadPoolExecutor(n_threads) as executor:
        # submit all tasks
        futures = [executor.submit(_load_sl_scores, p, group, model_name) for p in filepaths]
        # process all results
        results = {}
        for future in as_completed(futures):
            # open the file and load the data
            sl, df = future.result()
            results[sl] = df
        return results


def _load_sl_scores(filepat, group, model_name):
    data = load_hdf5(filepat)
    scores = data[group][model_name]["scores"]
    for k, v in scores.items():
        scores[k] = np.squeeze(v)
    return data["searchlight_id"], scores


def load_file_parallel(filepaths, nthreads=10):
    # create the thread pool
    with ThreadPoolExecutor(nthreads) as executor:
        # submit all tasks
        futures = [executor.submit(load_hdf5, p) for p in filepaths]
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            data, filepath = future.result()
            # report progress
        return [fut.result() for fut in futures]
