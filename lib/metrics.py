'''
Date: 2021-01-13 16:30:25
LastEditTime: 2021-01-13 16:32:48
Description: Metrics including MAE/RMSE/MAPE
FilePath: /DMGAN/lib/metrics.py
'''
import numpy as np
# import tensorflow as tf

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /=  np.mean((mask))
        mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
        loss = np.abs(preds-labels) / labels
        loss = loss * mask
        loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
        return np.mean(loss)

# def masked_mape_np(preds,labels,null_val=np.nan):
#     mask=labels!=0.0
#     return np.fabs((labels-preds)/np.clip(labels, 0.1, 1)).mean()

# def masked_mape_np(preds,labels, null_val=np.nan):
#     '''
#     Mean absolute percentage error.
#     :param labels: np.ndarray or int, ground truth.
#     :param preds: np.ndarray or int, prediction.
#     :param axis: axis to do calculation.
#     :return: int, MAPE averages on all elements of input.
#     '''
#     if not isinstance(preds, np.ndarray):
#         preds = preds.cpu().numpy()
#         labels = labels.cpu().numpy()
#     mape = (np.abs(preds - labels) / (np.abs(labels)+1e-5)).astype(np.float64)
#     mape = np.where(mape > 5, 5, mape)
#     return np.mean(mape)


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    # mape = stemgnn_mape(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse