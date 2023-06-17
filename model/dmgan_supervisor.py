'''
Date: 2021-01-13 16:58:31
LastEditTime: 2021-01-13 21:02:47
FilePath: /DMGAN/model/dmgan_supervisor.py
'''
import os
import time

import numpy as np
import torch
from lib import utils, metrics
from model.dmgan_model import DMGANModel
from lib.loss import masked_mae_loss

class DMGAN_Supervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self.cuda_idx = int(kwargs.get('cuda_idx'))
        self.device = torch.device("cuda:{}".format(self.cuda_idx) if torch.cuda.is_available() else "cpu")
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm')
        self.steps = self._train_kwargs.get('steps')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)        
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dmgan_model = DMGANModel(adj_mx, self._logger, self.device, **self._model_kwargs)
        
        self.dmgan_model = dmgan_model.cuda(self.cuda_idx)
        self._logger.info(self._model_kwargs.get('filter_type'))
        self._logger.info("Model created")
        self._logger.info(dmgan_model)

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dmgan_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dmgan_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        if bool(self._kwargs.get('test').get('test_model')):
            if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
                path = 'data/model/pretrained/METR-LA/' 
            elif self._data_kwargs['dataset_dir'] == 'data/PEMS-BAY':
                path = 'data/model/pretrained/PEMS-BAY/'
        else:
            path = 'models/'
        self._setup_graph()
        assert os.path.exists(path + 'epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(path + 'epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dmgan_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

        
        self._logger.info('Load model and get val acc!')
        self.evaluate(dataset='val',)
        self._logger.info('Load model and test!')
        self.evaluate(dataset='test')

    def _setup_graph(self):
        with torch.no_grad():
            self.dmgan_model = self.dmgan_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dmgan_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dmgan_model = self.dmgan_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            for i, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dmgan_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(np.expand_dims(y_truth, axis=2))
                y_preds_scaled.append(np.expand_dims(y_pred, axis=2))

                mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
                mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
                rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(t + 1, mae, mape, rmse)
                )
                

            y_truths_scaled = np.concatenate(y_truths_scaled, axis=2)
            y_preds_scaled = np.concatenate(y_preds_scaled, axis=2)
            # if dataset == 'test':
            #     np.save('test_pred.npy', y_preds_scaled)
            #     np.save('test_label.npy', y_truths_scaled)
            mae = metrics.masked_mae_np(y_preds_scaled.reshape(-1, 1), y_truths_scaled.reshape(-1, 1), 0)
            mape = metrics.masked_mape_np(y_preds_scaled.reshape(-1, 1), y_truths_scaled.reshape(-1, 1), 0)
            rmse = metrics.masked_rmse_np(y_preds_scaled.reshape(-1, 1), y_truths_scaled.reshape(-1, 1), 0)
            self._logger.info(
                    "ALL -> MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(mae, mape, rmse)
                )


            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dmgan_model.parameters(), lr=base_lr, eps=epsilon)
        self._logger.info('Start training ...')

        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * 0

        val_loss_lst = []
        for epoch_num in range(self._epoch_num + 1, epochs):
            if epoch_num in self.steps: 
                if self._train_kwargs['enable_batch_seen_update']:
                    batches_seen = 1
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

            self.dmgan_model = self.dmgan_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()
            self._logger.info("batches_seen:{}".format(batches_seen) )

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.dmgan_model(x, y, batches_seen)

                if batches_seen == 0:
                    optimizer = torch.optim.Adam(self.dmgan_model.parameters(), lr=base_lr, eps=epsilon)
                    

                loss = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dmgan_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            val_loss_lst.append(val_loss)
            end_time = time.time()

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, optimizer.param_groups[0]['lr'],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % log_every) == log_every - 1 and epoch_num >= 15:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, optimizer.param_groups[0]['lr'],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    self._logger.info("Best Epoch [{}],  min val MAE: [{:.4f}]".format(np.min(val_loss_lst), val_loss_lst.index(np.min(val_loss_lst))))
                    break
        
        self._logger.info("Best Epoch [{}],  min val MAE: [{:.4f}]".format(np.min(val_loss_lst), val_loss_lst.index(np.min(val_loss_lst))))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):

        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
