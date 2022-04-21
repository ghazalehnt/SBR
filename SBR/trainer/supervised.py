import operator
import time
from os.path import exists, join

import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
import numpy as np

from SBR.utils.metrics import calculate_metrics, log_results
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class SupervisedTrainer:
    def __init__(self, config, model, device, logger, exp_dir, test_only=False, relevance_level=1,
                 users=None, items=None):
        self.model = model
        self.device = device
        self.logger = logger
        self.test_only = test_only  # todo used?
        self.relevance_level = relevance_level
        self.valid_metric = config['valid_metric']
        self.patience = config['early_stopping_patience']
        self.best_model_path = join(exp_dir, 'best_model.pth')
        self.best_valid_output_path = join(exp_dir, 'best_valid_output.json')
        self.test_output_path = join(exp_dir, 'test_output.json')
        self.users = users
        self.items = items

        if config['loss_fn'] == "BCE":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss and do not apply the sigmoid beforehand
        elif config['loss_fn'] == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        # elif config["loss_fn"] == "CE":  ## todo do we need this???
            # self.loss_fn = torch.nn.CrossEntropyLoss
        else:
            raise ValueError(f"loss_fn {config['loss_fn']} is not implemented!")

        if config['optimizer'] == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        elif config['optimizer'] == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not implemented!")

        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
        self.use_amp = config['use_amp']

        self.epochs = config['epochs']
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_saved_valid_metric = np.inf if self.valid_metric == "valid_loss" else -np.inf
        if exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['epoch']
            self.best_saved_valid_metric = checkpoint['best_valid_metric']
            self.model.to(device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("last checkpoint restored")


        self.model.to(device)

    def fit(self, train_dataloader, valid_dataloader):
        early_stopping_cnt = 0
        comparison_op = operator.lt if self.valid_metric == "valid_loss" else operator.gt

        # eval here first mainly to test why it is very slow:
        # epoch = -1
        # outputs, ground_truth, valid_loss, users, items = self.predict(valid_dataloader)
        # print(f"Valid loss epoch {epoch}: {valid_loss:.4f}")
        # results = calculate_metrics(ground_truth, outputs, users, items, self.relevance_level, 0.5)
        # results["valid_loss"] = valid_loss.item()
        # self.logger.add_scalar('epoch_metrics/best_epoch', self.best_epoch, epoch)
        # self.logger.add_scalar('epoch_metrics/best_valid_metric', self.best_saved_valid_metric, epoch)
        # for k, v in results.items():
        #     self.logger.add_scalar(f'epoch_metrics/valid_{k}', v, epoch)
        ### until here

        for epoch in range(self.start_epoch, self.epochs):
            if early_stopping_cnt == self.patience:
                print(f"Early stopping after {self.patience} epochs not improving!")
                break

            self.model.train()

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            start_time = time.time()
            train_loss, total_count = 0, 0

            # for loop going through dataset
            for batch_idx, batch in pbar:
                # data preparation
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.time() - start_time

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(batch)
                    loss = self.loss_fn(output, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # loss.backward()
                # self.optimizer.step()
                train_loss += loss
                total_count += label.size(0)

                # compute computation time and *compute_efficiency*
                process_time = time.time() - start_time - prepare_time
                compute_efficiency = process_time / (process_time + prepare_time)
                pbar.set_description(
                    f'Compute efficiency: {compute_efficiency:.4f}, '
                    f'loss: {loss.item():.4f},  epoch: {epoch}/{self.epochs}'
                    f'prep: {prepare_time:.4f}, process: {process_time:.4f}')
                start_time = time.time()
            train_loss /= total_count

            # udpate tensorboardX  TODO for logging use what  mlflow, files, tensorboard
            self.logger.add_scalar('epoch_metrics/epoch', epoch, epoch)
            self.logger.add_scalar('epoch_metrics/train_loss', train_loss, epoch)

            # evaluate every epochs
            if True:
                outputs, ground_truth, valid_loss, users, items = self.predict(valid_dataloader, self.use_amp)
                print(f"Valid loss epoch {epoch}: {valid_loss:.4f}")
                results = calculate_metrics(ground_truth, outputs, users, items, self.relevance_level, 0.5)
                results["valid_loss"] = valid_loss.item()
                for k, v in results.items():
                    self.logger.add_scalar(f'epoch_metrics/valid_{k}', v, epoch)
                if comparison_op(results[self.valid_metric], self.best_saved_valid_metric):
                    self.best_saved_valid_metric = results[self.valid_metric]
                    self.best_epoch = epoch
                    checkpoint = {
                        'epoch': self.best_epoch,
                        'best_valid_metric': self.best_saved_valid_metric,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict()
                        }
                    torch.save(checkpoint, self.best_model_path)
                    early_stopping_cnt = 0
                else:
                    early_stopping_cnt += 1
                self.logger.add_scalar('epoch_metrics/best_epoch', self.best_epoch, epoch)
                self.logger.add_scalar('epoch_metrics/best_valid_metric', self.best_saved_valid_metric, epoch)

    def evaluate(self, test_dataloader, valid_dataloader):
        # load the best model from file.
        # because we may call evaluate right after fit and in this case need to reload the best model!
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.best_epoch = checkpoint['epoch']
        self.best_saved_valid_metric = checkpoint['best_valid_metric']
        print("best model loaded!")

        outputs, ground_truth, valid_loss, internal_user_ids, internal_item_ids = self.predict(valid_dataloader)
        log_results(self.best_valid_output_path, ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items)
        results = calculate_metrics(ground_truth, outputs, internal_user_ids, internal_item_ids, self.relevance_level, 0.5)
        results["loss"] = valid_loss.item()
        for k, v in results.items():
            self.logger.add_scalar(f'final_results/validation_{k}', v)
        print(f"\nValidation results - best epoch {self.best_epoch}: {results}")

        outputs, ground_truth, test_loss, internal_user_ids, internal_item_ids = self.predict(test_dataloader)
        log_results(self.test_output_path, ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items)
        results = calculate_metrics(ground_truth, outputs, internal_user_ids, internal_item_ids, self.relevance_level, 0.5)
        results["loss"] = test_loss.item()
        for k, v in results.items():
            self.logger.add_scalar(f'final_results/test_{k}', v)
        print(f"\nTest results - best epoch {self.best_epoch}: {results}")

    def predict(self, eval_dataloader, use_amp=False):
        # bring models to evaluation mode
        self.model.eval()

        outputs = []
        ground_truth = []
        user_ids = []
        item_ids = []
        eval_loss, total_count = 0, 0
        pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))

        start_time = time.time()
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # data preparation
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.time() - start_time

                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = self.model(batch)
                    loss = self.loss_fn(output, label)
                eval_loss += loss
                total_count += label.size(0)  # TODO remove if not used
                process_time = time.time() - start_time - prepare_time
                proc_compute_efficiency = process_time / (process_time + prepare_time)

                ## TODO maybe later: having the qid names userid+itemid corresponding to the outputs and metrics
                ## for debugging. it needs access to actual user_id and item_id
                ground_truth.extend(label.squeeze().tolist())
                outputs.extend(output.squeeze().tolist())
                user_ids.extend(batch[
                                    INTERNAL_USER_ID_FIELD].squeeze().tolist())  # TODO internal? or external? maybe have an external one
                item_ids.extend(batch[INTERNAL_ITEM_ID_FIELD].squeeze().tolist())
                postprocess_time = time.time() - start_time - prepare_time - process_time

                pbar.set_description(
                    f'Compute efficiency: {proc_compute_efficiency:.4f}, '
                    f'loss: {loss.item():.4f},  prep: {prepare_time:.4f},'
                    f'process: {process_time:.4f}, post: {postprocess_time:.4f}')
                start_time = time.time()

            eval_loss /= total_count
        return outputs, ground_truth, eval_loss, user_ids, item_ids
