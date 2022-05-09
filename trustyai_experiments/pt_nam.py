import numpy as np
import pandas as pd
import pytorch_lightning as pl
from nam.config import defaults
from nam.data import NAMDataset
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class NAM_FICO:
    def __init__(self, config=None, data_path: str = 'datasets/FICO/heloc_dataset_v1.csv'):
        self.columns = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile',
                        'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                        'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver',
                        'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades',
                        'MSinceMostRecentInqexcl7days',
                        'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden',
                        'NetFractionInstallBurden',
                        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
                        'PercentTradesWBalance']
        self.continuous_cols = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
                                'AverageMInFile',
                                'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                                'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades',
                                'NumTradesOpeninLast12M',
                                'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
                                'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden',
                                'NetFractionInstallBurden',
                                'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
                                'NumBank2NatlTradesWHighUtilization',
                                'PercentTradesWBalance']
        self.categorical_cols = ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
        self.target_col = ['RiskPerformance']
        if config is None:
            config = defaults()
            config.data_path = data_path
            config.num_epochs = 50
            config.lr = 1e-2
            config.num_workers = 8
            config.batch_size = 128
            config.early_stopping_patience = 15
            config.logdir = 'saved_models'

        self.config = config

        df = pd.read_csv(data_path)
        df.rename(columns={'RiskPerformance': 'target'}, inplace=True)
        df['weights'] = np.ones(len(df))

        self.dataset = NAMDataset(config,
                                  data_path=df, weights_column='weights',
                                  features_columns=self.categorical_cols + self.continuous_cols,
                                  targets_column='target')
        model = NAM(
            config=config,
            name="NAM_FICO",
            num_inputs=len(self.dataset[0][0]),
            num_units=get_num_units(config, self.dataset.features),
        )
        self.litmodel = LitNAM(config, model)

    def train(self):
        tb_logger = TensorBoardLogger(save_dir=self.config.logdir,
                                      name='NAM_FICO',
                                      version=f'fold_1')

        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir +
                                                       "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=self.config.save_top_k,
                                              mode='min')

        self.trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback],
                                  max_epochs=self.config.num_epochs)
        trainloader, valloader = self.dataset.train_dataloaders()
        return self.trainer.fit(self.litmodel,
                                train_dataloader=trainloader,
                                val_dataloaders=valloader)

    def evaluate(self):
        return self.trainer.test(test_dataloaders=self.dataset.test_dataloaders())

    def predict(self, data_df: pd.DataFrame):
        if isinstance(data_df, pd.Series):
            data_df = pd.DataFrame(data_df).T
        if isinstance(data_df, np.ndarray):
            data_df = pd.DataFrame(data=data_df, columns=self.columns)

        df = data_df.copy()
        if 'RiskPerformance' in df.columns:
            df.rename(columns={'RiskPerformance': 'target'}, inplace=True)
        df['weights'] = np.ones(len(df))
        if not 'target' in df.columns:
            df['target'] = np.zeros(len(df))

        dataset = NAMDataset(self.config,
                             data_path=df, weights_column='weights',
                             features_columns=self.categorical_cols + self.continuous_cols,
                             targets_column='target')

        return self.trainer.predict(dataloaders=dataset.data_loaders())
