import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig


class TabularFICO:
    def __init__(self):
        self.columns = ['ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile',
                        'NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec',
                        'PercentTradesNeverDelq','MSinceMostRecentDelq','MaxDelq2PublicRecLast12M','MaxDelqEver',
                        'NumTotalTrades','NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days',
                        'NumInqLast6M','NumInqLast6Mexcl7days','NetFractionRevolvingBurden','NetFractionInstallBurden',
                        'NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization',
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
        self.categorical_cols=['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
        self.target_col = ['RiskPerformance']
        data_config = DataConfig(
            target=self.target_col,
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=512,
            max_epochs=50,
            auto_select_gpus=False,
            gpus=0,  # index of the GPU to use. 0, means CPU
        )
        optimizer_config = OptimizerConfig()

        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            layers="100-500-100",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
            learning_rate=1e-4
        )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        self.model = tabular_model

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        return self.model.fit(train=train_df, validation=valid_df)

    def evaluate(self, test_df: pd.DataFrame):
        return self.model.evaluate(test_df)

    def save(self, path: str = "../saved_models/fico_tabular_basic"):
        self.model.save_model(path)

    def load(self, path: str = None):
        if path is None:
            self.model = self.model.load_best_model()
        else:
            self.model = TabularModel.load_from_checkpoint(path)

    def predict(self, data_df: pd.DataFrame):
        if isinstance(data_df, pd.Series):
            data_df = pd.DataFrame(data_df).T
        if isinstance(data_df, np.ndarray):
            data_df = pd.DataFrame(data=data_df, columns=self.columns)

        return self.model.predict(data_df, ret_logits=True)
