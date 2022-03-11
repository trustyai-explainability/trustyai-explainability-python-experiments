import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig

data_config = DataConfig(
    target=['RiskPerformance'],
    continuous_cols=['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile',
                     'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                     'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades', 'NumTradesOpeninLast12M',
                     'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
                     'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
                     'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
                     'PercentTradesWBalance'],
    categorical_cols=['MaxDelq2PublicRecLast12M', 'MaxDelqEver'],
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    auto_select_gpus=False,
    gpus=0,  # index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
    learning_rate=1e-3
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

train_df = pd.read_csv('datasets/FICO/heloc_dataset_v1.csv')
tabular_model.fit(train=train_df, validation=train_df[:10])  # TODO: use a validation set
result = tabular_model.evaluate(train_df[-10:])  # TODO: use a test set
tabular_model.save_model("models/fico_tabular_basic")
print(result)
