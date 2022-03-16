import pandas as pd
import pytorch_lightning as pl
from nam.config import defaults
from nam.data import NAMDataset
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# get raw FICO data
data_path = 'datasets/FICO/heloc_dataset_v1.csv'
df = pd.read_csv(data_path)

# preprocess data to be transformed into torch tensors
cat_vars = ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
cont_vars = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile',
             'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
             'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades', 'NumTradesOpeninLast12M',
             'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
             'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
             'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
             'PercentTradesWBalance']
df.rename(columns={'RiskPerformance': 'target'}, inplace=True)
df['weights'] = np.ones(len(df))
# train model
config = defaults()
config.data_path = data_path
config.num_epochs = 50
config.lr = 1e-2
config.num_workers = 8
config.batch_size = 128
config.early_stopping_patience = 15
config.logdir = 'saved_models'

print(f'Training NAM with config {config}')

dataset = NAMDataset(config,
                     data_path=df, weights_column='weights',
                     features_columns=cat_vars + cont_vars,
                     targets_column='target')

trainloader, valloader = dataset.train_dataloaders()
trainloader.num_workers = config.num_workers
valloader.num_workers = config.num_workers

model = NAM(
    config=config,
    name="NAM_FICO",
    num_inputs=len(dataset[0][0]),
    num_units=get_num_units(config, dataset.features),
)

tb_logger = TensorBoardLogger(save_dir=config.logdir,
                              name=f'{model.name}',
                              version=f'fold_1')

checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir +
                                               "/{epoch:02d}-{val_loss:.4f}",
                                      monitor='val_loss',
                                      save_top_k=config.save_top_k,
                                      mode='min')

litmodel = LitNAM(config, model)
trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback],
                     max_epochs=config.num_epochs)
trainer.fit(litmodel,
            train_dataloader=trainloader,
            val_dataloaders=valloader)

## Testing the trained model
trainer.test(test_dataloaders=dataset.test_dataloaders())
