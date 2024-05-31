# DOoC

## Usage


### Train


```python
# Regression train
import random
import torch
from torch import nn
import torch.optim as optim

from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig

from dooc import models, datasets


# datasets
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
ds = datasets.MutSmi(tokenizer)
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
mutations = [[random.choice([0, 1]) for _ in range(3008)],
             [random.choice([0, 1]) for _ in range(3008)]]
# mutations contains 0/1 encoding information of the genome
values = [0.85, 0.78]
smiles_src, smiles_tgt, mutations_src, out = ds(smiles, mutations, values)

# MutSmiFullConnection train
model = models.MutSmiFullConnection()
model.load_pretrained_ckpt('/path/to/drugcell.ckpt', '/path/to/moltx.ckpt')
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
            lr=1e-04,
            foreach=False
        )

optimizer.zero_grad()
pred = model(smiles_src, smiles_tgt, mutations_src)
loss = mse_loss(pred, out)
loss.backward()
optimizer.step()

torch.save(model.state_dict(), '/path/to/mutsmifullconnection.ckpt')

# MutSmiXAttention train
model = models.MutSmiXAttention()
model.load_pretrained_ckpt('/path/to/drugcell.ckpt', '/path/to/moltx.ckpt')
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
            lr=1e-04,
            foreach=False
        )

optimizer.zero_grad()
pred = model(smiles_src, smiles_tgt, mutations_src)
loss = mse_loss(pred, out)
loss.backward()
optimizer.step()

torch.save(model.state_dict(), '/path/to/mutsmixattention.ckpt')
```

### Inference

```python
import random
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig
from dooc import pipelines, models


# MutSmiFullConnection
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
model = models.MutSmiFullConnection()
model.load_ckpt('/path/to/mutsmifullconnection.ckpt')
pipeline = pipelines.MutSmiFullConnection(smi_tokenizer=tokenizer, model=model)
mutations = [random.choice([0, 1]) for _ in range(3008)]
smiles = "CC[N+](C)(C)Cc1ccccc1Br"
predict = pipeline(mutations, smiles) # e.g. 0.85

# MutSmiXAttention
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
model = models.MutSmiXAttention()
model.load_ckpt('/path/to/mutsmixattention.ckpt')
pipeline = pipelines.MutSmiXAttention(smi_tokenizer=tokenizer, model=model)
mutations = [random.choice([0, 1]) for _ in range(3008)]
smiles = "CC[N+](C)(C)Cc1ccccc1Br"
predict = pipeline(mutations, smiles) # e.g. 0.85
```
