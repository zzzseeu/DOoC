# DOoC

## Usage


### Train


```python
# Regression train
from moltx import tokenizers
from dooc import models, datasets, nets


tk = tokenizers.MoltxTokenizer.from_pretrain(models.AdaMRTokenizerConfig.Prediction)
ds = datasets.MutSmiXAttention(tokenizer=tk, device=torch.device('cpu'))
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
mutations = [[1, 0, 0, ...], [1, 0, 1, ...]]
# e.g.
# import random
# [random.choice([0, 1]) for _ in range(3008)]
values = [0.85, 0.78]
smiles_src, smiles_tgt, mutations_src, out = ds(smiles, mutations, values)

model = models.MutSmiXAttention()
model.load_pretrained_ckpt('/path/to/drugcell.ckpt', '/path/to/moltx.ckpt')

crt = nn.MSELoss()

optim.zero_grad()
pred = model(smiles_src, smiles_tgt, mutations_src)
loss = crt(pred, out)
loss.backward()
optim.step()

torch.save(model.state_dict(), '/path/to/mutsmixattention.ckpt')
```

### Inference

```python
from dooc import pipelines, models
# dooc
model = models.MutSmiXAttention()
model.load_ckpt('/path/to/mutsmixattention.ckpt')
pipeline = pipelines.MutSmiXAttention()
pipeline([1, 0, 0, ...], "C=CC=CC=C")
# 0.85


```
