# Source Code for Graph Mining Team Project

### Environment

 
1. Make Conda environment \
   **(Both One2One and Survival use the same virtual environment)**
   ```
      conda env update --file requirements.yml
   ```

2. Activate conda environment
   ```
      conda activate AI607
   ```

## One2One Task
### Description
This task involves predicting the winner of a match between two players. The possible outcomes are 'P1', 'P2', and 'DRAW'. The main architecture of this model is a Graph Neural Network, and as a sub-task, Matrix Factorization was also performed.

### Key Hyper-Parameters
- `num_aspect`: The number of aspects extracted from players' features.


### Return Value
- One of ('P1', 'P2', 'DRAW')

### Example
1. Training the GNN-Based Model with only the training dataset, and inference with the validation dataset 
   ```
   python main.py --model gnn_based_train
   ```

2. Training the GNN-Based Model with the training dataset and the validation dataset, and inference with the test dataset
   ```  
   python main.py --model gnn_based_test
   ```

3. Training the MF Model with the training dataset, inference with the validation dataset  
   ```
   python main.py --model mf_based_train
   ```
## Survival Task
### Description
This task involves predicting the score of each player in each Battle Royale Game. The possible score range from 0 to 9.
The main architecture of this model is a [Hypergraph Networks with Hyperedge Neurons (HNHN)](https://arxiv.org/abs/2006.12278).

### Hyper-Parameters
- `n_hidden`: number of hidden dimension (default: 100)
- `n_epoch`: number of epoch (default: 200)
- `batch_size`: batch size (default: 64)
- `normalize`: whether to normalize node feature (default: False)
- `n_layers`: number of layers (default: 1)
- `lr`: learning rate (default: 0.04)

### Return Value
- One of Score from 0 to 9

### Example
1. Training the HNHN with only the training dataset, and inference with the validation dataset 
   ```
   python task2_main.py
   ```
2. Training the HNHN with the training dataset and the validation dataset, and inference with the test dataset
   ```
   python task2_main_test.py
   ```

### Citation
```
@article{HNHN2020,
  title         = {HNHN: Hypergraph Networks with Hyperedge Neurons},
  author 	= {Dong, Yihe and Sawin, Will and Bengio, Yoshua},
  url       	= {https://arxiv.org/abs/2006.12278},
  journal 	= {ICML Graph Representation Learning and Beyond Workshop},
  year          = {2020}
  }
```