# Transformer Encoder and Decoder Optimization
## Connor Gag

### How to run this
Run the main.py file with the argument for the part. There are 3 options:
`python main.py --part 1`
`python main.py --part 2`
`python main.py --part 3`

In part 3, we put more parameters into the model because we are trying out different architectures. These are encoded in the variable 'embedding_strategy' and you can just set this variable in main.py.

The positional embedding strategies are as follows:
- 'basic_positional'
- 'sinusoidal'
- 'alibi'