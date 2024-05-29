# GAI-enhanced-DRL

## Proposed Framework
***Part A (GAN-enhanced GAI):*** We enhance the critic network of DRL by using GAN. Specifically, the generator network outputs estimated action values, while the target generator network obtains the target action values. The discriminator network attempts to minimize the distance between the estimated action values and the target action values calculated by the Bellman operator.  
***Part B (VAE-enhanced GAI):*** We use VAE to reduce the dimensionality of the high-dimensional state space to reduce the computational complexity issue in DRL. In this case, we train the VAE with data and use the decoder to extract representations of the state space, which are then used as inputs for the actor and critic networks. Additionally, VAE can construct a latent representation space for continuous parameters conditioned on state and embedding of discrete actions to handle hybrid actions.  
***Part C (Transformer-enhanced GAI):*** We enhance the actor network of DRL by using Transformer. Specifically, we replace the Multi-Layer Perceptron (MLP) with a network based on the attention mechanism of Transformer to analyze the current state in the environment.  
***Part D (GDM-enhanced GAI):*** We improve the policy network of DRL by employing the reverse process of GDM. Specifically, we treat the policy network as a denoiser, progressively adding denoising noise to the initial Gaussian noise to recover or discover the optimal actions.
## Run the Program
To create a new conda environment, execute the following command:

```python
conda create --name GAIDRL python==3.10
```
Activate the created environment with the following command:

```python
conda activate GAIDRL
```
Install the following packets using pip:

```python
pip install gym==0.26.2
pip install scipy==1.13.
pip install matplotlib==3.8.4
pip install numpy==1.26.4
pip install scipy==1.13.0
```
Run the different algorithm:

```python
GAN-enhanced TD3: run GAN_TD3_simple.py;
VAE-enhanced TD3: run VAE_TD3.py ;
Transformer-enhanced TD3: run Attention_TD3_double.py;
GDM-enhanced TD3: run mainDM3.py.
```

## Bibtex

```python
@article{sun2024,
        title={Generative AI for Deep Reinforcement Learning: Framework, Analysis, and Use Cases},
        author={Geng Sun, Wenwen Xie, Dusit Niyato, Fang Mei, Hongyang Du, Jiawen Kang, Shiwen Mao},
        journal={arXiv preprint arXiv:2404.10556},
        year={2024}
      }
```
