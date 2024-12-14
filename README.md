# DenseNet(2017)


Link to Paper:

***“*Densely Connected Convolutional Networks*” - 2017***

[https://arxiv.org/pdf/1608.06993v5](https://arxiv.org/pdf/1608.06993v5)

---


# 1. Introduction

**Gao Huang       |       Zhuang Liu       |       Laurens Van der Maaten**

<br><br>

> **Motivation**
> 

---

*“Convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output”*

<br><br> 

> **Significance of this paper**
> 

---

- **Densely connected architecture:**
    
    Connects each layer to every other layer in a feed-forward fashion.
    
    $`L`$-layers network has $`\frac{L(L+1)}{2}`$ direct connections
    
- **DenseNet has several compelling advantages:**
    - Alleviate the vanishing-gradient problem
    - Strengthen feature propagation
    - Encourage feature reuse
    - Substantially reduce the number of parameters(Counter-intuitive)

<br><br> 

> **Experiments & Results**
> 

---

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/image.png)

“Deep Networks with Stochastic Depth”

[https://arxiv.org/pdf/1603.09382](https://arxiv.org/pdf/1603.09382)

<br><br><br>

# 2. Architecture


Crucially, in contrast to ResNets, **DenseNet never combines features through summation** before they are passed into a layer; instead, **DenseNet combines features by concatenating** them.

> **Schematic Architecture**
> 

---

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/image%201.png)

> **Dense Connectivity**
> 

---

**Residual Connection vs. Dense Connectivity**

---

**Residual Connection**

Feedforward networks connect the output of the $\ell$-th layer as input to the ($\ell$ + 1)-th layer

$$
x_\ell = H_\ell(x_{\ell−1}) + x_{\ell−1}
$$

**Dense Connectivity**

Direct connections from any layer to all subsequent layers. Consequently, the $\ell$-th layer receives the feature-maps of all preceding layers, $x_0, . . . , x_{\ell−1}$, as input

$$
x_\ell = H_\ell([x_0,x_1,...,x_{\ell−1}])
$$

where $[x_0, x_1, . . . , x_{\ell−1}]$ refers to the concatenation of the feature-maps produced in layers $0, ...  , \ell − 1$.

Parameter reduction

---

A possibly counter-intuitive effect of this dense connectivity pattern is that it requires *fewer* parameters than traditional convolutional networks, as there is no need to re-learn redundant feature-maps.

![Screenshot 2024-08-13 at 6.04.42 AM.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/Screenshot_2024-08-13_at_6.04.42_AM.png)

In the above diagram, the feature-maps from the first layer connected in orange lines are **NOT** counted as “parameters” because they are only learned in the first layer and only “transferred(copied)” to the deeper layers. Similarly, yellow feature maps from the second layer connected via yellow lines are **NOT** counted as “parameters” as they are only learned/updated in the second layer and transferred to the later layers.


> **Composite Function**
> 

---

**Change of orders of Conv Blocks**

---

![Traditional Conv Block](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/4ac22748-4425-4f56-aaf7-7a5c7aedfd37.png)

Traditional Conv Block

![Conv Block in DenseNet](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/be9f3cd9-6429-42a1-bf55-73c30f3e00eb.png)

Conv Block in DenseNet

Composite function of three consecutive operations: 

- Batch normalization (BN)
- followed by a rectified linear unit (ReLU)
- and a 3 × 3 convolution (Conv)

This new architecture is first proposed in “[Identity mappings in deep residual networks](https://arxiv.org/pdf/1603.05027)” which is viewed as the activation functions (ReLU and BN) as **“pre-activation”** of the weight layers, in contrast to conventional wisdom of **“post-activation”**. This led to easier training:

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/image%202.png)



> **Transition layers**
> 

---

**Motivation**

---

The concatenation operation is not viable when the size of feature-maps changes.

**Transition Layer**

---

An essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in our architecture we divide the network into multiple densely connected *dense blocks:*

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/image%203.png)

Each Transition Layer consists of

$$
BN-Conv(1× 1)-AvgPool(2\times2) 
$$

Down-sampling refers to the process of reducing the spatial dimensions (i.e., height and width) of the feature maps while retaining the depth (number of channels) of the feature maps. This is typically done to progressively decrease the size of the input as it passes through the network, which helps to reduce the computational complexity and to capture more abstract features at different levels of the network.


> Code Implementation
> 

---

```python
class Transition(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels/2, kernel_size=(1, 1), bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.transition(x)
        return x
```



> **Compression**
> 

---

Compression

---

To further improve model compactness, we can reduce the number of feature-maps at transition layers. If a dense block contains $m$ feature-maps, we let the following transition layer generate $⌊θm⌋$ output feature-maps, where $0 < θ ≤ 1$ is referred to as the compression factor. 

- When θ = 1, the number of feature-maps across transition layers remains unchanged.
- We refer the DenseNet with $θ < 1$ as DenseNet-C, and we set $θ = 0.5$ in our experiment. When both the bottleneck and transition layers with $θ < 1$ are used, we refer to our model as DenseNet-BC.

> Transition layer with Compression applied
> 

---

- Conv layer reduces number of channels into half.
- AvgPool2d layer reduces spatial dimensions into half.

![Screenshot 2024-08-13 at 3.31.17 PM.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/Screenshot_2024-08-13_at_3.31.17_PM.png)

In code, compression is done in transition layer:

```python
nn.Conv2d(in_channels=in_channels, out_channels=in_channels/2, kernel_size=(1, 1), bias=False)
```


> **Bottleneck**
> 

---

**Dense Block is composed of Bottlenecks**

---

![Screenshot 2024-08-13 at 3.04.49 PM.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/Screenshot_2024-08-13_at_3.04.49_PM.png)

Each Bottleneck is 

$$
BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) 
$$

version of $`H_\ell`$. Paper lets each $`1×1`$ convolution produce $`4k`$ feature-maps.

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/311f1ccd-093f-4124-a2a7-4de417818109.png)

So each bottleneck with input of $n$-channels come out as ($n + k$)-channels.

$$
\text{Input}(n) -1\times1 \text{ Conv} (4k) - 3\times3 \text{ Conv}(n + 4k)
$$

So input is compressed into $4k$ channels then come out as $n+4k$ channels.

```python
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=4*growth_rate, kernel_size=(1, 1), bias=False)
        )
        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*growth_rate, out_channels=growth_rate, kernel_size=(3, 3), padding=1, bias=False)
        )

    def forward(self, x):
        input_feat = x
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        # input: [N, in_channels, H, W] + x: [N, growth_rate, H, W] = [N, in_channels + growth_rate, H, W]  
        return torch.cat([input_feat, x], dim=1)
```


> **Growth Rate**
> 

---

Definition of Growth Rate

---

If each function $H_\ell$ produces $k$ feature-maps, it follows that the $\ell$-th layer has $k_0 + k × (\ell − 1)$ input feature-maps, where  $k_0$  is the number of channels in the input layer. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, *e.g.*, $k = 12$. We refer to the hyperparameter $k$ as the *growth rate* of the network


Interpretation

---

One explanation is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network’s **“collective knowledge”**. One can view the feature-maps as the ***global state*** of the network. Each layer adds $k$ feature-maps of its own to this state. The growth rate regulates how much new information each layer con- tributes to the global state. The global state, once written, can be accessed from everywhere within the network and, unlike in traditional network architectures, there is no need to replicate it from layer to layer.

</aside>

> **Architecture**
> 

---

![image.png](DenseNet(2017)%2014d314b8d56d4c6fa4dee3ca7ed98741/image%204.png)

```python
import torch
from torch import nn

class DenseNet(nn.Module):
    def __init__(self, block_list, growth_rate, num_classes):
        super().__init__()

        self.growth_rate = growth_rate
        self.in_channels = 2 * self.growth_rate
        
        # First Conv layer and pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, biase=False),
            nn.BatchNorm2d(2*growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Dense Blocks
        self.denseblock1 = self._make_dense_block(block_list[0])
        self.denseblock2 = self._make_dense_block(block_list[1])
        self.denseblock3 = self._make_dense_block(block_list[2])
        self.denseblock4 = self._make_dense_block(block_list[3], last_block=True)
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)
        

    def _make_dense_block(self, num_BottleNecks, last_block=False):
        layers = []
        for _ in range(num_BottleNecks):
            layers.append(BottleNeck(self.in_channels, self.growth_rate)
            self.in_channels += self.growth_rate # In a dense block, each layer's channel increase by growth_rate
        
        if last_block:
            layers.append(nn.BatchNorm2d(self.in_channels))    
            layers.append(nn.ReLU())
        else: # If not a last dense block, append Transition layer
            layers.append(Transition(self.in_channels))
            self.in_channels //= 2 # before moving to next Dense Block, reduce the channels by factor of 2

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.denseblock1(x)
        x = self.denseblock2(x)
        x = self.denseblock3(x)
        x = self.denseblock4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
```

# 3. Implementation in Code

$\color{black}\rule{365px}{3px}$

<aside>
<img src="https://www.notion.so/icons/chart-area_green.svg" alt="https://www.notion.so/icons/chart-area_green.svg" width="40px" /> Datasets downloaded from Kaggle

---

Animals-10 Dataset consists of ~25k images of 10 classes of animals:

[https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download](https://www.kaggle.com/datasets/alessiocorrado99/animals10?resource=download)

</aside>

<aside>
<img src="https://www.notion.so/icons/snippet_green.svg" alt="https://www.notion.so/icons/snippet_green.svg" width="40px" /> Code Github Link

---

https://github.com/h383kim/DenseNet

</aside>
