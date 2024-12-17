# Transformer_GRACE

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/office/20241128211830.png)

fill the GRACE gap using novel Transformer method  

Wang, L., & Zhang, Y. (2024). Filling GRACE data gap using an innovative transformer-based deep learning approach. Remote Sensing of Environment, 315, 114465. https://doi.org/10.1016/j.rse.2024.114465  

Please considering using cuda GPU to train the newwork

The environment recommended are:

* python3.8.12  
* cuda_11.6.1_511.
* cudnn_8.3.2.7.0

## fill the data gap product

Single month data would be found at figshare below

[![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.24614958.svg)](https://doi.org/10.6084/m9.figshare.24614958)

We would also provide full month ``.nc`` file, you can download from [Google Drive][https://drive.google.com/file/d/1l7oQdkEUl-bvoH7OTg9mKlaIybtuscqA/view?usp=drive_link] or [BaidunetDisk](https://pan.baidu.com/s/1Zah2ILpn-DTjDG-WimT_TQ?pwd=ancc)


## input data

Note: the original input require large data which cannot be uploaded to GitHub or shared easily, such as ERA5 datasets. 
As a result, we have uploaded it to figure share.

[![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.24297604.svg)](https://doi.org/10.6084/m9.figshare.24297604)

## Build the model

```
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper
    
    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
        
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    
    """
    def __init__(self):
        super(TransformerTimeSeries,self).__init__()
        self.input_embedding = context_embedding(Config.in_channels+1,256,40)
        self.positional_embedding = torch.nn.Embedding(512,256)

        
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256,nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        
        self.fc1 = torch.nn.Linear(256,1)
        
    def forward(self,x,y,attention_masks):
        
        z = torch.cat((y,x),1)

        z_embedding = self.input_embedding(z).unsqueeze(1).permute(3, 1, 0, 2)
        x1 = x.type(torch.long)
        x1[x1 < 0] = 0
        positional_embeddings = self.positional_embedding(x1).permute(2, 1, 0, 3)
        
        input_embedding = z_embedding+positional_embeddings
        input_embedding1 = torch.mean(input_embedding, 1)
        transformer_embedding = self.transformer_decoder(input_embedding1,attention_masks)

        output = self.fc1(transformer_embedding.permute(1,0,2))
        
        return output


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=Config.in_channels,embedding_size=256,k=40):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x)
        return torch.tanh(x)
```

## evaluate the model performance

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/20231012175509.png)

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/20231012175537.png)

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/20231012175557.png)

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/20231012175619.png)

## NOTE: The code of log-sparse is not included, this only shows the full causal convolution.  

We have additional research that uses these strategies, the code for sparse convolution will be released later.
Here we provide part of the pseudo-code:

```
import torch
import torch.nn.functional as F
import numpy as np

class LogSparseCausalConv1d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_dilation,
                 stride=1,
                 bias=True):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            kernel_size: The size of kernel
            max_dilation: Maximum expansion rate of logarithmic sparse 
            stride: Stride
            bias: Whether to use bias
        """
        super(LogSparseCausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_dilation = max_dilation
        self.stride = stride
        self.bias = bias

        # 初始化权重和偏置
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size, in_channels, seq_len = x.size()

        # Appropriate padding of inputs to ensure causality
        padding = (self.kernel_size - 1) * self.max_dilation
        x = F.pad(x, (padding, 0))  # 左侧填充

        # Compute the log sparse expansion rate, set to logarithmic growth
        dilations = self._get_log_dilations(seq_len)

        output = torch.zeros(batch_size, self.out_channels, seq_len, device=x.device)

        # Iterate over dilations, convolve using different expansion rates
        for i, dilation in enumerate(dilations):
            conv_output = F.conv1d(x, self.weight, self.bias, stride=self.stride, dilation=dilation)
            output[:, :, i] = conv_output[:, :, i]

        return output

    def _get_log_dilations(self, seq_len):
        """
        Generate sparse logarithmically spaced convolutional expansion rates based on time series length and maximum expansion rate.
        """
        dilations = [1]
        for i in range(1, seq_len):
            dilation = min(2 ** int(np.log2(i + 1)), self.max_dilation)
            dilations.append(dilation)
        return dilations

```

