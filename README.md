# AgePrediction
## Summary
While companies generally seek to better understand their customers to improve marketing efforts, such knowledge is often limited. For example, in B2C settings, information like customer age can be informative for key performance indicators such as retention rate, yet it is often unavailable. One possible solution is to estimate customer age based on first names: when parents choose a name for their child, they are influenced by time-specific trends. However, authoritative datasets are costly, may not cover all first names in a customer base, and often provide average birth years that may not accurately reflect the underlying distribution. Moreover, such datasets can become outdated and typically lack uncertainty estimates.

As an alternative, I present an approach that uses frequency data on 65,373 names from 1909 to 2023, provided by the Swiss Federal Statistical Office (https://www.bfs.admin.ch/asset/en/32208760). A model is trained to predict birth years based on names. The training process includes a pretraining and a finetuning stage. During pretraining, a bidirectional transformer with multiple attention layers learns patterns in names (feature extraction). During finetuning, these features are used to predict the most likely birth year associated with a given name.

<img src="figures/project_sketch.svg" alt="Project outline" width="80%" height="80%" />

The trained model estimates birth years with a Smoothed Mean Absolute Error (SMAE) of 15.8 years. This error cannot be brought lower than 11.8 years on average, due to names usually reoccurring in multiple years, which introduces estimation uncertainty. In comparison to a benchmark SMAE of 19.0 years (from knowing the mean of the outcome variable and nothing else), this model performance signifies a reduction in error of 44.1%. The model can be readily adapted by businesses to estimate the age distribution of their customer base and validated using customer subsets for which age data is available.

<img src="figures/smae_plot.png" alt="Model performance" width="80%" height="80%" />

Further performance improvements are likely achievable by incorporating additional data, such as customer behavior derived from engagement with online services.

## Model specifics
### Pretraining
At the pretraining stage, tokens are first encoded into trainable embedding vectors of length 128. I employ four attention heads, intertwined with 2 linear layers, 2 layer norm and 3 dropout layers. There are six of these attention layers. After 300 training epochs, the pretraining model arrives at a Cross Entropy Loss of 1.37. This denotes an accuracy of 24.6% at predicting the masked character, which is a 14-fold improvement over an untrained model picking characters at random (1/57=1.8%). 

### Finetuning
At the finetuning stage, a linear regressor is placed on top of the pretrained Transformer, consisting of 4 linear layers sandwiching ReLU- and dropout layers, as well as a residual neural network. After a warmup period for the regressor, the pretrained transformer weights are unfrozen in order to be finetuned during the task of predicting the likely birth year associated with each name.