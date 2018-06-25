# VAE Supervised DNGR: Discriminative Learning of Network Representation

The network can be seen everywhere in our daily life and the applications of network structure are really widespread, especially the network systems involving users. Due to the particularity of the network structure, it is challenging to deal with network relationships (finding a suitable network structure representations). I have implemented a recently proposed methods DNGR to learn an effective low dimensional representation on the Wiki dataset in the previous experiments. This time, in this project, I chose to add label information (making the learning model become supervised) and use variational auto-encoder to training data, then compare different results and find the best model. I proposed a model combining the above two ideas called the Variational Auto Encoder Supervised DNGR (VAESDNGR). Various evaluation criterion has been used to assess the classification performances and the t-distributed stochastic neighbor embedding (t-SNE) method was used to make the results visualization. Through the experiments on the limited Wiki dataset, I concluded that the three improved models perform better than baseline and VAESDNGR performs best among these four.

## About the Dataset
For consistency with previous experiments, I still use the Wiki dataset. As described in my previous project, the Wiki dataset provided by openNE toolkit [https://github.com/thunlp/OpenNE](https://github.com/thunlp/OpenNE) for node classification
task, which contains 2405 web pages from 19 categories and 17981 links among them.

#### About the details of this project, please read the PDF file: [VAE Supervised DNGR](https://github.com/Mandy001/VAESDNGR/blob/master/VAE%20Supervised%20DNGR.pdf) in the same repository.
