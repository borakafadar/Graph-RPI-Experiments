_Briefings in Bioinformatics_ , 2025, **26(3)** , bbaf292 

**https://doi.org/10.1093/bib/bbaf292 Problem Solving Protocol** 



# **Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies** 

Jiahui Guan 1,2,‡, Lantian Yao 1,3,‡, Peilin Xie1,3,‡, Zhihao Zhao1, Dian Meng2, Tzong-Yi Lee 4,5,*, Junwen Wang 2,*, 

Ying-Chih Chiang<sup>1,3,</sup> * 

1Kobilka Institute of Innovative Drug Discovery, School of Medicine, The Chinese University of Hong Kong, Shenzhen, 2001 Longxiang Road, 518172, Shenzhen, China 

2Division of Applied Oral Sciences and Community Dental Care, Faculty of Dentistry, The University of Hong Kong, 34 Hospital Road, Hong Kong SAR, China 

3School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen, 2001 Longxiang Road, 518172, Shenzhen, China 

4Institute of Bioinformatics and Systems Biology, National Yang Ming Chiao Tung University, Hsinchu, Taiwan 

5Center for Intelligent Drug Systems and Smart Bio-Devices (IDS2B), National Yang Ming Chiao Tung University, 300 Hsinchu, Taiwan 

*Corresponding authors. Ying-Chih Chiang, E-mail: chiangyc@cuhk.edu.cn; Junwen Wang, E-mail: junwen@hku.hk; Tzong-Yi Lee, E-mail: leetzongyi@nycu.edu.tw ‡Jiahui Guan, Lantian Yao and Peilin Xie contributed equally to this work. 

#### Abstract 

RNA–protein interactions (RPIs) are essential for many biological functions and are associated with various diseases. Traditional methods for detecting RPIs are labor-intensive and costly, necessitating efficient computational methods. In this study, we proposed a novel sequence-based RPI prediction framework based on graph neural networks (GNNs) that addressed key limitations of existing methods, such as inadequate feature integration and negative sample construction. Our method represented RNAs and proteins as nodes in a unified interaction graph, enhancing the representation of RPI pairs through multi-feature fusion and employing selfsupervised learning strategies for model training. The model’s performance was validated through five-fold cross-validation, achieving accuracy of 0.880, 0.811, 0.950, 0.979, 0.910, and 0.924 on the RPI488, RPI369, RPI2241, RPI1807, RPI1446, and RPImerged datasets, respectively. Additionally, in cross-species generalization tests, our method outperformed existing methods, achieving an overall accuracy of 0.989 across 10 093 RPI pairs. Compared with other state-of-the-art RPI prediction methods, our approach demonstrates greater robustness and stability in RPI prediction, highlighting its potential for broad biological applications and large-scale RPI analysis. 

Keywords: RNA–protein interactions; graph neural networks; multi-feature fusion; self-supervised learning 

## Introduction 

RNA–protein interactions (RPIs) are fundamental to numerous cellular processes, playing pivotal roles in transcription, translation, splicing, chromatin remodeling, and gene regulation. These interactions are crucial for maintaining cellular homeostasis and facilitating the intricate execution of biological functions [1, 2]. The vital role of RPIs in diverse biological pathways highlights their importance in both normal cellular physiology and disease mechanisms [3]. Disruptions in RPIs are associated with various diseases, including amyotrophic lateral sclerosis, multiple forms of cancer, and immune disorders [4]. Understanding the complex mechanisms governing RPIs is essential for uncovering fundamental biological processes and may offer new pathways for therapeutic interventions [5]. Therefore, identifying RPIs is crucial in molecular biology and biomedicine [6]. 

Traditional experimental methods for detecting RPIs, although precise and informative, are often labor-intensive, time-consuming, and costly [3]. These drawbacks limit their feasibility in high-throughput studies and large-scale applications. Techniques such as immunoprecipitation, cross-linking, and RNA affinity 

purification demand extensive manual effort and are difficult to scale [7]. Furthermore, the reproducibility and consistency of results can vary, complicating the experimental approach. To address these limitations, a variety of computational strategies have been proposed for RPI prediction [8]. These include physicsbased methods such as molecular docking and molecular dynamics simulations [9], which model the structural and energetic aspects of RNA-protein binding, as well as statistical and co-evolutionary approaches that identify correlated mutation patterns indicative of functional interactions [10]. These approaches rely on different types of input data and assumptions, and often serve complementary roles in RPI research. However, they also face certain limitations. For instance, physics-based simulations typically require detailed 3D structural information and are computationally expensive, making them difficult to scale for large numbers of interactions. Similarly, statistical methods may be constrained by alignment quality and phylogenetic bias. As a result, these approaches are generally not well-suited for rapid, high-throughput prediction. 

Machine learning and deep learning have been widely applied in bioinformatics to address these challenges, owing to their 

**Received:** February 18, 2025. **Revised:** May 28, 2025. **Accepted:** May 31, 2025 © The Author(s) 2025. Published by Oxford University Press. 

> This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License (https://creativecommons.org/ licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact reprints@oup.com for reprints and translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact journals.permissions@oup.com. 

2 | Guan _et al._ 

ability to learn complex patterns from high-dimensional biological data. For example, Zhao et al. [11] enhanced peptide and protein toxicity prediction by incorporating channel attention into convolutional neural networks, while Le [12] discussed the potential of GNN-based frameworks, such as EmerGNN, for predicting emerging drug interactions by leveraging biomedical network structures. In addition, Le et al. [13] proposed a new pipeline for predicting protein crystallization propensity using feature selection, dimensionality reduction, and machine learning models. 

In recent years, several computational methods for RPIs prediction have been proposed. Muppirala et al. [14] introduced RPISeq, which used RNA and protein sequence encoding vectors generated by conjoint triad features and input them into random forests and support vector machines (SVMs) for prediction. Zhang et al. [15] proposed LPI-CNNCP, which extracts features from RNA and protein sequences using high-order one-hot encoding, and employs a convolutional neural network with a copy-padding strategy to manage variable-length inputs. Li et al. [16] introduced Capsule-LPI, a multichannel capsule network framework for predicting long non-coding RNA (lncRNA)–protein interactions. It integrates sequence features, motif information, physicochemical properties, and secondary structure through dedicated featurelearning subnetworks, followed by a capsule network to capture complex feature relationships. Huang et al. [17] developed LPI-CSFFR, a CNN-based method that combines sequence, secondary structure, and physicochemical features of lncRNAs and proteins using serial fusion and feature reuse strategies, enabling comprehensive representation learning for effective interaction prediction. Yu et al. [18] proposed RPI-MDLStack, based on a stacking strategy, which initially extracted sequence, physicochemical, structural, and evolutionary information from RNA and protein sequences using eight feature extraction methods. It then employed the least absolute shrinkage and selection operator to eliminate redundancy and generate optimal features. The stacking strategy combined multiple machine learning and deep learning models as base classifiers to learn the best features. Finally, the prediction scores were input into a discriminative model for further training. Wang et al. [19] proposed RPI-CapsuleGAN, which incorporated a convolutional attention mechanism into a generative adversarial capsule network to construct a classifier model. Elastic net feature selection was used to screen high-correlation features, reducing model complexity and preventing overfitting, while achieving strong predictive performance. Sun et al. [20] developed LPI-SKMSC, a clusteringbased method for lncRNA–protein interaction prediction that addresses class imbalance. It uses segmented k-mer frequencies to extract global and local features, and CNN-based encoders to map them into multiple spaces. Prediction is based on the distances between encoded features and cluster centers across these spaces. 

However, these methods face two main limitations. First, the aforementioned models construct negative samples that are balanced with the number of positive samples using techniques like random selection and score ranking, which limit the training data [21, 22]. Second, previous approaches fail to integrate RNA and protein features, resulting in a lack of holistic representation of RPI pairs. Models based on graph neural networks (GNNs) can effectively address these challenges. In GNNs, both RNA and proteins are represented as nodes within a single graph, with edges denoting their interactions. This method allows for a more comprehensive representation of initial information by incorporating all RNA and protein elements within the same graph. Moreover, the features of nodes in GNNs can integrate, 

enabling a more holistic characterization of RPIs through feature fusion [23–25]. 

In this study, we proposed a novel RPI prediction model named Graph-RPI, which utilized sequence information of RNA and proteins to construct an interaction graph network. The overall architecture was illustrated in Fig. 1. Initially, various sequence feature extraction methods were employed to represent RNA and protein sequences. Subsequently, a graph network was constructed to model the interactions, with nodes representing RNA and proteins, and edges representing their interactions. The GNN encoder, edge decoder, and degree decoder were then used to predict interaction outcomes. Additionally, during training, we incorporated self-supervised learning strategies, enabling the model to automatically learn latent associations between RNA and proteins and optimize feature representation. This approach not only enhanced the model’s generalization capability but also reduced dependence on large amounts of labeled data, resulting in greater robustness and predictive performance across diverse datasets. 

Our research provides a more efficient method for early RPI identification, thereby reducing manual costs. Compared to existing methods, our model demonstrates significant improvements in predictive performance. With the advancement of machine learning-based predictive tools, our approach holds promise as a reliable method for RPI identification and can contribute to a broader understanding and treatment of diseases associated with RPIs. 

## Materials and methods 

### **Dataset preparation** 

In this study, we integrated multiple sources of RPI datasets to validate our proposed method. The datasets used for training and evaluating the model include RPI488, RPI369, RPI2241, RPI1807, and RPI1446. The RPI369 and RPI2241 datasets were collected by Muppirala et al. [14] from the Protein–RNA Interface Database (PRIDB) [26]. RPI369 contains 369 non-redundant RPI pairs excluding ribosomal proteins or ribosomal RNAs, whereas RPI2241 includes 2241 RPI pairs from 943 RNA-protein complexes, encompassing ribosomal RNAs, non-coding RNAs, messenger RNAs, and other RNAs. The RPI488 dataset consists of 243 RPI pairs, extracted by Pan et al. [27] from 18 RNA-protein complexes in the Protein Data Bank [28]. The RPI1807 dataset, constructed by Suresh et al. [29] from the Nucleic Acid Database (NDB) [30] and PRIDB, comprises 1807 RPI pairs. The RPI1446 dataset, screened by Zhang et al. [15] from the RPI2241, RPI1807, and RPI488 datasets, includes 1446 long non-coding RNA (length _>_ 200 nt)–protein interaction pairs. Furthermore, we constructed a comprehensive dataset named RPImerged by combining the five benchmarks, with all duplicate RPI pairs removed, to evaluate the performance of our model on a larger and more diverse dataset. Summaries of all datasets are provided in Table 1 and Supplementary Table S1. 

For model evaluation, we employed two validation strategies on the above datasets. First, we randomly split each dataset into a training set and a test set with an 8:2 ratio. In this setting, the test set consists of non-overlapping RNA-protein pairs, including both positive (RPI) and negative (non-RPI) samples, ensuring no data leakage from the training set. Second, to rigorously evaluate model performance while ensuring fold independence, we employed a five-fold cross-validation strategy based on hierarchical clustering of pairwise sequence similarity [31, 32]. Specifically, 

| 3 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 



<!-- Start of picture text -->
(A) Data collection and feature extraction<br>Database<br>NAC Kmer DPCP<br>PDB<br>PseDNC PCPseDNC\| CKSNAP<br>RNA sequence\|<br>NDB\|<br>PRIDB\| AAC] PAAC DDE<br>NPInter v3.0} Protein sequence CKSAAGP] QSOrder ESM-2<br>(C) Interaction mask and negative sampling (B) RNA-Protein interactions graph]<br>RNA node) RNA feature Initial interaction)<br>Mask interaction)<br>Protein node Protein feature\| Negative sampling\|<br>(D) Model construction and results output<br>Edge reconstruction]<br>Edge Prediction results\|<br>Decoder Edge<br>loss\| Interaction \|<br>.\|<br>GNN\|<br>Encoder<br>Node: »O@OOE Degree. No<br>Degree interaction\|<br>Decoder Degree:<br>Degree reconstruction\|<br><!-- End of picture text -->

Figure 1. The flowchart of this study, including: (A) Data collection and feature extraction. (B) RNA-protein representation graph construction. (C) Interaction mask and negative sampling. (D) Model construction and results output. 

Table 1. Overview of RPI datasets 

|**Datasets**|**Species**|**RPI pairs**|**non-RPI pairs**|**RNAs**|**Proteins**|
|---|---|---|---|---|---|
|RPI488|-|243|245|25|247|
|RPI369|-|369|-|332|338|
|RPI2241|-|2241|-|842|2043|
|RPI1807|-|1807|1436|1078|1807|
|RPI1446|-|1446|1560|325|1446|
|RPImerged|-|3928|5840|1477|4800|
|RPI_H|_Homo sapiens_|7317|-|1874|118|
|RPI_M|_Mus musculus_|1847|-|1939|60|
|RPI_S|_Saccharomyces cerevisiae_|670|-|80|108|
|RPI_E|_Escherichia coli_|183|-|39|40|
|RPI_D|_Drosophila melanogaster_|67|-|17|31|
|RPI_C|_Caenorhabditis elegans_|9|-|5|8|



4 | Guan _et al._ 

we first computed the pairwise similarity between each RNAprotein pair using the following scoring function: 





where _S_ RNA and _S_ protein are alignment-based similarity scores obtained using global sequence alignment [33]. For RNA, we used a scoring scheme with match = 1, mismatch = –1, gap open = –2, and gap extension = –0.5. For protein sequences, we used the BLOSUM62 substitution matrix with gap open = –10 and gap extension = –0.5 [34, 35]. The weight coefficient _α_ was set to 0.5 to equally balance RNA and protein contributions. Based on the similarity matrix, a distance matrix was computed as: 



where higher similarity corresponds to lower distance. We then applied hierarchical clustering using average linkage and partitioned the resulting dendrogram into five clusters, each serving as one-fold. This ensures that highly similar RNA-protein pairs are grouped into the same fold, effectively minimizing redundancy across folds and enabling a more stringent evaluation of generalization performance. Supplementary Fig. S1 presents bar plots of the mean pairwise sequence similarity within folds (intra-fold) and between folds (inter-fold) for each of the six RPI datasets. Across all datasets, the intra-fold similarity is clearly higher than the inter-fold similarity, indicating that the hierarchical clustering strategy effectively groups similar RNA–protein interaction pairs into the same fold while ensuring dissimilarity across different folds. In addition, Supplementary Fig. S2 reports intra- and interfold similarities computed separately based on RNA and protein alignments. In both cases, intra-fold similarity remains higher than inter-fold similarity, further supporting the validity of the fold construction procedure. 

To further assess the model’s ability to generalize to unseen data, we additionally introduced independent test sets derived from previous studies [18, 36]. These independent test sets include lncRNA–protein interactions from six species: _Homo sapiens_ (7317 pairs), _Mus musculus_ (1847 pairs), _Saccharomyces cerevisiae_ (670 pairs), _Escherichia coli_ (183 pairs), _Drosophila melanogaster_ (67 pairs), and _Caenorhabditis elegans_ (9 pairs), which serve to evaluate the cross-species transferability of the proposed framework. 

Specifically, these independent test sets were first constructed by collecting experimentally validated ncRNA–protein interactions from NPInter v3.0 [37] based on the corresponding species. For _H. sapiens_ , lncRNA–protein interaction pairs were further refined by retrieving lncRNA and protein sequence information from the GENCODE v29 and UniProt databases [38, 39], respectively. For _M. musculus_ , lncRNA annotations were obtained from the GENCODE M20 database [38]. For the other four species, lncRNA information was obtained from the NONCODE database [40]. 

### **Feature extraction for RNA sequence** 

We utilize six feature extraction methods for RNA sequence representation including nucleic acid composition (NAC), Occurrence frequency of _k_ neighboring nucleic acids (Kmer), dinucleotide physicochemical properties (DPCP), pseudo dinucleotide composition (PseDNC), parallel correlation pseudo dinucleotide composition (PCPseDNC), and composition of _k_ -spaced nucleic acid pair (CKSNAP). 

#### _Nucleic acid composition_ 

The NAC calculates the frequency of each nucleic acid type in a RNA sequence. The frequencies of all four natural nucleic acids can be calculated as follows: 



where _N(t)_ is the number of nucleic acid type _t_ , while _N_ is the length of a nucleotide sequence. 

#### _Occurrence frequency of k neighboring nucleic acids_ 

The kmer represents the occurrence frequencies of k neighboring nucleic acids [41]. The Kmer type (k=3) descriptor can be calculated as follows: 



where _N(t)_ is the number of kmer type _t_ , while _N_ is the length of a nucleotide sequence. 

#### _Dinucleotide physicochemical properties_ 

The DPCP descriptor can be devised as follows: 



where DPCP _i_ represents one of the physicochemical properties of a dinucleotide _i_ , and _fi_ represents the frequency of dinucleotide _i_ in the sequence. The physicochemical properties for RNA dinucleotide are listed in Supplementary Table S2 [42]. 

#### _Pseudo dinucleotide composition_ 

The PseDNC incorporates contiguous local sequence-order information and global sequence-order information into the feature vector of the RNA sequence [43]. The PseDNC encoding is defined as follows: 



where the vector _D_ contains the features of the nucleotide sequence. Each feature _dk_ is calculated as follows: 



where _fk_ is the normalized occurrence frequency of dinucleotide type _k_ in the nucleotide sequence.<sup>�16</sup> _i_ =1<sup>_fi_isthetotalsumof</sup> occurrence frequencies of all dinucleotides. _w_ is the weight factor, ranging from 0 to 1. _θj_ is the _j_ -tier correlation factor, defined as follows: 



where _L_ is the length of the nucleotide sequence. _Ri_ is the dinucleotide at position _i_ . _�(RiRi_ + _j)_ is the correlation function, defined as follows: 



| 5 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

where _μ_ is the number of physicochemical indices. _Pu(RiRi_ + _j)_ is the numerical value of the _u_ -th physicochemical index of the dinucleotide _Ri_ at position _i_ . _Pu(RjRi_ + _j)_ is the corresponding value of the dinucleotide _Rj_ at position _j_ . 

of PAAC is governed by two pivotal parameters: the counted-rank correlation, denoted as _λ_ , and the weight factor, denoted as _ω_ . The dimensionality of the feature is determined by _λ_ and is expressed as (20 + _λ_ ). A larger value of _ω_ accentuates the significance of sequence order in the PAAC feature [46]. 

#### _Parallel correlation pseudo dinucleotide composition_ 

The PCPseDNC builds upon the PseDNC by incorporating parallel sequence-order correlations into the feature vector [44]. The main difference lies in the definition of the correlation factor _θj_ , which in PCPseDNC is modified to reflect sequence-order correlations between all the most contiguous dinucleotides along the sequence: 



All other aspects of PCPseDNC follow the same structure and calculation as PseDNC, with the modified _θj_ enhancing the ability to capture parallel correlations across the sequence. 

#### _Composition of k-spaced nucleic acid pair_ 

The CKSNAP calculates the frequency of nucleic acid pairs separated by any _k_ nucleic acid ( _k_ = 0, 1, 2, _. . ._ , 5). Taking _k_ = 0 as an example, there are 16 0-spaced nucleic acid pairs (i.e. “AA,” “AC,” “AG,” “AU,” “CA,” “CC,” ‘CG,” “CU,” “GA,” “GC,” “GG,” “GU,” “UA,” “UC,” “UG,” and “UU”). A feature vector can be defined as follows: 



The value of each descriptor denotes the composition of the corresponding nucleic acid pair in the RNA sequence [45]. For instance, if the nucleic acid pair “AA” appears _m_ times in the nucleotide sequence, the composition of the nucleic acid pair “AA” is equal to _m_ divided by the total number of 0-spaced nucleic acid pairs ( _Ntotal_ ) in the nucleotide sequence. For _k_ = 0, 1, 2, 3, 4, and 5, the value of _Ntotal_ is _P_ − 1, _P_ − 2, _P_ − 3, _P_ − 4, _P_ − 5, and _P_ − 6 for a nucleotide sequence of length _P_ , respectively. 

### **Feature extraction for protein sequence** 

We utilize six feature extraction methods for protein sequence representation including amino acid composition (AAC), pseudoamino acid composition (PAAC), composition of _k_ -spaced amino acid group pairs (CKSAAGP), quasi-sequence order (QSOrder), dipeptide deviation from expected mean (DDE), and evolutionary scale modeling (ESM-2). 

#### _Amino acid composition_ 

The AAC calculates the frequency of occurrence of each amino acid in a sequence. The frequencies of all 20 natural amino acids can be calculated as follow: 



where _N(t)_ is the number of amino acid type _t_ in a sequence, and _L_ is the length of the sequence. 

#### _Pseudo-amino acid composition_ 

The PAAC provides a sophisticated representation of amino acid sequences by integrating both the inherent patterns observed in peptides and the sequential order of amino acids. The formulation 

#### _Composition of k-spaced amino acid group pairs_ 

The CKSAAGP calculates the frequency of occurrence of a particular amino acid pair with the same physicochemical properties separated by _k_ arbitrary residues. CKSAAGP categorizes 20 amino acids into five groups according to their physicochemical properties (Supplementary Table S3): aliphatic (g1), aromatic (g2), positively charged (g3), negatively charged (g4), and uncharged (g5) [45, 47]. For example, for a peptide sequence of length _L_ , assuming that _k_ is zero, there are 25 _(_ = 5 × 5 _)_ 0-spaced pairs, namely _g_ 1 _g_ 1, _g_ 1 _g_ 2,..., _g_ 5 _g_ 5. Thus, the 0-spaced CKSAAGP descriptor can be defined as follow: 



where _Ngi_ , _gj_ denotes the number of occurrences of each group pair _gi_ , _gj_ . 

#### _Quasi-sequence order_ 

The QSOrder employs a quasi-sequence descriptor approach. It analyzes the overall composition of amino acids and integrates information about their position in the protein sequence, fusing local and global features [48, 49]. 

For each amino acid type, a quasi-sequence-order descriptor can be defined as: 



where _fr_ is the normalized occurrence of amino acid type _r_ and _w_ is a weighting factor ( _w_ = 0.1). _nlag_ and _τd_ have the same definitions as described above. These are the first 20 quasi-sequence-order descriptors. The other 30 quasi-sequence-order descriptors are defined as: 



#### _Dipeptide deviation from expected mean_ 

The DDE offers a statistical perspective on the dipeptide distribution in proteins. It measures the deviation between the observed frequency of a dipeptide and its theoretically expected occurrence based on codon frequencies. DDE can reveal patterns or biases in protein composition. If a specific dipeptide appears more or less frequently than expected, it suggests that there might be evolutionary, functional, or structural reasons for this deviation. This encoding is formulated by computing three parameters: Dipeptide Composition (Dc), Theoretical Mean (Tm), and Theoretical Variance (Tv) [50, 51]. 

The measure _Dc(r_ , _s)_ for the dipeptide “rs” is defined as: 



6 | Guan _et al._ 

where _Nrs_ represents the number of dipeptides characterized by amino acid types _r_ and _s_ , while _N_ denotes the length of the protein or peptide sequence. 

The _Tm(r_ , _s)_ is articulated as: 



where _Cr_ and _Cs_ are the number of codons encoding the first and second amino acids of the dipeptide “rs,” respectively. _CN_ represents the total count of viable codons, excluding the three stop codons, amounting to 61. 

The variance for the dipeptide “rs,” _Tv(r_ , _s)_ , is expressed as: 



Lastly, the deviation _DDE(r_ , _s)_ is computed as: 



#### _Evolutionary scale modeling_ 

The ESM-2 pre-trained model, built on the Transformer architecture, is an advanced protein language model that has undergone unsupervised training on a large-scale protein sequence database. This training enables the model to effectively capture and extract both structural and functional features from protein sequences. When a protein sequence is input into the ESM-2 model, it encodes the sequence into a high-dimensional latent vector that represents the protein’s structural and functional properties. This latent vector, enriched with evolutionary, structural, and functional information, can then be utilized as input for various downstream tasks, such as protein structure prediction, functional annotation [52]. 

### **Framework of the proposed model** 

The proposed model mainly adopts the architecture of the graph autoencoder [53, 54]. Graph autoencoders are well suited for RPI graphs and can preserve and learn the structural information of graphs. Similar to traditional autoencoders, our model consists of two main components: an encoder and a decoder [55]. Additionally, we adopt a masking strategy based on the Bernoulli distribution to effectively mitigate the impact of noisy data in the graph. Furthermore, we implement a degree-based decoder to better capture the latent structure of the graphs. 

Supplementary Table S4 presents an explicit overview of the dimensionality of feature representations throughout the encoding and decoding process, including the input features, GNN layers, edge decoder, and degree decoder. 

#### _Encoder_ 

In our model, we adopt Graph Attention Network (GATConv) and Graph Isomorphism Network (GINConv) as the encoder layers [56, 57]. The core idea is to update the node representation by iteratively aggregating the node’s neighbor information. In the task of modeling the RNA-protein relationship, each RNA node (or protein node) is aggregated by its own features and the features of the associated protein nodes (or RNA nodes) at each iteration. Multi-layer perceptron (MLP) is adopted to update the aggregated node representations. 

First, GATConv aggregates node features through an attention mechanism. The input features for RNA and protein nodes are denoted as _R_<sup>0</sup> _a_<sup>and</sup><sup>_P_0</sup> _b_<sup>, respectively. Each node calculates attention</sup> coefficients for its neighbor nodes and aggregates neighbor features weighted by these coefficients. The feature update for RNA nodes using GATConv is given by: 



where _αab_ is the attention coefficient between the RNA node _a_ and the protein node _b_ , and _W_ is a learnable weight matrix. Similarly, the feature update for protein nodes is: 



Next, GINConv learns non-linear transformations of node features through a MLP. Each node aggregates its own features and the features of its neighbors and then applies an MLP for nonlinear transformation. The feature update for RNA nodes using GINConv is: 



where _ϵ_ 2 is a learnable scalar parameter. Similarly, the feature update for protein nodes is: 



After each convolutional layer, batch normalization and an ELU activation function are applied to stabilize training and introduce non-linearity. 

#### _Decoder_ 

The decoder in our model consists of two main components: the edge decoder and the degree decoder. Each component comprises two layers of MLPs. These decoders are designed to reconstruct distinct structural aspects of the RPI graph. 

The edge decoder is responsible for reconstructing the RNAprotein association matrix. Given the latent embeddings _z_ obtained from the encoder, the interaction between a pair of RNA and protein nodes _(i_ , _j)_ is modeled by first computing the element-wise product of their embeddings, denoted as _zi_ ⊙ _zj_ . This interaction vector is subsequently passed through a two-layer MLP to predict the probability of an association: 



The degree decoder is designed to reconstruct the connectivity degree of each node, representing the number of associated neighbors. For each node embedding _zi_ , a two-layer MLP is applied to regress its degree: 



| 7 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

These two decoders are jointly optimized to capture both edgelevel and node-level structural patterns, thereby enabling accurate reconstruction of the RPI topology. 

#### _Mask and negative sampling strategies_ 

To address the presence of noise in the RNA-protein graph, we devised a self-supervised masking strategy based on a probability distribution [58]. The core idea is to mask certain associations in the RNA-protein graph according to a probabilistic model. In each training iteration, the model employs a Bernoulli distribution to randomly mask some associations in the RNA-protein graph. Specifically, the set of known associations is sampled according to the Bernoulli distribution: 



where _p_ is a probability value between 0 and 1, indicating the masking ratio of the graph. By setting different _p_ values, varying numbers of edges are masked. Subsequently, the edge decoder and degree decoder work together to reconstruct these masked associations. 

In addition to the masking strategy, we employ a negative sampling technique to further enhance the model’s robustness [59]. Specifically, during each training epoch, negative samples are dynamically generated by randomly pairing RNA and protein nodes that do not form known or masked interactions in the current graph. These RNA-protein pairs are not labeled as interacting in the dataset and are treated as negatives during training. 

Given the masked edge set _E_<sup>ˆ</sup> , the negative samples are generated as: 



To optimize the model, we define two objective functions corresponding to the edge and degree decoders. 

The edge prediction loss is formulated using the binary crossentropy function, which quantifies the discrepancy between the predicted association probability _p_ and the binary ground truth label _y_ ∈{0, 1}: 



where _p_ is the predicted probability obtained from the edge decoder. 

The degree prediction loss is defined using the mean squared error between the predicted degrees and the actual degrees of all nodes: 



where _x_ denotes the total number of nodes, _yi_ is the ground truth degree of node _i_ , and _d_<sup>ˆ</sup> _i_ is the corresponding prediction from the degree decoder. 

The total training loss is expressed as a weighted combination of the two loss terms: 



where _α_ is a hyperparameter controlling the relative contribution of the degree loss to the overall objective. 

### **Model training and experimental setup** 

where _k_ denotes the number of negative samples. The NegativeSampling _(_ · _)_ function selects _k_ node pairs across the RNAprotein bipartite space such that _(ri_ , _pj)_ ∈ _/ E_<sup>ˆ</sup> , ensuring that none of the sampled negative pairs overlap with known or masked positive interactions. 

To ensure the validity of the training signals, the negative sampling process in our framework is deliberately conducted after the masking operation. In this design, a subset of known interactions is first masked and removed from the graph. Negative sampling is then performed on the remaining graph, which excludes both observed and masked edges. 

This guarantees that all sampled negative edges are truly absent from the current graph, including those removed for selfsupervised prediction. As a result, there is no overlap between positive and negative samples, effectively preventing label leakage and ensuring stable and unbiased model training. 

#### _Prediction and loss function_ 

Following the decoding process, the model reconstructs the RPI graph. The final representations of RNA and protein nodes are denoted as _Ri_ and _Pj_ , respectively. The association probability between RNA node _i_ and protein node _j_ is computed using the dot product of their embeddings: 



where _Ri_ and _Pj_ represent the learned embeddings of the _i_ -th RNA and _j_ -th protein nodes, respectively, and _Pi_ , _j_ denotes the predicted likelihood of interaction. 

To ensure reliable predictive performance, our model was trained for 500 epochs with an initial learning rate of 0.001. We implemented a learning rate scheduler to reduce the rate by a factor of 0.95 every five epochs, promoting optimal convergence and fitting. The Adam optimizer was employed with a weight decay of 5e– 5 to ensure stable and efficient parameter updates. To prevent overfitting, dropout with a rate of 0.5 was incorporated into the encoder and decoder modules, randomly deactivating neurons during training to reduce reliance on specific features. Additionally, the loss function includes a trade-off hyperparameter _α_ , which was set to 0.5 to balance edge and degree reconstruction tasks. The masking ratio _p_ for the self-supervised strategy was set to 0.4. These settings were used as the default configuration for all main experiments. This includes the comparisons of different feature fusion strategies, GNN layer types, and negative sampling methods. 

Our experimental development was based on Python [60], and we used the PyTorch framework to construct and train the model [61]. Lastly, we conducted the training process with 2 × Nvidia 2080 Ti GPUs. 

### **Evaluation metrics** 

In this study, we utilize accuracy (ACC), recall (RE), precision (PRE), specificity (SPE), F1-score (F1), Matthew’s correlation coefficient (MCC), and the area under the receiver operating characteristic curve (AUC) to evaluate the performance of the presented method [62]. Additionally, the false positive rate (FPR) and false negative rate (FNR) are reported to evaluate the model’s misclassification behavior across different datasets. These metrics are defined as 

8 | Guan _et al._ 

Table 2. Performance on the test sets of RPIs datasets 

|**Dataset**|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|
|---|---|---|---|---|---|---|---|
|RPI488|0.878|0.857|0.894|0.898|0.875|0.756|0.918|
|RPI369|0.873|0.887|0.863|0.859|0.875|0.747|0.813|
|RPI2241|0.955|0.949|0.960|0.961|0.954|0.910|0.989|
|RPI1807|0.985|0.983|0.986|0.986|0.985|0.970|0.990|
|RPI1446|0.953|0.955|0.952|0.952|0.954|0.907|0.986|
|RPImerged|0.933|0.947|0.921|0.919|0.934|0.866|0.973|



follows: 

















where TP, TN, FP, and FN denote true positives, true negatives, false positives, and false negatives, respectively. 

## Results and discussion **Performance evaluation** 

Our model’s performance across the test sets of RPI488, RPI369, RPI2241, RPI1807, RPI1446, and RPImerged is summarized in Table 2 and Supplementary Table S5. On the RPI488 dataset, the model achieved an ACC of 0.878 and an MCC score of 0.756. For RPI369, the model obtained an ACC of 0.873 and an MCC score of 0.747. The performance on the RPI2241 dataset yielded an ACC of 0.955 and an MCC score of 0.910. On the RPI1807 dataset, the model reached an ACC of 0.985 and an MCC score of 0.970. The ACC and MCC for RPI1446 were 0.953 and 0.907, respectively. Additionally, on the RPImerged dataset, the model achieved an 

ACC of 0.933 and an MCC score of 0.866, demonstrating its robustness on a larger and more heterogeneous dataset. 

To further evaluate the generalizability of our model, we conducted five-fold cross-validation on the RPI datasets. The average performance across the folds is summarized in Table 3 and Supplementary Table S6. On the RPI488 dataset, the model achieved an average ACC of 0.880 and an average MCC score of 0.767. For the RPI369 dataset, the average ACC was 0.811, with an MCC score of 0.630. The model achieved an ACC of 0.950 and an MCC score of 0.903 on the RPI2241 dataset. On the RPI1807 dataset, the average ACC and MCC score were 0.979 and 0.958, respectively. For the RPI1446 dataset, the model reached an average ACC of 0.910 and an MCC score of 0.821. Additionally, on the RPImerged dataset, the model achieved an average ACC of 0.924 and an MCC score of 0.849. These results confirm that the proposed method maintains consistently strong performance under crossvalidation, demonstrating good generalization across multiple datasets. 

Notably, the MCC scores on the RPI488 and RPI369 datasets are relatively lower compared to the other datasets. This is mainly due to two factors. First, both datasets contain a limited number of interaction pairs, leading to sparse interaction networks with reduced structural information. Second, the small number of positive samples intensifies the class imbalance, which significantly affects MCC because it is highly sensitive to false positives and false negatives. These factors jointly contribute to the lower MCC values observed on these two datasets. 

### **Ablation experiments** 

#### _Evaluation of mask ratio and alpha value settings_ 

In this study, we evaluated the impact of the mask ratio and the hyperparameter _α_ on the model’s predictive performance. Specifically, we first conducted five-fold cross-validation on each of the benchmark datasets and calculated the average performance on the test folds. These per-dataset averages were then further averaged across all datasets to produce a single representative curve for each hyperparameter setting. The results are summarized in Supplementary Fig. S3. 

As shown in Supplementary Fig. S3A, the mask ratio has a substantial impact on model performance across multiple evaluation metrics. The model achieves the best overall performance when the mask ratio is set to 0.4, particularly in terms of MCC, AUC, and ACC. This suggests that moderate masking improves the model’s ability to learn informative patterns by reducing noise while preserving critical structural information. Performance generally increases as the mask ratio increases from 0.1 to 0.4, but further increases beyond 0.5 lead to a consistent decline in most metrics, indicating that excessive masking can remove important connections in the graph and impair learning. Furthermore, at a mask ratio of 0.0, the lack of a reconstruction objective causes a sharp drop in AUC and ACC, while at 1.0 the encoder cannot 

| 9 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

Table 3. Average performance of five-fold cross-validation on multiple RPI datasets, shown as mean ± standard deviation across the <u>five folds</u> 

|**Dataset**|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|
|---|---|---|---|---|---|---|---|
|RPI488|0.880±0.052|0.864±0.095|0.897±0.052|0.897±0.066|0.877±0.055|0.767±0.100|0.923±0.046|
|RPI369|0.811±0.060|0.745±0.098|0.859±0.061|0.876±0.058|0.795±0.069|0.630±0.116|0.772±0.071|
|RPI2241|0.950±0.027|0.927±0.050|0.973±0.008|0.974±0.008|0.949±0.029|0.903±0.051|0.987±0.009|
|RPI1807|0.979±0.007|0.986±0.009|0.973±0.009|0.972±0.009|0.979±0.007|0.958±0.014|0.986±0.008|
|RPI1446|0.910±0.029|0.884±0.040|0.931±0.023|0.935±0.022|0.907±0.030|0.821±0.057|0.969±0.011|
|RPImerged|0.924±0.036|0.934±0.063|0.915±0.017|0.914±0.013|0.924±0.039|0.849±0.073|0.969±0.022|



see any positive links and performs at near-random levels. These extreme cases suggest that mask ratios between 0.1 and 0.9 offer a balance between information availability and training difficulty, which leads to improved self-supervised learning performance. 

Supplementary Fig. S3B illustrates the effect of the decoder loss weight _α_ on performance. The model achieves peak performance when _α_ = 0.5, with noticeable improvements in AUC, MCC, and ACC. Values of _α_ in the range of 0.2 to 0.6 maintain stable performance across most metrics. However, setting _α_ to extremes (such as 0.0 or 1.0) results in performance degradation, particularly in MCC, indicating that an overly imbalanced weighting between the reconstruction loss and the main classification objective can harm the model’s predictive capacity. These findings highlight the importance of tuning _α_ within a moderate range. 

#### _Analysis of different feature fusion_ 

To highlight the importance of feature fusion in RPI prediction, this study explored the integration of different RNA and protein features from the RPImerged dataset before inputting them into the model for prediction. Specifically, six types of features were extracted from RNA sequences: NAC, Kmer, DPCP, PseDNC, PCPseDNC, and CKSNAP. From protein sequences, five types of descriptors were extracted: AAC, PAAC, CKSAAGP, QSOrder, and DDE, as well as ESM-2 embeddings derived from a pre-trained protein language model. Each RNA feature was individually combined with a protein feature to form 36 distinct RNA-protein feature combinations. 

As presented in Table 4, among the 36 single-feature combinations, the pairing of Kmer and ESM-2 achieved the best overall performance, with an ACC of 0.883 and an MCC of 0.770. In contrast, the combination of PseDNC and ESM-2 attained the highest RE of 0.940. These results demonstrate the complementarity between different RNA and protein feature types. 

Furthermore, when all RNA and protein features were integrated into a combined representation, the model achieved the best overall performance, with an ACC of 0.924 and an MCC of 0.849. This significantly outperformed any individual RNA-protein feature pair. The superior performance of the fused features confirms that feature integration can effectively capture comprehensive information from both RNA and protein sequences, thereby enhancing the predictive capability of the RPI model. 

#### _Comparison of different GNN layers_ 

We conducted ablation experiments on the RPImerged dataset to evaluate the impact of different GNN architectures on model performance. The configurations tested included a baseline model without GNN, in which node features are transformed by fully connected layers without incorporating neighborhood information, as well as several widely used GNN layers, including Graph Convolutional Network, Graph SAmple and aggreGatE, Graph Isomorphism Network (GIN), Graph Attention Network 

(GAT), and their combined models [23]. As shown in the Fig. 2 and Supplementary Table S7, incorporating GNN layers consistently improved performance compared to the no-GNN baseline, indicating that neighborhood aggregation plays an important role in learning informative representations for RPI prediction. 

Among all configurations, the combination of GAT and GIN achieved the best performance, with an ACC of 0.924 and an MCC of 0.849. This superior performance can be attributed to the complementary strengths of these models. In the initial phase, GAT utilizes an attention mechanism to adaptively assign different weights to each node’s neighbors. This approach allows for a weighted average during the initial feature extraction phase, based on the importance of neighboring nodes, thereby identifying key nodes and edges. Subsequently, GIN further integrates these features through its powerful aggregation method, which retains more information. GAT’s ability to integrate information from neighboring nodes in the initial stage is complemented by GIN’s detailed fusion and processing of this information in the second stage, resulting in more refined feature representations. 

#### _Investigation of different negative sampling strategies_ 

To investigate the influence of negative sampling strategies on model performance, we conducted a comparative evaluation of five strategies: without negative sampling, random negative sampling, file-based negative sampling, mixed negative sampling, and hard negative sampling. The experiments were performed on the RPImerged dataset, and the results are summarized in Table 5. 

For the setting without negative sampling, negative pairs are derived from the initial data split and the masking mechanism in the self-supervised module, without any additional sampling strategy. In the random sampling strategy, negative RNA-protein pairs are uniformly sampled from unobserved interactions, excluding those present in the test set. File-based sampling uses a predefined set of negative pairs constructed based on prior annotations. Mixed sampling combines equal proportions of filebased and randomly selected negatives. Hard negative sampling identifies negative samples that are most similar to positive pairs in the feature space, thus posing greater difficulty for the model to distinguish. 

Among the five strategies, random sampling consistently achieved the best performance across most evaluation metrics, including ACC (0.924), F1 score (0.924), and AUC (0.969). Hard negative sampling also achieved promising results, though slightly lower than random sampling. The setting without negative sampling also yielded competitive performance, with an ACC of 0.903, MCC of 0.782, and F1 score of 0.931, indicating that the model maintained strong classification capability even without negative sampling. This result suggests that the negative samples generated through the masking mechanism in the self-supervised module provided sufficient contrastive signal during training. However, the improvement in ACC from 0.903 

10 | Guan _et al._ 

Table 4. Performance comparison of different feature fusion on RPImerged dataset, with results shown as mean ± standard deviation across five-fold cross-validation 

|**RNA Feature**|**Protein**<br>|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|
|---|---|---|---|---|---|---|---|---|
||**Feature**||||||||
|NAC|AAC|0.618±0.077|0.289±0.196|0.870±0.065|0.946±0.044|0.400±0.213|0.311±0.117|0.798±0.040|
||CKSAAGP|0.688±0.085|0.570±0.226|0.750±0.084|0.806±0.137|0.625±0.171|0.396±0.158|0.781±0.046|
||DDE|0.818±0.042|0.717±0.083|0.899±0.023|0.920±0.018|0.796±0.057|0.651±0.075|0.889±0.033|
||ESM-2|0.880±0.025|0.932±0.041|0.844±0.018|0.828±0.021|0.885±0.025|0.764±0.052|0.918±0.019|
||PAAC|0.694±0.074|0.472±0.163|0.844±0.037|0.916±0.026|0.592±0.159|0.431±0.127|0.846±0.024|
||QSOrder|0.704±0.073|0.548±0.174|0.796±0.061|0.860±0.066|0.637±0.128|0.433±0.134|0.812±0.028|
|Kmer|AAC|0.694±0.053|0.472±0.101|0.846±0.056|0.916±0.026|0.602±0.091|0.432±0.100|0.827±0.054|
||CKSAAGP|0.699±0.080|0.514±0.161|0.807±0.049|0.884±0.013|0.620±0.135|0.427±0.144|0.805±0.041|
||DDE|0.812±0.039|0.703±0.090|0.900±0.020|0.921±0.021|0.787±0.055|0.642±0.068|0.881±0.044|
||ESM-2|0.883±0.023|0.936±0.029|0.846±0.019|0.829±0.020|0.889±0.022|0.770±0.047|0.922±0.020|
||PAAC<br>|0.688±0.093|0.461±0.214|0.834±0.057|0.915±0.046|0.568±0.224|0.416±0.167|0.831±0.041|
||QSOrder|0.684±0.058|0.475±0.125|0.813±0.041|0.893±0.025|0.594±0.106|0.406±0.103|0.820±0.037|
|DPCP|AAC|0.719±0.090|0.532±0.188|0.843±0.073|0.906±0.039|0.640±0.154|0.473±0.163|0.822±0.060|
||CKSAAGP|0.698±0.059|0.543±0.108|0.786±0.072|0.852±0.052|0.639±0.088|0.417±0.116|0.769±0.059|
||DDE|0.772±0.091|0.629±0.182|0.875±0.040|0.915±0.016|0.723±0.138|0.570±0.160|0.836±0.049|
||ESM-2|0.843±0.044|0.870±0.089|0.825±0.016|0.817±0.015|0.846±0.049|0.690±0.091|0.899±0.024|
||PAAC|0.698±0.018|0.496±0.032|0.831±0.023|0.899±0.015|0.621±0.029|0.432±0.036|0.799±0.015|
||QSOrder|0.682±0.060|0.489±0.127|0.791±0.037|0.875±0.009|0.599±0.108|0.394±0.108|0.800±0.039|
|PseDNC|AAC<br>|0.641±0.066<br>|0.365±0.167<br>|0.824±0.092<br>|0.917±0.053<br>|0.487±0.150<br>|0.341±0.114<br>|0.767±0.091<br>|
||CKSAAGP|0.689±0.087|0.565±0.224|0.751±0.077|0.813±0.116|0.623±0.181|0.396±0.161|0.788±0.055|
||DDE|0.824±0.046|0.734±0.093|0.895±0.025|0.914±0.019|0.804±0.062|0.661±0.083|0.891±0.036|
||ESM-2|0.880±0.024|0.940±0.032|0.840±0.021|0.821±0.026|0.887±0.023|0.767±0.049|0.920±0.019|
||PAAC|0.697±0.075|0.472±0.159|0.852±0.039|0.923±0.022|0.595±0.157|0.439±0.130|0.852±0.036|
||QSOrder<br>|0.704±0.072<br>|0.549±0.171<br>|0.795±0.061<br>|0.859±0.066<br>|0.639±0.125<br>|0.433±0.132<br>|0.810±0.029<br>|
|PCPseDNC|AAC|0.659±0.077|0.396±0.182|0.836±0.079|0.922±0.046|0.518±0.171|0.374±0.134|0.767±0.114|
||CKSAAGP|0.686±0.083|0.562±0.228|0.755±0.086|0.811±0.141|0.620±0.170|0.396±0.154|0.782±0.050|
||DDE|0.824±0.047|0.735±0.098|0.894±0.025|0.914±0.020|0.804±0.064|0.661±0.085|0.891±0.036|
||ESM-2|0.879±0.026|0.934±0.036|0.841±0.020|0.824±0.022|0.885±0.025|0.763±0.053|0.919±0.019|
||PAAC|0.699±0.075|0.489±0.170|0.840±0.036|0.910±0.032|0.604±0.160|0.438±0.127|0.848±0.031|
||QSOrder|0.701±0.071|0.543±0.171|0.793±0.060|0.859±0.066|0.633±0.125|0.428±0.130|0.812±0.029|
|CKSNAP|AAC|0.645±0.056|0.368±0.145|0.853±0.124|0.922±0.067|0.495±0.136|0.356±0.102|0.794±0.070|
||CKSAAGP|0.699±0.081|0.524±0.162|0.800±0.062|0.875±0.029|0.625±0.128|0.426±0.149|0.802±0.044|
||DDE|0.810±0.034|0.695±0.073|0.902±0.020|0.924±0.018|0.783±0.047|0.638±0.059|0.882±0.040|
||ESM-2<br>|0.880±0.023<br>|0.938±0.037<br>|0.841±0.017<br>|0.823±0.020<br>|0.887±0.023<br>|0.766±0.048<br>|0.917±0.019<br>|
||PAAC|0.686±0.091|0.489±0.250|0.809±0.065|0.883±0.089|0.575±0.229|0.409±0.162|0.816±0.033|
||QSOrder|0.684±0.055|0.468±0.110|0.821±0.039|0.900±0.019|0.592±0.099|0.408±0.099|0.819±0.037|
|Combined|Combined||||||||
|features|features|0.924±0.036|0.934±0.063|0.915±0.017|0.914±0.013|0.924±0.039|0.849±0.073|0.969±0.022|
||||Performa|nce comparison of di|fferent GNN layers\|||||
||1.0||||||||
||09]||||||||
||08||||||||
||5||||||||
||0.7||||||Metric||
||||||||ACC<br>||
||||||||RE\|<br>||
||||||||‘<br>PRE\|<br>||
||0.6\|||||||McC<br>Ft<br>SPE||
||||||||AUC<br>||
||OF||||Oo<br><aS\|||||



Figure 2. Performance comparison of different GNN layers. The dark blue nodes mark the highest point of each metric for different combinations of GNN layers. 

| 11 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

Table 5. Performance evaluation of different negative sampling strategies on RPImerged dataset, with results shown as mean ± standard deviation across five-fold cross-validation 

|**Negative sampling strategies**|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|
|---|---|---|---|---|---|---|---|
|Without negative sampling|0.903±0.058|0.898±0.080|0.968±0.008|0.918±0.020|0.931±0.044|0.782±0.112|0.966±0.020|
|File|0.857±0.034|0.929±0.042|0.813±0.028|0.786±0.032|0.867±0.033|0.723±0.070|0.910±0.043|
|Hard|0.922±0.037|0.930±0.065|0.914±0.014|0.913±0.010|0.921±0.039|0.845±0.074|0.967±0.024|
|Mixed|0.895±0.027|0.934±0.053|0.867±0.009|0.857±0.006|0.899±0.029|0.794±0.057|0.944±0.028|
|Random|0.924±0.036|0.934±0.063|0.915±0.017|0.914±0.013|0.924±0.039|0.849±0.073|0.969±0.022|



Table 6. Performance evaluation of supervised and self-supervised regimes on RPImerged dataset, with results shown as mean ± standard deviation across five-fold cross-validation 

|**Regime**|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|**Training time (s)**|
|---|---|---|---|---|---|---|---|---|
|Supervised|0.768±0.016|0.670±0.049|0.836±0.039|0.866±0.045|0.742±0.023|0.549±0.031|0.829±0.050|37.0|
|Self-Supervised|0.924±0.036|0.934±0.063|0.915±0.017|0.914±0.013|0.924±0.039|0.849±0.073|0.969±0.022|38.8|



to 0.924 and in MCC from 0.782 to 0.849 under random sampling highlights its additional benefits in enhancing model calibration and reducing false positives. These enhancements suggest that random sampling enables a more effective trade-off between PRE and RE, thereby contributing to improved overall robustness. In contrast, file-based and mixed sampling strategies produced relatively weaker performance across most metrics. 

These results demonstrate that while the model can achieve competitive performance without negative sampling, the choice of random sampling strategy still has a measurable impact on overall model performance. 

#### _Comparison of supervised and self-supervised regimes_ 

To highlight the advantages of our self-supervised learning framework, we conducted an experiment comparing it against a conventional supervised training regime on the RPImerged dataset. In the supervised setting, the model is trained using fixed positive and negative edge labels. In contrast, the self-supervised variant introduces two key components: negative sampling, which resamples negative pairs at each training epoch, and a masking strategy that encourages the model to learn from partially observed graph structures. 

As shown in Table 6, the self-supervised approach consistently outperforms the supervised baseline across all evaluation metrics. For example, ACC improves from 0.768 to 0.924, MCC increases from 0.549 to 0.849, and AUC rises from 0.829 to 0.969. These results indicate that incorporating masking and dynamic sampling strategies helps the model leverage richer structural information and a more varied set of training examples. 

It is worth noting that the improved performance of the selfsupervised model does not lead to a significant increase in training cost. The average training time over 100 epochs increases slightly, from 37.0 seconds in the supervised setting to 38.8 seconds in the self-supervised setting, suggesting that the proposed approach remains computationally feasible. 

Overall, these results demonstrate the effectiveness of the selfsupervised learning framework and indicate its potential as a reliable alternative to conventional supervised training for RPI prediction. 

### **Model interpretability analysis** 

To enhance the interpretability of our model and to better understand the contribution of different input features to prediction outcomes, we employed SHAP (Shapley Additive Explanations) to 

quantify the contribution of each input feature to the model’s predictions for specific RPI pairs [63]. SHAP values were computed using the Captum interpretability library [64], which supports gradient-based attribution tailored for deep learning architectures. To demonstrate the interpretability of our framework, we selected two representative RNA-protein pairs for detailed case analysis: Protein 1S1I-W with RNA 1S1I-3 (Supplementary Fig. S4) and Protein 1FFK-P with RNA 1FFK-0 (Supplementary Fig. S5). 

For each case, input features were grouped into biologically meaningful categories. RNA descriptors included NAC, Kmer, DPCP, PseDNC, PCPseDNC, and CKSNAP, while protein descriptors consisted of AAC, PAAC, CKSAAGP, QSOrder, DDE, and pretrained ESM-2 embeddings. We then aggregated SHAP values within each group to evaluate their overall contribution to the prediction. To further examine the internal decision process, we ranked all features by their SHAP values and visualized the top 20 most influential ones for each pair. 

In the case of Supplementary Fig. S4, the NAC feature group showed the highest average SHAP value among RNA descriptors. Within this group, the feature NAC_A exhibited the strongest individual contribution, indicating its relevance to the model’s decision. On the protein side, while the group-level importance scores across categories such as DDE, ESM-2, and QSOrder were relatively close, individual features from DDE and ESM-2 were ranked highest in terms of SHAP values, suggesting their significant influence on the final output. Similarly, in Supplementary Fig. S5, the NAC group remained the most dominant contributor among RNA features, again with NAC_A ranking highest. For proteins, the most influential individual features were mainly derived from ESM embeddings and DDE descriptors, including ESM_123, ESM_48, and DDE227. These findings indicate that while group-level contributions may be balanced, certain individual descriptors carry greater weight in driving model predictions. 

To further explore why models based on specific feature combinations, such as DPCP+ACC or PseDNC+AAC, perform poorly as shown in Table 4, we analyzed representative RNA-protein pairs that were misclassified under these combinations but correctly predicted when using the full set of features. Specifically, Protein 1HR0-I with RNA 1HR0-A failed to be predicted accurately when using only DPCP+ACC features, and Protein 3PYU-I with RNA 3PYU-A was misclassified under PseDNC+AAC (Supplementary Figs S6 and S7). However, when all features were combined, both predictions became correct, suggesting that additional feature types provided necessary complementary information. 

12 | Guan _et al._ 

SHAP analysis was performed on these two examples to understand the key factors behind the improved predictions. For both protein nodes, the ACC feature group had very low importance. In contrast, DDE and ESM-2 contributed substantially to the final predictions. For RNAs, NAC had the highest contribution, while the other five RNA feature types had relatively low importance. 

A closer inspection of the top 20 individual features by SHAP value further supported these observations. For the protein nodes, most of the top contributors originated from DDE and ESM-2, with additional features from CKSAAGP and QSOrder. On the RNA side, while NAC remained dominant, a few individual features from the DPCP group also exhibited high contributions. This indicates that although certain feature groups may show low overall importance, specific descriptors within them can still play a critical role when combined with other features. 

Together, these analyses demonstrate that models using only DPCP+ACC or PseDNC+AAC are often not able to capture the complexity of RPIs. In contrast, combining diverse and complementary feature groups allows the model to learn richer and more complete feature representations, which leads to more accurate predictions. These results show the importance of using multiview feature fusion and help explain how different types of features affect model performance at both the overall and individual levels. 

### **Analysis of Graph-encoded features for RNA–protein interactions** 

To analyze the graph-encoded features for RPIs, we extracted RPI features from five datasets (RPI488, RPI369, RPI2241, RPI1807, and RPI1446) using our model and visualized these high-dimensional features using t-distributed Stochastic Neighbor Embedding (tSNE) for dimensionality reduction [65]. The t-SNE visualizations reveal the distribution patterns of positive samples (RNA-protein pairs with interactions) and negative samples (RNA-protein pairs without interactions) in a two-dimensional space. As observed across all five datasets on Fig. 3, positive and negative samples are generally well-separated with only minimal overlap, indicating that the model has effectively learned the interaction features of these datasets. In the RPI488 (Fig. 3A) and RPI369 (Fig. 3B) datasets, although good separation between positive and negative samples is achieved, the sample points are relatively dispersed, which may be attributed to the smaller dataset size. In contrast, for the RPI2241 (Fig. 3C), RPI1807 (Fig. 3D), and RPI1446 (Fig. 3E) datasets, where the number of RNA-protein pairs is larger, most samples show a more clustered pattern within the same class. This suggests that our model effectively captures RPI features and can learn more robust feature representations, particularly in larger datasets. In addition, the visualization of the RPImerged dataset (Fig. 3F), which integrates all five datasets, also demonstrates a clear separation between interaction and non-interaction pairs. This highlights the model’s ability to generalize its feature encoding across heterogeneous data sources and reinforces its robustness when applied to more diverse RPI scenarios. Although t-SNE does not directly evaluate the model’s classification performance, it provides an intuitive understanding of the model’s capability to learn discriminative feature embeddings, helping to identify performance differences across various datasets and guiding future model optimization efforts. 

### **Comparison with other existing RPI prediction methods** 

To ensure a fair and rigorous comparison, we selected several state-of-the-art and widely adopted RPI prediction methods that could be consistently reproduced across all datasets. These 

include RPI-CapsuleGAN [19], LPI-SKMSC [20], RPI-MDLStack [18], LPI-CNNCP [15], RPISeq-RF [14], and RPISeq-SVM [14]. All baseline models were carefully reproduced based on the authors’ publicly available source code and detailed instructions in their original publications. 

As shown in Table 7 and Supplementary Table S8, we observe that our proposed model (Graph-RPI) outperforms all competing methods on the RPI369, RPI1446, RPI1807, RPI2241, and RPImerged datasets. The performance advantages of Graph-RPI are especially pronounced on the larger and more complex datasets (RPI1446, RPI1807, RPI2241, and RPImerged), highlighting its robustness and scalability in modeling realistic and large-scale RPI networks. 

On the RPI1446 dataset, Graph-RPI achieves an average ACC of 0.910, outperforming the baseline methods by 4 to 29 percentage points. The MCC is 0.821, which is 7 to 67 percentage points higher than those of the competing approaches. On the RPI1807 dataset, our model achieves an ACC of 0.979 and an MCC of 0.958, significantly exceeding all baselines, whose MCC values range only from 0.202 to 0.475. Similarly, on RPI2241, Graph-RPI attains an ACC of 0.950 and an MCC of 0.903, outperforming all competing methods by at least 10 percentage points in MCC. On the largest dataset, RPImerged, our model achieves an ACC of 0.924 and an MCC of 0.849, while the best-performing baseline (RPI-CapsuleGAN) only reaches 0.858 in ACC and 0.706 in MCC. Baseline methods such as RPISeqRF and RPISeqSVM exhibit a substantial performance drop on the RPImerged dataset, which can be attributed to two main factors. First, the merged dataset introduces considerable distributional heterogeneity across species, RNA categories, and sequence lengths, making it challenging for models that rely on fixed handcrafted features to generalize. Second, baseline methods typically predict each RNA–protein pair independently, overlooking the many-to-many interaction structure inherent in biological systems. In contrast, Graph-RPI models the entire dataset as a heterogeneous graph and leverages message passing to capture global interaction patterns, resulting in improved robustness and generalization. 

In addition, although our performance on the RPI488 dataset was slightly inferior to RPISeqSVM, this can be attributed to the smaller number of RNA nodes in this dataset. Our method, which is based on graph networks, contrasts with previous methods that constructed balanced datasets to achieve better and more balanced results on smaller datasets. However, our approach does not require the construction of negative samples, thus aligning more closely with the real-world structure of RPI data. 

To rigorously evaluate the statistical significance of the observed performance gains, we conducted paired t-tests between Graph-RPI and each baseline method across all datasets and evaluation metrics [66, 67]. For each dataset, we performed pairwise comparisons using results from five independent crossvalidation folds, covering standard metrics such as ACC, F1, AUC, and MCC. 

The results of the statistical analysis, as summarized in Supplementary Table S9, reveal that Graph-RPI achieves statistically significant improvements, particularly in MCC, AUC, and F1, on several datasets. For instance, on RPI2241, GraphRPI significantly outperforms all baseline methods on nearly all metrics. Similar trends are observed on RPI1807, RPI1446, and RPImerged, where p-values are below 0.01 for multiple critical evaluation metrics, indicating that the improvements are not due to random variation. 

To control for the risk of Type I errors resulting from multiple hypothesis testing, we applied the Benjamini–Hochberg correction to control the False Discovery Rate (FDR) [68, 69]. After computing raw p-values from the paired t-tests, we adjusted them using the 

| 13 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 



<!-- Start of picture text -->
(A)RPI488\| (B)RPI369\| (C)RPI2241]<br>Interaction a] Interaction] Interaction<br>No Interaction No Interactionj No Interaction\|<br>‘S<br>.¢ MY<br>*<br>_<br>S<br>\|<br>(D)RPI1807 (E)RPI1446 (F)RPlmerged<br>InteractionJ Interaction\| InteractionJ<br>No Interaction\| NoInteraction\| No Interaction<br>oe)<br>rf<br>oy<br>a]<br><!-- End of picture text -->

Figure 3. Visualization of encoded features for RPI prediction across six datasets: (A) RPI488, (B) RPI369, (C) RPI2241, (D) RPI1807, (E) RPI1446, and (F) RPImerged. 

Benjamini–Hochberg procedure with a significance threshold of _α_ = 0.05. This correction ensures that the expected proportion of false positives among all rejected hypotheses remains acceptably low, thereby enhancing the reliability of our statistical conclusions. 

Even after applying FDR correction, Graph-RPI retains advantages over the baseline methods across many datasets. In particular, on RPI2241, RPI1807, RPI1446, and RPImerged, a substantial number of evaluation metrics show adjusted p-values below the corrected significance threshold, confirming that the performance gains of our model are both consistent and statistically valid. 

### **Performance evaluation on independent test sets** 

To verify the generalization capability of our model on multispecies datasets and to assess its ability to explore and predict unknown RPIs, we used the RPI prediction model trained on RPI1807 to predict 10 093 RPI pairs from six independent test datasets. The prediction results were compared with those of RPICapsuleGAN [19], LPI-SKMSC [20], RPI-MDLStack [18], LPI-CNNCP [15], RPISeq-RF [14], and RPISeq-SVM [14], as detailed in Fig. 4 and Supplementary Table S10. 

Our method achieved the highest performance on all datasets. Notably, our model accurately predicted all RPIs in RPI_M and RPI_C, and achieved an ACC of 0.990 on RPI_H. Across the total 10 093 RPI pairs in the combined datasets, our method successfully predicted 9986 pairs, yielding an overall ACC of 

0.989. This represents an improvement of 0.011 to 0.499 over other methods. 

To further evaluate the model’s generalization ability on more challenging cases, we computed sequence similarity matrices between each independent dataset and the RPI1807 dataset, and selected the 20% of RNA-protein pairs with the lowest similarity scores for evaluation. These results are summarized in Supplementary Table S11. The model continued to perform well on these difficult subsets, correctly predicting 1413 out of 1463 pairs in RPI_H, 105 out of 134 in RPI_S, 34 out of 36 in RPI_E, and 12 out of 13 in RPI_D. In both RPI_M and RPI_C, all selected pairs were correctly predicted. 

The prediction results on the multi-species datasets demonstrate that our method possesses strong generalization ability and stable predictive performance, which contributes to advancing research in RPI prediction. 

### **Prediction of high probability RNA–protein interaction networks** 

Predicting RPI networks is crucial for understanding the intricate relationships between RNA and proteins, their structural and functional roles, and the underlying mechanisms of their interactions in various biological processes. In this study, we employed our model to predict the top 30 RPI pairs with the highest interaction probabilities across six species datasets. Each dataset represents RNAs and proteins as nodes and their interactions as edges in the RPI network (Supplementary Fig. S8). In these network diagrams, green nodes represent RNAs, blue nodes 

14 | Guan _et al._ 

Table 7. Performance comparison with other RPI prediction methods on multiple RPI datasets, with results shown as mean ± standard deviation across five-fold cross-validation 

|**Dataset**|**Method**|**ACC**|**RE**|**PRE**|**SPE**|**F1**|**MCC**|**AUC**|
|---|---|---|---|---|---|---|---|---|
||RPISeqRF|0.787±0.203|0.595±0.412|0.957±0.046|0.979±0.018|0.638±0.380|0.621±0.354|0.874±0.158|
||RPISeqSVM|0.888±0.169|0.820±0.319|0.910±0.111|0.955±0.030|0.836±0.273|0.785±0.319|0.936±0.102|
||LPI-CNNCP|0.761±0.133|0.584±0.276|0.889±0.095|0.938±0.032|0.671±0.225|0.562±0.241|0.883±0.144|
|RPI488|RPI-MDLStack|0.787±0.195|0.591±0.392|0.965±0.038|0.984±0.016|0.648±0.354|0.625±0.338|0.873±0.113|
||LPI-SKMSC|0.861±0.180|0.784±0.311|0.873±0.188|0.935±0.061|0.813±0.272|0.720±0.370|0.897±0.178|
||RPI-CapsuleGAN|0.826±0.155|0.697±0.306|0.946±0.062|0.974±0.025|0.765±0.261|0.689±0.282|0.861±0.131|
||Graph-RPI|0.880±0.052|0.864±0.095|0.897±0.052|0.897±0.066|0.877±0.055|0.767±0.100|0.923±0.046|
||RPISeqRF|0.784±0.173|0.620±0.346|0.907±0.086|0.948±0.036|0.677±0.311|0.601±0.310|0.897±0.129|
||RPISeqSVM|0.806±0.127|0.719±0.244|0.846±0.092|0.894±0.020|0.761±0.206|0.622±0.240|0.898±0.065|
||LPI-CNNCP|0.774±0.036|0.895±0.133|0.727±0.038|0.660±0.078|0.794±0.046|0.582±0.096|0.807±0.032|
|RPI369|RPI-MDLStack|0.663±0.190|0.656±0.242|0.937±0.021|0.606±0.266|0.744±0.194|0.191±0.125|0.694±0.078|
||LPI-SKMSC|0.741±0.043|0.803±0.064|0.908±0.090|0.305±0.262|0.846±0.027|0.088±0.153|0.381±0.214|
||RPI-CapsuleGAN|0.634±0.200|0.653±0.239|0.882±0.064|0.385±0.195|0.731±0.197|0.027±0.129|0.512±0.144|
||Graph-RPI|0.811±0.060|0.745±0.098|0.859±0.061|0.876±0.058|0.795±0.069|0.630±0.116|0.772±0.071|
||RPISeqRF|0.691±0.193|0.429±0.391|0.767±0.251|0.953±0.031|0.480±0.378|0.406±0.379|0.770±0.221|
||RPISeqSVM|0.746±0.219|0.623±0.382|0.715±0.294|0.869±0.086|0.645±0.354|0.483±0.460|0.786±0.232|
||LPI-CNNCP|0.703±0.014|0.639±0.032|0.734±0.025|0.767±0.033|0.682±0.015|0.410±0.027|0.774±0.009|
|RPI2241|RPI-MDLStack|0.854±0.015|0.831±0.010|0.871±0.025|0.876±0.025|0.850±0.014|0.708±0.030|0.877±0.010|
||LPI-SKMSC|0.622±0.186|0.507±0.352|0.672±0.276|0.548±0.287|0.555±0.334|0.029±0.207|0.527±0.240|
||RPI-CapsuleGAN|0.847±0.005|0.850±0.012|0.845±0.002|0.844±0.005|0.848±0.006|0.695±0.009|0.871±0.009|
||Graph-RPI|0.950±0.027|0.927±0.050|0.973±0.008|0.974±0.008|0.949±0.029|0.903±0.051|0.987±0.009|
||RPISeqRF|0.662±0.182|0.386±0.359|0.805±0.228|0.960±0.030|0.453±0.347|0.381±0.340|0.820±0.210|
||RPISeqSVM|0.724±0.198|0.571±0.351|0.772±0.273|0.894±0.086|0.624±0.318|0.472±0.403|0.822±0.212|
||LPI-CNNCP|0.557±0.057|0.250±0.105|0.802±0.135|0.945±0.011|0.374±0.150|0.243±0.129|0.866±0.052|
|RPI1807|RPI-MDLStack|0.653±0.162|0.387±0.284|0.948±0.046|0.988±0.009|0.499±0.278|0.437±0.252|0.879±0.085|
||LPI-SKMSC|0.665±0.171|0.544±0.299|0.747±0.253|0.671±0.294|0.613±0.291|0.202±0.313|0.650±0.266|
||RPI-CapsuleGAN|0.677±0.168|0.438±0.298|0.943±0.035|0.980±0.005|0.548±0.267|0.475±0.264|0.698±0.149|
||Graph-RPI|0.979±0.007|0.986±0.009|0.973±0.009|0.972±0.009|0.979±0.007|0.958±0.014|0.986±0.008|
||RPISeqRF|0.636±0.173|0.334±0.341|0.749±0.259|0.954±0.029|0.402±0.334|0.318±0.337|0.778±0.211|
||RPISeqSVM|0.653±0.227|0.472±0.371|0.654±0.340|0.847±0.121|0.521±0.351|0.313±0.481|0.702±0.308|
||LPI-CNNCP|0.627±0.050|0.601±0.272|0.684±0.141|0.655±0.205|0.560±0.209|0.287±0.062|0.720±0.017|
|RPI1446|RPI-MDLStack|0.874±0.006|0.889±0.015|0.855±0.015|0.860±0.019|0.871±0.005|0.749±0.011|0.912±0.004|
||LPI-SKMSC|0.624±0.166|0.454±0.303|0.688±0.246|0.708±0.263|0.528±0.293|0.153±0.286|0.623±0.247|
||RPI-CapsuleGAN|0.842±0.009|0.858±0.017|0.822±0.008|0.828±0.010|0.840±0.010|0.685±0.018|0.866±0.012|
||Graph-RPI|0.910±0.029|0.884±0.040|0.931±0.023|0.935±0.022|0.907±0.030|0.821±0.057|0.969±0.011|
||RPISeqRF|0.554±0.008|0.039±0.022|0.236±0.120|0.920±0.012|0.066±0.037|-0.088±0.048|0.300±0.060|
||RPISeqSVM|0.505±0.016|0.042±0.033|0.137±0.093|0.833±0.028|0.064±0.049|-0.196±0.054|0.229±0.064|
||LPI-CNNCP|0.693±0.016|0.531±0.079|0.676±0.046|0.809±0.066|0.589±0.034|0.361±0.027|0.766±0.007|
|RPImerged|RPI-MDLStack|0.628±0.049|0.192±0.121|0.652±0.102|0.940±0.005|0.287±0.145|0.187±0.131|0.810±0.017|
||LPI-SKMSC|0.518±0.012|0.146±0.012|0.339±0.032|0.792±0.009|0.203±0.016|-0.080±0.023|0.425±0.059|
||RPI-CapsuleGAN|0.858±0.003|0.797±0.009|0.853±0.011|0.901±0.010|0.824±0.003|0.706±0.006|0.864±0.009|
||Graph-RPI|0.924±0.036|0.934±0.063|0.915±0.017|0.914±0.013|0.924±0.039|0.849±0.073|0.969±0.022|



represent proteins, and edges illustrate predicted interactions, providing valuable insights for interpretability analysis. 

As depicted in Supplementary Fig. S8A, a total of 30 highprobability RPI pairs were predicted in the _H. sapiens_ dataset. These include interactions between proteins such as P42574 (Caspase-3) and P61964 (a WD40 repeat protein), and RNAs like AC006369.2 and TRIM36-IT1. Each predicted pair suggests a potential regulatory mechanism with implications for cellular function and disease. 

To further evaluate the biological plausibility of these predictions, we selected the aforementioned RNA-protein pairs and employed AlphaFold3 to predict their structural conformations [70]. Additionally, the experimentally validated interaction between CDKN2B-AS1 and Q15022 was used as a positive control for comparison. As shown in Supplementary Fig. S9, the predicted complexes exhibited consistent interaction patterns, with specific amino acid residues forming hydrogen bonds with corresponding RNA nucleotides [71]. The binding interfaces of the predicted pairs 

closely resembled those of the validated complex, indicating both structural feasibility and potential biological relevance. 

Building upon these structural insights, we further explored the biological context of the predicted RNA-protein pairs. For instance, P42574 (Caspase-3) is a key cysteine protease involved in the execution phase of apoptosis and has been linked to immunological infertility [72]. In terms of RNA, AC006369.2 is one of 16 differentially expressed ferroptosis-related lncRNAs associated with skin squamous cell carcinoma. These lncRNAs play a role in regulating the tumor immune microenvironment, potentially influencing gene expression related to ferroptosis and immune checkpoints [73]. The interaction between P42574 and AC006369.2 might suggest a crosstalk between apoptosis and ferroptosis pathways within certain tumor microenvironments, indicating that AC006369.2 may modulate Caspase-3 expression or activity through its RNA-mediated mechanisms. 

Similarly, P61964, a member of the WD40 repeat protein family, is known to act as a scaffold for protein-protein interactions and 

| 15 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 



<!-- Start of picture text -->
Dataset: RPI_H\| Dataset: RPI_M\| Dataset: RPI_S]<br>8000 4<br>2000, 641) 647] 647<br>7245 7236\| 7147 1800, 18471847 1843 1847] 600 \| 603, 589<br>7000} 6812\| 1709<br>543]<br>6446 1600\|<br>500f<br>6000 \|<br>1400]<br>&<br>‘5256\| 1200- 1171]<br>5000<br>1000<br>907]<br>4000 f 800 { 2 3004 293<br>3600<br>600 {<br>3000 200)<br>z<br>Dataset: RPI_EJ Dataset: RPI_D] Dataset: RPI_C]<br>200<br>704<br>4] 10<br>63 62<br>181) 607 58 59)<br>180{<br>53]<br>169] 51]<br>504<br>160\|<br>153] 151] 404 6<br>1404 132] 143] 30: 30 5 4]<br>g 204<br>1204 2<br>104<br>107<br>100]<br>H<br>Total RPI pairs<br>(98.9%) (97.8%) (97.5%)\|<br>9986 (91.1%)\| 9866 9844\| (93.2%)<br>10000\| 9199 9403\|<br>8000 (71.1%)\|<br>7178]<br>i<br>a 6000 (49.0%)\|<br>4941)<br>4000\|<br>Graph-RPI RPI-CapsuleGAN\| LPI-SKMSC\| RPI-MDLStack\| LPI-CNNCP] RPISeq-SVM RPISeq-RF\|<br><!-- End of picture text -->

Figure 4. Performance comparison with other existing RPI prediction methods on six species datasets. 

is implicated in various cellular pathways, including transcription regulation and the ubiquitin-proteasome system. WD40 repeat proteins are associated with several diseases, such as cancer, neurological disorders, metabolic syndromes, viral infections, and = inflammatory conditions [74]. In parallel, TRIM36-IT1, a long noncoding RNA related to clear cell renal cell carcinoma (ccRCC), has been identified as an independent prognostic factor influencing overall survival in ccRCC patients. TRIM36-IT1 is involved in the competing endogenous RNA network, modulating specific mRNA expression by binding to microRNAs (e.g. hsa-mir-21) [75]. This regulation might influence protein interactions indirectly, thereby contributing to cancer progression. 

These interactions, predicted by our model, illustrate the complex interplay between RNAs and proteins in _H. sapiens_ , highlighting how specific RNA molecules and proteins may influence each other’s function and regulation. By mapping these predicted interactions, we can better understand the underlying molecular mechanisms that govern cellular processes. Extending this analysis to RPI networks in other species datasets (Supplementary Fig. S8B–F) could provide further valuable insights into conserved or species-specific cellular regulatory mechanisms. Such comparative analyses could reveal fundamental principles of RPIs, help identify potential targets for therapeutic interventions, and improve our understanding of the evolutionary dynamics of these 

16 | Guan _et al._ 

interactions across different organisms. Additionally, the results of these high-probability RPI predictions are provided in Tables S12-S17 for further reference and detailed analysis. 

in future work, we believe this framework holds promise for advancing RPI-related research and its applications in disease understanding and drug development. 

## Conclusion 

In this study, we addressed key challenges in RPI prediction by developing a self-supervised graph-based framework that effectively integrates RNA and protein sequence information into a unified interaction graph. Traditional RPI prediction methods often suffer from limitations such as insufficient feature representation, high model complexity, and poor generalization. Our proposed model leverages GNNs to learn from the topological structure of RPI networks, where each node is iteratively updated based on its neighboring nodes, rather than treating each interaction in isolation. This design enables the model to capture global patterns in the interaction network, leading to improved predictive performance across multiple benchmark datasets. 

We evaluated our model on five widely used benchmark datasets (RPI488, RPI369, RPI2241, RPI1807, and RPI1446), which are commonly adopted in the field due to their reliable annotations and well-established standards for model comparison. To further assess the generalization capability of our method, we conducted prediction experiments on independent test datasets derived from six species, including _H. sapiens_ , _M. musculus_ , _S. cerevisiae_ , _E. coli_ , _D. melanogaster_ , and _C. elegans_ . These datasets were collected from public resources such as NPInter and contain experimentally validated interactions that were not involved in training. The results demonstrate strong generalization performance across species and unseen data. 

We also acknowledge several limitations in the current study. First, although the independent test datasets contain experimentally supported interactions, they are still curated from public databases and may not fully capture the complexity of real biological or clinical settings. Due to practical constraints related to experimental conditions and time, we have not conducted large-scale biological or clinical validation. As part of future work, we plan to collaborate with experimental or clinical teams to collect newly generated RPI data under more realistic biological conditions, which will be used to further evaluate and improve our framework. 

Second, our model does not explicitly incorporate contextspecific information such as cell type or tissue origin [76, 77], as the current benchmark datasets lack such annotations. However, we recognize that RPIs are often highly dependent on biological context. In future research, we aim to integrate cell- or tissuespecific datasets generated through high-throughput technologies such as CLIP-seq, RNA-seq, and proteomic profiling [78, 79]. These data will allow us to build stratified RPI networks and evaluate model performance in different biological environments. 

Finally, although the current study does not directly link predicted RPIs to clinical outcomes, we recognize the importance of exploring the clinical relevance of RPI networks. As a future extension, we plan to incorporate publicly available resources such as TCGA [80], ClinVar [81], or disease-specific expression profiles to investigate whether predicted interactions correlate with disease states or other clinically meaningful features. This will support the translational potential of our framework in areas such as biomarker discovery and therapeutic target identification. 

In summary, our method provides an effective and scalable solution for RPI prediction, with demonstrated performance on both benchmark and independent datasets. By extending the model to integrate biological context and clinically relevant data 

##### **Key Points** 

- In this study, We proposed a novel RNA–protein interaction (RPI) prediction framework using graph autoencoder, which integrates RNA and protein sequence features into a unified graph structure. 

- The model employs self-supervised learning strategies, including masking mechanism and negative sampling, to enhance feature representation and improve prediction robustness without relying on explicit negative sample construction. 

- Extensive experiments demonstrate that our model achieves strong predictive performance across multiple datasets, achieving results comparable to or exceeding existing RPI prediction methods 

- Our approach shows strong generalization capability across multi-species datasets, suggesting its potential for broad applications in disease-related RPI research and therapeutic target identification. 

## Acknowledgments 

The authors sincerely appreciate the Kobilka Institute of Innovative Drug Discovery, The Chinese University of Hong Kong (Shenzhen), the Faculty of Dentistry, the University of Hong Kong, and the “Center for intelligent Drug Systems and Smart Biodevices” from The Featured Areas Research Center Program within the framework of the Higher Education Sprout Project by the Ministry of Education in Taiwan. 

## Author contributions 

J.H.G. and Y.-C.C. presented the idea. J.H.G. implemented the framework. J.H.G., P.L.X. and Z.H.Z. collected the data. J.H.G. and D.M. analyzed the predictive results. Y.-C.C., J.W.W., T.-Y.L., and L.T.Y, provided expert guidance for the experiments. Y.-C.C., J.W.W., and T.-Y.L. supervised the research project. 

Competing interests: The authors declare no competing interests. 

## Funding 

This work was supported by Shenzhen Science and Technology Innovation Commission (JCYJ20230807114206014), Guangdong Province Basic and Applied Basic Research Fund (2025A1515011753), and the Kobilka Institute of Innovative Drug Discovery, The Chinese University of Hong Kong, Shenzhen, China. This work was financially supported by collaborative research fund of Hong Kong RGC (C7015-23G), seed funding for collaborative research (2207101590) from the University of Hong Kong to J Wang. This work was also financially supported by the Center for Intelligent Drug Systems and Smart Biodevices (IDS2B) from The Featured Areas Research Center Program within the framework of the Higher Education Sprout Project and the Yushan Young Fellow Program (114C51N039) by the Ministry of Education (MOE), National Science and Technology Council (NSTC 113-2221-E-A49160-MY3, 113-2321-BA49- 025 and 114-2634-F-039-001), and The 

| 17 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

National Health Research Institutes (NHRI-EX114-11320BI) in Taiwan. 

## Data availability 

The full experimental code and datasets are available at https:// github.com/GGCL7/Graph-RPI-Experiments. The standalone software for single or multiple RNA-protein pair prediction is available at https://github.com/GGCL7/Graph-RPI. 

## References 

1. Re A, Joshi T, Kulberkyte E. _et al._ RNA–protein interactions: an overview. _Methods Mol Biol_ 2014; **1097** :491–521. https://doi. org/10.1007/978-1-62703-709-9_23 

2. Jankowsky E, Harris ME. Specificity and nonspecificity in RNA– protein interactions. _Nat Rev Mol Cell Biol_ 2015; **16** :533–44. https:// doi.org/10.1038/nrm4032 

3. Ramanathan M, Porter DF, Khavari PA. Methods to study RNA– protein interactions. _Nat Methods_ 2019; **16** :225–34. https://doi. org/10.1038/s41592-019-0330-1 

4. Khalil AM, Rinn JL. RNA–protein interactions in human health and disease. In: _Seminars in Cell & Developmental Biology_ , Vol. **22** , pp. 359–65. Amsterdam: Elsevier, 2011. 

5. Hermann T. Strategies for the design of drugs targeting RNA and RNA–protein complexes. _Angew Chem Int Ed_ 2000; **39** :1890–904. https://doi.org/10.1002/1521-3773(20000602)39:11 _<_ 1890::AIDANIE1890 _>_ 3.0.CO;2-D 

6. Das A, Sinha T, Shyamal S. _et al._ Emerging role of circular RNA–protein interactions. _Non-coding RNA_ 2021; **7** :48. https://doi. org/10.3390/ncrna7030048 

7. Puton T, Kozlowski L, Tuszynska I. _et al._ Computational methods for prediction of protein–RNA interactions. _J Struct Biol_ 2012; **179** : 261–8. https://doi.org/10.1016/j.jsb.2011.10.001 

8. Jones S. Protein–RNA interactions: structural biology and computational modeling techniques. _Biophysical reviews_ 2016; **8** : 359–67. https://doi.org/10.1007/s12551-016-0223-9 

9. Sun L-Z, Jiang Y, Zhou Y. _et al._ RLDOCK: a new method for predicting RNA–ligand interactions. _J Chem Theory Comput_ 2020; **16** : 7173–83. https://doi.org/10.1021/acs.jctc.0c00798 

10. Yang S, Wang J, Ng RT. Inferring RNA sequence preferences for poorly studied RNA-binding proteins based on co-evolution. _BMC Bioinf_ 2018; **19** :1–12. https://doi.org/10.1186/ s12859-018-2091-8 

11. Zhao Z, Gui J, Yao A. _et al._ Improved prediction model of protein and peptide toxicity by integrating channel attention into a convolutional neural network and gated recurrent units. _ACS Omega_ 2022; **7** :40569–77. https://doi.org/10.1021/acsomega.2c05881 

12. Le NQK. Predicting emerging drug interactions using gnns. _Nat Comput Sci_ 2023; **3** :1007–8. https://doi.org/10.1038/ s43588-023-00555-7 

13. Le NQK, Li W, Cao Y. Sequence-based prediction model of protein crystallization propensity using machine learning and twolevel feature selection. _Brief Bioinform_ 2023; **24** :bbad319. https:// doi.org/10.1093/bib/bbad319 

14. Muppirala UK, Honavar VG, Dobbs D. Predicting RNA-protein interactions using only sequence information. _BMC Bioinf_ 2011; **12** :1–11. 

15. Zhang S-W, Zhang X-X, Fan X-N. _et al._ LPI-CNNCP: Prediction of lncRNA-protein interactions by using convolutional neural network with the copy-padding trick. _Anal Biochem_ 2020; **601** :113767. https://doi.org/10.1016/j.ab.2020.113767 

16. Li Y, Sun H, Feng S. _et al._ Capsule-LPI: a lncRNA–protein interaction predicting tool based on a capsule network. _BMC Bioinf_ 2021; **22** :246. https://doi.org/10.1186/s12859-021-04171-y 

17. Huang X, Shi Y, Yan J. _et al._ LPI-CSFFR: combining serial fusion with feature reuse for predicting lncRNA-protein interactions. _Comput Biol Chem_ 2022; **99** :107718. https://doi.org/10.1016/ j.compbiolchem.2022.107718 

18. Bin Y, Wang X, Zhang Y. _et al._ RPI-MDLSTACK: predicting RNA– protein interactions through deep learning with stacking strategy and LASSO. _Appl Soft Comput_ 2022; **120** :108676. 

19. Wang Y, Wang X, Chen C. _et al._ RPI-CapsuleGAN: predicting RNA-protein interactions through an interpretable generative adversarial capsule network. _Pattern Recognit_ 2023; **141** :109626. https://doi.org/10.1016/j.patcog.2023.109626 

20. Sun D-Z, Sun Z-L, Liu M. _et al._ LPI-SKMSC: predicting lncRNA– protein interactions with segmented k-mer frequencies and multi-space clustering. _Interdiscip Sci: Comput Life Sci_ 2024; **16** : 378–91. https://doi.org/10.1007/s12539-023-00598-4 

21. Tang X, Machimura T, Li J. _et al._ A novel optimized repeatedly random undersampling for selecting negative samples: a case study in an svm-based forest fire susceptibility assessment. _J Environ Manage_ 2020; **271** :111014. https://doi.org/10.1016/ j.jenvman.2020.111014 

22. Zhang X, Yates A, Lin J. A little bit is worse than none: ranking with limited training data. In: _Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing_ , pp. 107–12, 2020. 

23. Zonghan W, Pan S, Chen F. _et al._ A comprehensive survey on graph neural networks. _IEEE Trans Neural Networks Learn Syst_ 2020; **32** :4–24. https://doi.org/10.1109/TNNLS.2020.2978386 

24. Zhou J, Cui G, Shengding H. _et al._ Graph neural networks: A review of methods and applications. _AI Open_ 2020; **1** :57–81. https://doi.org/10.1016/j.aiopen.2021.01.001 

25. Hou Z, Liu X, Cen Y. _et al._ Graphmae: Self-supervised masked graph autoencoders. In: _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . New York, NY, USA: Association for Computing Machinery, pp. 594–604, 2022. 

26. Lewis BA, Walia RR, Terribilini M. _et al._ PRIDB: a protein–RNA interface database. _Nucleic Acids Res_ 2010; **39** :D277–D282. https:// doi.org/D277 

27. Pan X, Fan Y-X, Yan J. _et al._ Ipminer: Hidden ncRNA-protein interaction sequential pattern mining with stacked autoencoder for accurate computational prediction. _BMC Genomics_ 2016; **17** : 1–14. 

28. Berman HM, Westbrook J, Feng Z. _et al._ The protein data bank. _Nucleic Acids Res_ 2000; **28** :235–42. https://doi.org/10.1093/ nar/28.1.235 

29. Suresh V, Liu L, Adjeroh D. _et al._ RPI-Pred: predicting ncRNAprotein interaction using sequence and structural information. _Nucleic Acids Res_ 2015; **43** :1370–9. https://doi.org/10.1093/ nar/gkv020 

30. Qiongshi L, Ren S, Ming L. _et al._ Computational prediction of associations between long non-coding RNAs and proteins. _BMC Genomics_ 2013; **14** :1–10. 

31. Corpet F. Multiple sequence alignment with hierarchical clustering. _Nucleic Acids Res_ 1988; **16** :10881–90. https://doi.org/10.1093/ nar/16.22.10881 

32. Ran X, Xi Y, Yonggang L. _et al._ Comprehensive survey on hierarchical clustering algorithms and the recent developments. _Artif Intell Rev_ 2023; **56** :8219–64. https://doi.org/10.1007/ s10462-022-10366-3 

33. Needleman SB, Wunsch CD. A general method applicable to the search for similarities in the amino acid sequence 

18 | Guan _et al._ 

   - of two proteins. _J Mol Biol_ 1970; **48** :443–53. https://doi. org/10.1016/0022-2836(70)90057-4 

34. Leelananda SP, Kloczkowski A, Jernigan RL. Fold-specific sequence scoring improves protein sequence matching. _BMC Bioinf_ 2016; **17** :1–14. https://doi.org/10.1186/s12859-016-1198-z 

35. Sepehr Ashrafzadeh G, Golding B, Ilie S. _et al._ Scoring alignments by embedding vector similarity. _Brief Bioinform_ 2024; **25** :bbae178. https://doi.org/10.1093/bib/bbae178 

36. Fan X-N, Zhang S-W. LPI-BLS: predicting lncRNA–protein interactions with a broad learning system-based stacked ensemble classifier. _Neurocomputing_ 2019; **370** :88–93. https://doi. org/10.1016/j.neucom.2019.08.084 

37. Hao Y, Wu W, Li H. _et al._ NPInter v3. 0: an upgraded database of noncoding RNA-associated interactions. _Database_ 2016; **2016** :baw057. https://doi.org/10.1093/database/baw057 

38. Frankish A, Diekhans M, Ferreira A-M. _et al._ Gencode reference annotation for the human and mouse genomes. _Nucleic Acids Res_ 2019; **47** :D766–73. https://doi.org/10.1093/nar/gky955 

39. UniProt Consortium. Uniprot: A worldwide hub of protein knowledge. _Nucleic Acids Res_ 2019; **47** :D506–15. https://doi. org/10.1093/nar/gky1049 

40. Dechao B, Kuntao Y, Sun S. _et al._ NONCODE v3. 0: integrative annotation of long noncoding RNAss. _Nucleic Acids Res_ 2012; **40** :D210–5. 

41. Hampson S, Kibler D, Baldi P. Distribution patterns of overrepresented k-mers in non-coding yeast DNA. _Bioinformatics_ 2002; **18** :513–28. https://doi.org/10.1093/bioinformatics/18.4.513 

42. Friedel M, Nikolajewa S, Sühnel J. _et al._ DiProDB: a database for dinucleotide properties. _Nucleic Acids Res_ 2009; **37** :D37–40. https://doi.org/10.1093/nar/gkn597 

43. Chen W, Feng P-M, Lin H. _et al._ iSS-PseDNC: identifying splicing sites using pseudo dinucleotide composition. _Biomed Res Int_ 2014; **2014** :623149. 

44. Chen Z, Liu X, Zhao P. _et al._ iFeatureOmega: an integrative platform for engineering, visualization and analysis of features from molecular sequences, structural and ligand data sets. _Nucleic Acids Res_ 2022; **50** :W434–47. https://doi.org/10.1093/nar/ gkac351 

45. Chen K, Kurgan LA, Ruan J. Prediction of flexible/rigid regions from protein sequences using k-spaced amino acid pairs. _BMC Struct Biol_ 2007; **7** :1–13. https://doi.org/10.1186/1472-6807-7-25 

46. Chou K-C. Prediction of protein cellular attributes using pseudoamino acid composition. _Proteins Struct Funct Bioinf_ 2001; **43** : 246–55. https://doi.org/10.1002/prot.1035 

47. Yao L, Guan J, Li W. _et al._ Identifying antitubercular peptides via deep forest architecture with effective feature representation. _Anal Chem_ 2024; **96** :1538–46. https://doi.org/10.1021/acs. analchem.3c04196 

48. Chou K-C. Prediction of protein subcellular locations by incorporating quasi-sequence-order effect. _Biochem Biophys Res Commun_ 2000; **278** :477–83. https://doi.org/10.1006/bbrc.2000.3815 

49. Guan J, Yao L, Xie P. _et al._ A two-stage computational framework for identifying antiviral peptides and their functional types based on contrastive learning and multi-feature fusion strategy. _Brief Bioinform_ 2024; **25** :bbae208. https://doi.org/10.1093/bib/ bbae208 

50. Ghulam A, Ali F, Sikander R. _et al._ ACP-2DCNN: deep learningbased model for improving prediction of anticancer peptides using two-dimensional convolutional neural network. _Chemom Intel Lab Syst_ 2022; **226** :104589. https://doi.org/10.1016/ j.chemolab.2022.104589 

51. Guan J, Yao L, Chung C-R. _et al._ Predicting anti-inflammatory peptides by ensemble machine learning and deep learning. 

_J Chem Inf Model_ 2023; **63** :7886–98. https://doi.org/10.1021/acs. jcim.3c01602 

52. Lin Z, Akin H, Rao R. _et al._ Evolutionary-scale prediction of atomic-level protein structure with a language model. _Science_ 2023; **379** :1123–30. https://doi.org/10.1126/science.ade2574 

53. Pan S, Hu R, Long G. _et al._ Adversarially regularized graph autoencoder for graph embedding arXiv preprint arXiv:1802.04407. 2018. 

54. Zhou Z, Zhuo L, Xiangzheng F. _et al._ Joint masking and self-supervised strategies for inferring small molecule-mirna associations. _Mol Ther Nucleic Acids_ 2024; **35** :102103. https://doi. org/10.1016/j.omtn.2023.102103 

55. Wang C, Pan S, Long G. _et al._ MGAE: arginalized graph autoencoder for graph clustering. In: _Proceedings of the 2017 ACM on Conference on Information and Knowledge Management_ . New York, NY, USA: Association for Computing Machinery, pp. 889–98, 2017. 

56. Velickovi´cˇ P, Cucurull G, Casanova A. _et al._ Graph attention networks arXiv preprint arXiv:1710.10903. 2017. 

57. Peng Y, Lin Y, Jing X-Y. _et al._ Enhanced graph isomorphism network for molecular admet properties prediction. _IEEE Access_ 2020; **8** :168344–60. https://doi.org/10.1109/ACCESS.2020. 3022850 

58. Li J, Wu R, Sun W. _et al._ What’s behind the mask: understanding masked graph modeling for graph autoencoders. In: _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . New York, NY, USA: Association for Computing Machinery (ACM), pp. 1268–79, 2023. 

59. Yang Z, Ding M, Zhou C. _et al._ Understanding negative sampling in graph representation learning. In: _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_ . New York, NY, USA: Association for Computing Machinery, pp. 1666–76, 2020. 

60. Van Rossum G, Drake Jr FL. _Python Tutorial_ , Vol. **620** . The Netherlands: Centrum voor Wiskunde en Informatica Amsterdam, 1995. 

61. Paszke A. Pytorch: an imperative style, high-performance deep learning library arXiv preprint arXiv:1912.01703. 2019. 

62. Chicco D, Tötsch N, Jurman G. The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation. _BioData Mining_ 2021; **14** :1–22. https://doi.org/10.1186/ s13040-021-00244-z 

63. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: _Advances in Neural Information Processing Systems_ 2017; **30** . 

64. Kokhlikyan N, Miglani V, Martin M. _et al._ Captum: a unified and generic model interpretability library for pytorch arXiv preprint arXiv:2009.07896. 2020. 

65. Van der Maaten L, Hinton G. Visualizing data using t-SNE. _J Mach Learn Res_ 2008; **9** :2579–2605. 

66. Lakens D. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and anovas. _Front Psychol_ 2013; **4** :863. 

67. Demsarˇ J. Statistical comparisons of classifiers over multiple data sets. _J Mach Learn Res_ 2006; **7** :1–30. 

68. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. _J R Stat Soc B Methodol_ 1995; **57** :289–300. https://doi.org/10.1111/ j.2517-6161.1995.tb02031.x 

69. Kwong KS, Holland B, Cheung SH. A modified benjamini– hochberg multiple comparisons procedure for controlling the false discovery rate. _J Stat Plann Inference_ 2002; **104** :351–62. https://doi.org/10.1016/S0378-3758(01)00252-X 

| 19 

Graph-RPI: predicting RNA–protein interactions via graph autoencoder and self-supervised learning strategies 

70. Abramson J, Adler J, Dunger J. _et al._ Accurate structure prediction of biomolecular interactions with AlphaFold 3. _Nature_ 2024; **630** : 493–500. https://doi.org/10.1038/s41586-024-07487-w 

71. Humphrey W, Dalke A, Schulten K. VMD: visual molecular dynamics. _J Mol Graph_ 1996; **14** :33–8. https://doi.org/10.1016/ 0263-7855(96)00018-5 

72. Bohring C, Krause E, Habermann B. _et al._ Isolation and identification of sperm membrane antigens recognized by antisperm antibodies, and their possible role in immunological infertility disease. _Mol Hum Reprod_ 2001; **7** :113–8. https://doi.org/10.1093/ molehr/7.2.113 

73. Guo S, Chen J, Yi X. _et al._ Identification and validation of ferroptosis-related lncRNA signature as a prognostic model for skin cutaneous melanoma. _Front Immunol_ 2022; **13** :985051. https://doi.org/10.3389/fimmu.2022.985051 

74. Song R, Wang Z-D, Schapira M. Disease association and druggability of wd40 repeat proteins. _J Proteome Res_ 2017; **16** :3766–73. https://doi.org/10.1021/acs.jproteome.7b00451 

75. Wang J, Zhang C, He W. _et al._ Construction and comprehensive analysis of dysregulated long non-coding rna-associated competing endogenous rna network in clear cell renal cell carcinoma. _J Cell Biochem_ 2019; **120** :2576–93. https://doi.org/10.1002/ jcb.27557 

76. Ramanathan M, Majzoub K, Rao DS. _et al._ RNA–protein interaction detection in living cells. _Nat Methods_ 2018; **15** :207–12. https:// doi.org/10.1038/nmeth.4601 

77. Huang A, Zheng H, Zhiye W. _et al._ Circular RNA-protein interactions: functions, mechanisms, and identification. _Theranostics_ 2020; **10** :3503–17. https://doi.org/10.7150/thno.42174 

78. Budak G, Srivastava R, Janga SC. Seten: a tool for systematic identification and comparison of processes, phenotypes, and diseases associated with RNA-binding proteins from condition-specific clip-seq profiles. _RNA_ 2017; **23** :836–46. https:// doi.org/10.1261/rna.059089.116 

79. Milek M, Wyler E, Landthaler M. Transcriptome-wide analysis of protein–RNA interactions using high-throughput sequencing. In: _Seminars in Cell & Developmental Biology_ , Vol. **23** , pp. 206–12. Amsterdam: Elsevier, 2012. 

80. Tomczak K, Czerwi ´nska P, Wiznerowicz M. Review the cancer genome atlas (TCGA): an immeasurable source of knowledge. _Contemporary Oncology/Współczesna Onkologia_ 2015; **2015** : 68–77. 

81. Landrum MJ, Lee JM, Benson M. _et al._ ClinVar: public archive of interpretations of clinically relevant variants. _Nucleic Acids Res_ 2016; **44** :D862–8. https://doi.org/10.1093/nar/ gkv1222 

© The Author(s) 2025. Published by Oxford University Press. This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License (https://creativecommons.org/licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact reprints@oup.com for reprints and translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact journals.permissions@oup.com. _Briefings in Bioinformatics_ , 2025, **26(3)** , bbaf292 https://doi.org/10.1093/bib/bbaf292 Problem Solving Protocol 

