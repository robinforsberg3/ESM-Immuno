# ESM-Immuno
Fine-tuning ESM-2 using LoRA for neoantigen immunogenicity prediction

## Introduction:
Somatic mutations in cancerous cells can result in tumor-specific antigens (neoantigens) presented by the major histocompatibility complex, which are recognized by the host's immune system (Lang et al.,
2022; Pishesha, Harmand and Ploegh, 2022). Recent advancements in genomic sequencing and proteomics have allowed for accurate identification of potentially immunogenic neoantigens that can facilitate a
humoral immune response (Fang et al., 2022; Purcell, Ramarathinam and Ternette, 2019). Predicting which neoantigens produce an immune response remains an important challenge in developing neoantigen
vaccines (Carri et al., 2023; Li et al., 2021). Common approaches rely heavily on binding affinity of the peptide-MHC complex (pMHC), neglecting important immunological features hindering predictive
ability (Jurtz et al., 2017; Shao et al., 2020). Recent methods have incorporated immunologically informed features such as host and mutated peptide dissimilarity, pMHC stability, and T cell selectivity
(Albert et al., 2023; Li et al., 2024). Due to limited and biased datasets, immunologically informed models still struggle to capture the complex relationships that determine neoantigen immunogenicity. 

This project proposes the use protein language models, trained on extensive amino acid sequence datasets. Protein language models harnessing the network architecture of large language models have become
widely used for learning structural and functional features of proteins based on sequence alone (Brandes et al., 2022; Jumper et al., 2021). Features learned can aid in prediction tasks where data
availability is limited by fine-tuning the models on task-specific datasets (Zhang and Liu, 2024). Fine-tuning protein models on downstream tasks such as immunogenicity prediction is computationally
expensive, thus parameter efficient fine-tuning methods are incorporated in this project. 

## Materials and Methods:
The dataset in this project was obtained from The Cancer Epitope Database and Analysis Resource (CEDAR) which can be downloaded from here: https://cedar.iedb.org/. The selection criteria to generate the
dataset is as follows: Include Positive Assays, Include Negative Assays, No MHC assays, No B cell assays, MHC Restriction Type: Class I, Host: Homo sapiens (human), Exclude viral antigens, Exclude
germline/self/host antigens. The dataset contains both the mutated and wild-type peptide along with its corresponding HLA class 1 allele, immune response, and assay type used to determine T cell activity.
This project utilizes a polymorphic approach including A, B, and C alleles. 

The raw data comes from a large collection of T cell assays from several studies using different methodologies to measure T cell activity in response to presented cancer neoantigens.  As a result, the
measured immune response has an associated bias. To address this, multiple assay-specific subsets were used to train the model with assay-type biological activity yielding highest accuracy and thus as part
of the data preparation process, only samples from assay-type biological activity were selected.

Only peptides of length 8-13 were kept, adhering to the binding pocket specificity of the MHC class 1 molecule (Nguyen, Szeto and Gras, 2021). Peptides with improper amino acid abbreviations and HLA
molecules with incorrect format were also discarded. Each HLA type was matched to its corresponding sequence taken from the Immuno Polymorphism Database: https://www.ebi.ac.uk/ipd/imgt/hla/alleles/.
Occurrences where two samples had identical peptide and HLA sequence but different immune response labels, only the sample corresponding to a positive immune response was retained. 

The final dataset includes an equal amount of positive and negative samples, totalling 970 samples for the entire dataset split into 80% training set and 20% validation set. The test set includes 3 alleles
(HLA-A*02:02, HLAB*51:08, HLA-B38) that are not included in the training set allowing to test for generalizability. Both wild-type, mutated and HLA sequence were merged to match the input dimensionality
and format of the protein language model. This project fine-tunes protein language model ESM-2 developed by Meta Fundamental AI Research Protein Team (FAIR) (Lin et al., 2022). ESM-2 trained on protein
sequences from the UniRef database leverages the BERT transformer architecture to predict randomly masked amino acids within the sequence. Since ESM-2 was trained for a different task than binary
classification, a classification head is attached to the end of the model. To parameter efficiently fine-tune this model, LoRA (Low-Rank Adaptation) (Hu et al., 2021) with rank 5 and alpha set to 10 is
applied to dense layers of the network, freezing all other layers except for the output projection layer of the classification head. As a result 233,282 parameters out of the full 8,067,009 of the model
is trained. The model was trained for 16 epochs with batch size of 8.

## Results:
To investigate immunogenicity prediction ESM-2 was fine-tuned using LoRA on cancer neoantigen data from CEDAR for 16 epochs on the training set and thereafter validated using the validation set. Validation
accuracy reached 84% with 84 samples correctly classified as immunogenic, 79 non-immunogenic, 18 false positives and 13 false negatives. This shows the model is effective at differentiating immunogenic
from non-immunogenic neoantigens based solely on amino acid sequence. This is supported by AUC score of 0.87, indicating a well-seperated decision boundary. All three alleles that were not present in the
training set were correctly classified, indicating the models ability to generalise to unseen alleles. 

<p div align="center">
  <img src="https://github.com/user-attachments/assets/c52fbaa9-35f1-4d19-b80a-57cd9295f7a0" width="400" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/c8c66938-2e42-43dd-8982-5340f4e76149" width="400" style="margin-right: 20px;" />
</p>

## Discussion:
To address small and often biased neo-antigen datasets, this project utilized structural and functional protein representations learned through fine-tuning a pre-trained protein language model, ESM-2,
allowing the model to focus on immunogenic specific features instead of relearning general protein properties. This can be demonstrated through high accuracy and high AUC score on a task-specific
neoantigen dataset. The model was able to generalize to unseen alleles, however only three examples is not sufficient to confidently assess this and only functions as promising potential. 
This project narrowed the scope of neo-antigen datasets filtering by assay type biological activity, limiting experimental bias. Although the model performed reasonably well, the dataset is not
significantly large enough to confidently assess the models performance in a realistic clinical setting where the pool of peptides is magnitudes larger.  

Another bottleneck of a limited training dataset is the lack of exposure to HLA alleles, although this project incorporated A, B, and C alleles, more alleles would allow the model to learn more
generalizable relationships across alleles. Future analysis will involve performance comparison against other state-of-the-art models on the dataset used in this project, along with other published
neoantigen datasets. This will allow for realistic assessment of model performance. Although the field is progressing at a fast pace there is still a significant lack of high quality harmonised datasets
that can serve as a benchmark to assess and develop more clinically relevant models. More research is needed through closely working with clinics and wet labs to produce and assess more reliable and
relevant models and datasets. 

## References:
Albert, B.A., Yang, Y., Shao, X.M., Singh, D., Smith, K.N., Anagnostou, V. and Karchin, R. (2023). Deep neural networks predict class I major histocompatibility complex epitope presentation and transfer
learn neoepitope immunogenicity. Nature Machine Intelligence, [online] pp.1–12. doi:https://doi.org/10.1038/s42256-023-00694-6.

Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. and Linial, M. (2022). ProteinBERT: a universal deep-learning model of protein sequence and function. Bioinformatics, 38(8), pp.2102–2110.
doi:https://doi.org/10.1093/bioinformatics/btac020.

Carri, I., Schwab, E., Podaza, E., Garcia, M., José Mordoh, Nielsen, M. and Barrio, M.M. (2023). Beyond MHC binding: immunogenicity prediction tools to refine neoantigen selection in cancer patients.
Exploration of Immunology, [online] pp.82–103. doi:https://doi.org/10.37349/ei.2023.00091.

Fang, X., Guo, Z., Liang, J., Wen, J., Liu, Y., Guan, X. and Li, H. (2022). Neoantigens and their potential applications in tumor immunotherapy (Review). Oncology Letters, 23(3).
doi:https://doi.org/10.3892/ol.2022.13208.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S. and Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv (Cornell University).
doi:https://doi.org/10.48550/arxiv.2106.09685.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S.A.A., Ballard, A.J., Cowie, A.,
Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J. and Back, T. (2021). Highly Accurate Protein Structure Prediction with Alphafold. Nature, [online] 596(7873), pp.583–589.
doi:https://doi.org/10.1038/s41586-021-03819-2.

Jurtz, V., Paul, S., Andreatta, M., Marcatili, P., Peters, B. and Nielsen, M. (2017). NetMHCpan-4.0: Improved Peptide–MHC Class I Interaction Predictions Integrating Eluted Ligand and Peptide Binding
Affinity Data. The Journal of Immunology, 199(9), pp.3360–3368. doi:https://doi.org/10.4049/jimmunol.1700893.

Lang, F., Schrörs, B., Löwer, M., Türeci, Ö. and Sahin, U. (2022). Identification of neoantigens for individualized therapeutic cancer vaccines. Nature Reviews Drug Discovery, 21(4), pp.261–282.
doi:https://doi.org/10.1038/s41573-021-00387-y.

Li, G., Iyer, B., Prasath, V.B.S., Ni, Y. and Salomonis, N. (2021). DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity. Briefings in Bioinformatics,
22(6). doi:https://doi.org/10.1093/bib/bbab160.

Li, S., Tan, Y., Ke, S., Hong, L. and Zhou, B. (2024). Immunogenicity Prediction with Dual Attention Enables Vaccine Target
Selection. arXiv (Cornell University). doi:https://doi.org/10.48550/arxiv.2410.02647.

Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S. and Rives, A. (2022). Evolutionary
scale prediction of atomic level protein structure with a language model. BioRxiv. doi:https://doi.org/10.1101/2022.07.20.500902.

Nguyen, A.T., Szeto, C. and Gras, S. (2021). The pockets guide to HLA class I molecules. Biochemical Society Transactions, 49(5), pp.2319–2331. doi:https://doi.org/10.1042/bst20210410.
Pishesha, N., Harmand, T.J. and Ploegh, H.L. (2022). A guide to antigen processing and presentation. Nature Reviews Immunology, [online] 22(751-764), pp.1–14. doi:https://doi.org/10.1038/s41577-022-00707
2.

Purcell, A.W., Ramarathinam, S.H. and Ternette, N. (2019). Mass spectrometry–based identification of MHC-bound peptides for immunopeptidomics. Nature Protocols, 14(6), pp.1687–1707.
doi:https://doi.org/10.1038/s41596-019-0133-y.

Shao, X., Bhattacharya, R., Huang, J.C., I.K. Ashok Sivakumar, Tokheim, C., Zheng, L., Hirsch, D., Kaminow, B., Omdahl, A., Bonsack, M., Riemer, A.B., Velculescu, V.E., Anagnostou, V., Pagel, K.A. and
Karchin, R. (2020). High-Throughput Prediction of MHC Class I and II Neoantigens with MHCnuggets. Cancer immunology research, 8(3), pp.396–408. doi:https://doi.org/10.1158/2326-6066.cir-19-0464.

Zhang, S. and Liu, J.K. (2024). SeqProFT: Applying LoRA Finetuning for Sequence-only Protein Property Predictions. arXiv (Cornell University). doi:https://doi.org/10.48550/arxiv.2411.11530.

