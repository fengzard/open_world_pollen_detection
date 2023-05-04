# Deep learning and pollen detection in the Open World


Image stack, annotation, and ground-truth mask data can be accessed from the Illinois Databank (Feng et al. 2023): https://databank.illinois.edu/datasets/IDB-5855228?code=xjXBDHYYMDtcHgN1gqQWaYOb2NT_YbUrbfKpBeZkYWc.

**Authors**: Jennifer T. Feng, Shu Kong, Timme Donders, Surangi Punyasena

**Last edited**: March 12, 2023

## Abstract
1.	Fossil pollen-based paleoclimatic and paleoecological reconstructions rely on visual identifications that can be automated using computer vision. To date, the majority of automated approaches have focused on pollen classification in well-controlled environments, with few existing protocols for pollen detection and whole-slide image processing. Automated pollen detection is prerequisite for high-throughput pollen analysis. New slides captured in the open world potentially introduce rare and novel taxa, making pollen detection in the open world much more challenging than pollen classification in controlled environments.
2.	We explored pollen detection in the open world by focusing on three significant yet underexplored issues. We first addressed **taxonomic bias** – missed detections of smaller, rarer pollen types. We fused an expert model trained on this minority class with our general pollen detector. We next addressed **domain gaps** – differences in image magnification and resolution across microscopes – by fine-tuning our detector on images from a new imaging domain. Lastly, we developed **continual learning** workflows that integrated expert feedback and allowed detectors to improve over time. We simulated human-in-the-loop annotation of three microscope slides by using a trained detector to detect specimens on new slides, validating the detections and tagging incorrect detections, and using the corrected detections to re-train the detector for the new time period.
3.	In our experiment addressing taxonomic bias, fusing the expert model with the general detector improved the detection performance measured by mean average precision (mAP) from 73.21% to 75.09%, and increased detection recall by 2%. Recall at the 20% precision level for three small-grained taxa increased 11%, 25%, and 50%. In our experiment addressing domain gaps, fine-tuning the general detector on images from the new domain increased mAP in the new domain from 32.56% to 65.99%. In our experiment addressing continual learning, we increased mAP in consecutive time periods using human-in-the-loop annotations on a held-out validation set from 41.10% to 59.93%.
4.	Effective pollen detectors open new avenues of paleoecology research, creating long-term, high-quality observational records for paleoclimate analyses, improving the accuracy of diversity estimates, and helping with the discovery of rare pollen types in deep-time material. Our methods can be applied to other visually diverse biological data, including algae, fungal spores, and plant cuticle.

**Keywords:** continual learning, deep learning, domain gaps, open-world, palynology, pollen grain detection, rare species, small grains, taxonomic bias

