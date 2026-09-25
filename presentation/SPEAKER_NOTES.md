# Speaker notes (target: 11 minutes, limit 12)

Numbers in *italics* are read from the slides (they come from `results/`); say them as shown.

| # | slide | time | what to say |
|---|---|---|---|
| 1 | Title | 0:15 | Who we are; the project compares a learned CNN with a wavelet scattering network on fracture detection and explains both. |
| 2 | Task and questions | 0:40 | Binary task, all body regions. Four questions: learned vs fixed features, what the filters look like, which XAI method works and which ones cannot be applied to ScatNet, and whether our own Occlusion matches Captum's. |
| 3 | Dataset | 0:40 | Three splits from Kaggle. We load grey (X-rays have one channel), 224x224. Train for CV and final training, val only to pick the final epoch, test used once. Augmentation list. |
| 4 | Duplicates | 0:50 | The key data finding: many images are copies of the same radiograph. If copies fall on both sides of a split, CV measures memory. So folds are grouped and we also report accuracy on test images without a copy in train. |
| 5 | Architectures | 1:00 | Three feature extractors, one classifier (exam requirement). CNN: 4 conv blocks, first layer 7x7 so the filters are readable. ScatNet: Kymatio, J=4, L=8, order 2, 417 maps. Both end on 14x14, so only the input size of the classifier differs. ResNet18 as a pretrained reference. |
| 6 | ScatNet | 0:45 | The three orders of coefficients; modulus is the non-linearity, averaging the pooling; invariance to shifts below 16 px and stability to deformations; nothing is learned, which is both its strength and its limit. |
| 7 | Protocol | 0:40 | Grouped stratified 5-fold CV, score at the last epoch (no selection on the fold), best model by CV F1, final model on all of train with val picking the epoch, test once with a bootstrap interval and McNemar. |
| 8 | CV results | 0:45 | Mean accuracy and F1 per model (exam item 5). Learning curves: train and validation on the same axes; comment on the gap (overfitting) per model. |
| 9 | Test | 0:45 | Test accuracy with its 95% interval, all above 75%. Clean-test column: how much copies inflate the score. Confusion matrices: missed fractures are the costly error. |
| 10 | Best model and why | 1:00 | Say which model wins and whether McNemar says the gap is real. Why: fixed wavelets vs learned filters vs pretrained deep features; where the parameters are. |
| 11 | Filters | 0:45 | CNN filters vs Morlet wavelets; the frequency plot: wavelets tile scales and angles uniformly, the CNN concentrates on what the X-rays need. |
| 12 | Six XAI methods | 0:45 | Families; the two applicability limits on ScatNet: Grad-CAM (no learned conv maps) and Guided Backprop (only the classifier has ReLUs). |
| 13-14 | XAI grids | 0:50 | Where the maps agree (bone edges, fracture line) and where they do not (background, text markers). ScatNet maps are smoother / more spread. |
| 16 | Scratch vs Captum | 0:30 | Algorithm in four steps; identical up to float error, also checked by a unit test. |
| 17 | Deletion test | 0:40 | How faithfulness is measured; which method wins on each model; Random as the reference. |
| 18 | Agreement | 0:20 | Gradient methods agree among themselves, perturbation methods among themselves (optional slide if short on time). |
| 19 | Conclusions | 0:40 | One sentence per question from slide 2, plus the limitations. |

Total about 11:30. Backup slides (previous iteration, reproducibility) are for questions only.

## Likely questions

- *Why the same classifier?* So that the comparison isolates the feature extractor (exam requirement).
- *Why grey input?* X-rays have one channel; RGB would triple ScatNet's coefficients for nothing.
- *Why 7x7 in the first CNN layer?* 3x3 filters are too small to show a shape; 7x7 can be compared with the wavelets.
- *Why J=4?* It brings ScatNet to the same 14x14 grid as the CNN; the largest wavelet (53 px) covers a bone width.
- *Why is Grad-CAM not applicable to ScatNet?* It weights *learned* feature maps by their gradient; ScatNet's maps are fixed wavelet responses, so the map would show where the wavelets respond, not what the network learned.
- *Is the CV estimate optimistic?* Folds are grouped by near-duplicates and scored at the last epoch; rotated copies can still slip through (limitation).
- *Why Occlusion from scratch?* It is fully specified (window, stride, baseline, averaging), so equality with Captum is a strict test.
