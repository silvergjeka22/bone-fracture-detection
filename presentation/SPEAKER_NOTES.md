# Speaker notes (target: 11 minutes, limit 12)

Numbers in *italics* are read from the slides (they come from `results/`); say them as shown.

| # | slide | time | what to say |
|---|---|---|---|
| 1 | Title | 0:15 | Who we are; the project compares a learned CNN with a wavelet scattering network on fracture detection, explains both, and checks whether the explanations point at the fracture. |
| 2 | Task and questions | 0:40 | Binary task on hands, legs, hips, shoulders. Five questions: learned vs fixed features, what the filters look like, which XAI method works and which ones cannot be applied to ScatNet, whether our own Occlusion matches Captum's, and whether the explanations point at the fracture. |
| 3 | Dataset | 0:40 | FracAtlas: 4,083 X-rays, 717 fractured, a radiologist's box around every fracture: that is why we chose it. Only 1 in 6 is fractured, so the loss is weighted. Our own stratified split (no official one); train for CV and final training, val only to pick the final epoch, test used once. |
| 4 | Duplicates | 0:30 | Some X-rays are saved twice. If copies fall on both sides of a split, the score measures memory. So the split and the folds are grouped: copies always stay together. |
| 5 | Architectures | 0:45 | Three feature extractors, one classifier (exam requirement). CNN: 4 conv blocks, first layer 7x7 so the filters are readable. ScatNet: Kymatio, J=4, L=8, order 2, 417 maps. Both end on 14x14, so only the input size of the classifier differs. ResNet18 as a pretrained reference. |
| 6 | ScatNet | 0:35 | The three orders of coefficients; modulus is the non-linearity, averaging the pooling; invariance to shifts below 16 px and stability to deformations; nothing is learned, which is both its strength and its limit. |
| 7 | Protocol | 0:35 | Grouped stratified 5-fold CV, score at the last epoch (no selection on the fold), weighted loss, best model by CV F1, final model on all of train with val picking the epoch, test once with a bootstrap interval and McNemar. |
| 8 | CV results | 0:35 | Mean accuracy and F1 per model (exam item 5). Learning curves: train and validation on the same axes; comment on the gap (overfitting) per model. |
| 9 | Test | 0:40 | Test accuracy with its 95% interval, all above 75%. Accuracy alone is not enough: always answering "not fractured" gives ~82%, so read F1, recall and AUC. Confusion matrices: missed fractures are the costly error. |
| 10 | Best model and why | 0:50 | Say which model wins and whether McNemar says the gap is real. Why: fixed wavelets vs learned filters vs pretrained deep features; where the parameters are. |
| 11 | Filters | 0:35 | CNN filters vs Morlet wavelets; the frequency plot: wavelets tile scales and angles uniformly, the CNN concentrates on what the X-rays need. |
| 12 | Six XAI methods | 0:35 | Families; the two applicability limits on ScatNet: Grad-CAM (no learned conv maps) and Guided Backprop (only the classifier has ReLUs). |
| 13-14 | XAI grids | 0:50 | Where the maps agree (bone edges, fracture line) and where they do not (background, text markers). ScatNet maps are smoother / more spread. |
| 16 | Scratch vs Captum | 0:25 | Algorithm in four steps; identical up to float error, also checked by a unit test. |
| 17 | Deletion test | 0:30 | How faithfulness is measured; which method wins on each model; Random as the reference. |
| 18 | Heatmap to box | 0:35 | A heatmap is hard to read, a box is not. Three steps: keep the evidence for "fractured", keep the strongest half, box around the biggest blob. Dashed = radiologist, solid = ours; faint red outside the box = evidence we would otherwise hide. |
| 19 | Does it point at the fracture? | 0:40 | Hit rate per method and model against a random point and against YOLO (trained on the boxes). Say which method finds the fracture most often and whether the CNN or ScatNet looks at the right place. Misses on labels, casts or plates are shortcuts, not failures of the method. |
| 20 | Full pipeline | 0:45 | After Linda (2025): YOLOv8 draws the box (mAP vs the FracAtlas paper's 56.2%), the best classifier decides, the best XAI method explains; when classifier and detector disagree the X-ray goes to review. Say the review share and how many fractures the pipeline still misses. |
| 21 | Conclusions | 0:35 | One sentence per question from slide 2, plus the limitations. |

Total about 11:35. Backup slides (previous iteration, reproducibility, ScatNet boxes, method agreement) are for questions only.

## Likely questions

- *Why the same classifier?* So that the comparison isolates the feature extractor (exam requirement).
- *Why grey input?* X-rays have one channel; RGB would triple ScatNet's coefficients for nothing.
- *Why 7x7 in the first CNN layer?* 3x3 filters are too small to show a shape; 7x7 can be compared with the wavelets.
- *Why J=4?* It brings ScatNet to the same 14x14 grid as the CNN; the largest wavelet (53 px) covers a bone width.
- *Why is Grad-CAM not applicable to ScatNet?* It weights *learned* feature maps by their gradient; ScatNet's maps are fixed wavelet responses, so the map would show where the wavelets respond, not what the network learned.
- *Is the CV estimate optimistic?* Folds are grouped by near-duplicates and scored at the last epoch; rotated copies can still slip through (limitation).
- *Why Occlusion from scratch?* It is fully specified (window, stride, baseline, averaging), so equality with Captum is a strict test.
- *Why FracAtlas?* It is the only public fracture dataset of this size with the fractures marked by radiologists, so the explanations can be checked, not only looked at.
- *Why is accuracy not enough here?* 1 X-ray in 6 is fractured: always answering "not fractured" already gives about 82%. F1, recall and AUC of the fractured class tell the real story.
- *Why the pointing game and not only IoU?* Fractures are thin lines and the maps work on 16-px patches, so the IoU stays low even when the box is in the right place; the hottest point is a fairer test.
- *Does a box mean the model detects fractures?* No: it shows where the evidence for "fractured" is. The hit rate against a random point measures how often that is the real fracture.
- *Why YOLO if the classifiers already give boxes?* YOLO is trained *with* the radiologists' boxes, the classifiers never see one: YOLO shows how well a box can be found, the XAI boxes show where the classifier looks. The pipeline uses both and asks for review when they disagree.
- *Why not the whole paper (CT, graph network)?* FracAtlas has X-rays only, and the paper gives no numbers showing that the graph network helps.
