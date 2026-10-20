# s4mae -- encoder provenance

These weights are ViT-S/14 classifiers fine-tuned for the RSNA knee competition. Their **encoder
initialisation is not a public checkpoint**, which is the one way this bundle differs from our others.

    stock DINOv2  vit_small_patch14_dinov2.lvd142m  (Apache-2.0, Meta AI)
      -> continued in-domain pretraining by us: masked autoencoding (MAE) on RSNA competition
         images only, folds 0-3, step 150000 of a 300000-step schedule (sha 353becb85cf4053c)
      -> supervised fine-tuning, ep50nb recipe, encoder_lr_mult 0.081, 5 folds

The MAE pretraining used **competition images as unlabelled data**. No competition data is
redistributed in this bundle -- it contains model weights and our own source code only.
The self-supervised weights are our own work. The base DINOv2 checkpoint remains under Apache-2.0.

Fold 4 was RESERVED from all self-supervised pretraining (`build_ssl_cache.py` refuses a fold-4
study), so fold 4 is an evaluation fold the encoder never saw, labelled or unlabelled.

Input contract is identical to `ep50nb`: 336 px, 130 mm crop, band 0.20-0.80, 12 centre planes,
native-neighbour triplets, cache spec digest 92441bac40.
