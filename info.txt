How to use:

    -Put input images (as jpg) in order in an input folder (a folder is provided inside src)
    -Inside src, run "python main.py input-images" (or your input folder)
    -Wait. Takes around 1 min on cuda rtx 4060.
    -Output is a pdf file called "processed.pdf". Images merged in standard sort order.

Tools dir has some tools I used to make the training data

    -bbox_add_to_folders runs bounding box algorithm and saves to a folder
    -labeller is a gui 3 class labeller (ended up using 2 classes in ViT)
    -augment is data augmentation, 5x the training data
    -dataset_loader splits images into test/train, with an 80/20 split

ViT Parameters:

    Patches = 16x16
    Model/embedding/hidden dim = 16
    2 Transformer-Encoder Blocks
    4 Heads of attention
    Output dim = 2

ViT Architecture:

    Linear embedding
    Classification token
    Positional encoding (sin/cos)

    Transformer-Encoder Blocks:
        LayerNorm
        Multi Headed Self attention
        LayerNorm
        MLP
            Linear to 4*model_dim
            GELU
            Linear back to model_dim

    Classification MLP:
        model_dim -> 2
        softmax

5890 Total Trainable Parameters
97.8% Accuracy on ~30k training images
