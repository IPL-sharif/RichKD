import os
import torch
import clip
from torchvision import datasets, transforms
from tqdm import tqdm

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_NAME = "ViT-L/14"
DEVICE = "cuda"

SAVE_PATH = "clip_cache/multi_prompt/cifar100_train_clip_logits.pt"

TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a low resolution photo of a {}",
    "a bright photo of a {}",
    "a picture of a {}",
    "a detailed image of a {}",
    "a sketch of a {}",
    "a natural photo of a {} in the wild",
    "a realistic image of a {}",
    "a high quality photo of a {}",
]

BATCH_SIZE = 128


def build_prompts_for_class(name):
    return [t.format(name) for t in TEMPLATES]


def main():

    print(f"Loading CLIP model: {MODEL_NAME}")

    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    # -------------------------------------------------------
    # CIFAR-100 TRAIN DATA
    # -------------------------------------------------------
    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=preprocess
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Get class names
    dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True
    )

    class_names = dataset.classes

    # -------------------------------------------------------
    # BUILD TEXT EMBEDDINGS
    # -------------------------------------------------------
    print("Building CLIP text embeddings...")

    class_embeddings_list = []

    for class_name in tqdm(class_names, desc="Classes"):

        prompts = build_prompts_for_class(class_name)

        tokens = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():

            text_features = model.encode_text(tokens)

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

            # Average multiple prompts
            class_embed = text_features.mean(dim=0)

            class_embed = class_embed / class_embed.norm()

            class_embeddings_list.append(
                class_embed.cpu()
            )

    class_embeddings = torch.stack(
        class_embeddings_list,
        dim=0
    ).to(DEVICE)

    print(
        "Class embedding shape:",
        class_embeddings.shape
    )

    # -------------------------------------------------------
    # COMPUTE IMAGE FEATURES + LOGITS
    # -------------------------------------------------------
    all_logits = []
    all_features = []
    all_targets = []

    for images, targets in tqdm(
        train_loader,
        desc="Computing CLIP features"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        with torch.no_grad():

            # Image features
            image_features = model.encode_image(images)

            # Normalize
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

            # CLIP classification logits
            logits = image_features @ class_embeddings.T

        # Move to CPU to save GPU memory
        all_features.append(
            image_features.cpu()
        )

        all_logits.append(
            logits.cpu()
        )

        all_targets.append(
            targets.cpu()
        )

    # -------------------------------------------------------
    # CONCATENATE
    # -------------------------------------------------------
    all_features = torch.cat(
        all_features,
        dim=0
    )

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_targets = torch.cat(
        all_targets,
        dim=0
    )

    # -------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -------------------------------------------------------
    os.makedirs(
        os.path.dirname(SAVE_PATH),
        exist_ok=True
    )

    # -------------------------------------------------------
    # SAVE EVERYTHING
    # -------------------------------------------------------
    torch.save(
        {
            "features": all_features,
            "logits": all_logits,
            "targets": all_targets,
            "class_names": class_names,
        },
        SAVE_PATH
    )

    print("\nSaved:")
    print(SAVE_PATH)

    print("\nShapes:")
    print("features:", all_features.shape)
    print("logits:", all_logits.shape)
    print("targets:", all_targets.shape)


if __name__ == "__main__":
    main()