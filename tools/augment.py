import os
import uuid
from PIL import Image

input_folder = ''
output_folder = ''

os.makedirs(output_folder, exist_ok=True)

def generate_new_filename(original_filename, new_uuid):
    base_name, ext = os.path.splitext(original_filename)
    tag = base_name.split('_')[-1]
    return f"{new_uuid}_{tag}{ext}"

def augment_image(image_path):
    image = Image.open(image_path)
    aug_images = []
    
    # Rotate 90 degrees
    aug_images.append(image.rotate(90, expand=True))
    
    # Rotate 180 degrees
    aug_images.append(image.rotate(180, expand=True))
    
    # Rotate 270 degrees
    aug_images.append(image.rotate(270, expand=True))
    
    # Flip vertically
    aug_images.append(image.transpose(Image.FLIP_TOP_BOTTOM))
    
    # Flip horizontally
    aug_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    return aug_images

image_count = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        image_count += 1
        file_path = os.path.join(input_folder, filename)
        
        augmented_images = augment_image(file_path)
        
        for i, augmented_image in enumerate(augmented_images):
            new_uuid = str(uuid.uuid4())
            new_filename = generate_new_filename(filename, new_uuid)
            output_path = os.path.join(output_folder, new_filename)
            augmented_image.save(output_path)
            print(f"Processed image {image_count}, saved augmented image: {output_path}")
