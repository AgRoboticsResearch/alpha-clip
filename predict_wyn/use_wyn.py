import alpha_clip
import cv2

image = cv2.imread('F:\doctor\AlphaCLIP\image_wyn\straw_crop/1/1.png')
#alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="F:/doctor/AlphaCLIP/checkpoint/clip_b16_grit1m_fultune_8xe.pth", device="cpu"), 
alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="F:/doctor/AlphaCLIP/checkpoint/clip_b16_grit1m_fultune_8xe.pth", device="cuda"), 
image_features = model.visual(image, alpha)

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])
alpha = mask_transform(binary_mask * 255)