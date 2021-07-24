from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms

from torchvision.models import resnet50
from torchcam.cams import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import sys

sys.path.insert(0, '/kaggle/code/ConeDetectionPointpillarsV2/second/pytorch/models/backbones')
from BoTnet import *
from PIL import Image

# /home/starlet/kaggle/input/MURA-v1.1/train/XR_HUMERUS/patient01856/study1_positive/image1.png
image_size = 224
img_filename = '/home/starlet/kaggle/input/MURA-v1.1/train/XR_HUMERUS/patient01856/study1_positive/image2.png'
model = BotNet128_S1(num_classes=2, heads=8, resolution=(image_size, image_size))
model.load_state_dict(torch.load('/home/starlet/kaggle/code/ResearchAssistant/BotNet128_S1_224_pretrained_wfreeze_aug_bin_unfreeze_MURA_bestkappa.pth'))
model.eval()
cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')
# Get your input
# img = read_image("/home/starlet/kaggle/input/MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image1.png")
img = Image.open(img_filename).convert('RGB')
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Preprocess it for your chosen model
input_tensor = val_transform(img)

print(input_tensor.shape)
# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
print(out)
print(torch.max(out, 1))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# print(model)
# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(transforms.ToTensor()(img)),
                      to_pil_image(activation_map, mode='F'), alpha=0.9)
# Display it
# print(list(model.named_modules()))
plt.imshow(result);
plt.axis('off');
plt.tight_layout();
plt.show()

# model[('conv1', torch.Size([1, 64, 112, 112])), ('bn1', torch.Size([1, 64, 112, 112])), (
# 'relu', torch.Size([1, 64, 112, 112])), ('layer1.0.conv1', torch.Size([1, 64, 112, 112])), (
#       'layer1.0.bn1', torch.Size([1, 64, 112, 112])), ('layer1.0.conv2.0', torch.Size([1, 64, 56, 56])), (
#       'layer1.0.conv2', torch.Size([1, 64, 56, 56])), ('layer1.0.bn2', torch.Size([1, 64, 56, 56])), (
#       'layer1.0.conv3', torch.Size([1, 256, 56, 56])), ('layer1.0.bn3', torch.Size([1, 256, 56, 56])), (
#       'layer1.0.shortcut.0', torch.Size([1, 64, 56, 56])), ('layer1.0.shortcut.1', torch.Size([1, 256, 56, 56])), (
#       'layer1.0.shortcut.2', torch.Size([1, 256, 56, 56])), ('layer1.0.shortcut', torch.Size([1, 256, 56, 56])), (
#       'layer1.0', torch.Size([1, 256, 56, 56])), ('layer1.1.conv1', torch.Size([1, 64, 56, 56])), (
#       'layer1.1.bn1', torch.Size([1, 64, 56, 56])), ('layer1.1.conv2.0', torch.Size([1, 64, 56, 56])), (
#       'layer1.1.conv2', torch.Size([1, 64, 56, 56])), ('layer1.1.bn2', torch.Size([1, 64, 56, 56])), (
#       'layer1.1.conv3', torch.Size([1, 256, 56, 56])), ('layer1.1.bn3', torch.Size([1, 256, 56, 56])), (
#       'layer1.1.shortcut', torch.Size([1, 256, 56, 56])), ('layer1.1', torch.Size([1, 256, 56, 56])), (
#       'layer1.2.conv1', torch.Size([1, 64, 56, 56])), ('layer1.2.bn1', torch.Size([1, 64, 56, 56])), (
#       'layer1.2.conv2.0', torch.Size([1, 64, 56, 56])), ('layer1.2.conv2', torch.Size([1, 64, 56, 56])), (
#       'layer1.2.bn2', torch.Size([1, 64, 56, 56])), ('layer1.2.conv3', torch.Size([1, 256, 56, 56])), (
#       'layer1.2.bn3', torch.Size([1, 256, 56, 56])), ('layer1.2.shortcut', torch.Size([1, 256, 56, 56])), (
#       'layer1.2', torch.Size([1, 256, 56, 56])), ('layer1', torch.Size([1, 256, 56, 56])), (
#       'layer2.0.conv1', torch.Size([1, 128, 56, 56])), ('layer2.0.bn1', torch.Size([1, 128, 56, 56])), (
#       'layer2.0.conv2.0', torch.Size([1, 128, 28, 28])), (
#       'layer2.0.conv2.1.ChannelGate.avg_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate.max_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.0.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.0.conv2.1.ChannelGate', torch.Size([1, 128, 28, 28])), (
#       'layer2.0.conv2.1.CoordAtt.pool_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.0.conv2.1.CoordAtt.pool_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.0.conv2.1.CoordAtt.conv1', torch.Size([1, 16, 56, 1])), (
#       'layer2.0.conv2.1.CoordAtt.bn1', torch.Size([1, 16, 56, 1])), (
#       'layer2.0.conv2.1.CoordAtt.relu', torch.Size([1, 16, 56, 1])), (
#       'layer2.0.conv2.1.CoordAtt.conv_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.0.conv2.1.CoordAtt.conv_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.0.conv2.1.CoordAtt', torch.Size([1, 128, 28, 28])), ('layer2.0.conv2.1', torch.Size([1, 128, 28, 28])), (
#       'layer2.0.conv2', torch.Size([1, 128, 28, 28])), ('layer2.0.bn2', torch.Size([1, 128, 28, 28])), (
#       'layer2.0.conv3', torch.Size([1, 512, 28, 28])), ('layer2.0.bn3', torch.Size([1, 512, 28, 28])), (
#       'layer2.0.shortcut.0', torch.Size([1, 256, 28, 28])), ('layer2.0.shortcut.1', torch.Size([1, 512, 28, 28])), (
#       'layer2.0.shortcut.2', torch.Size([1, 512, 28, 28])), ('layer2.0.shortcut', torch.Size([1, 512, 28, 28])), (
#       'layer2.0', torch.Size([1, 512, 28, 28])), ('layer2.1.conv1', torch.Size([1, 128, 28, 28])), (
#       'layer2.1.bn1', torch.Size([1, 128, 28, 28])), ('layer2.1.conv2.0', torch.Size([1, 128, 28, 28])), (
#       'layer2.1.conv2.1.ChannelGate.avg_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate.max_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.1.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.1.conv2.1.ChannelGate', torch.Size([1, 128, 28, 28])), (
#       'layer2.1.conv2.1.CoordAtt.pool_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.1.conv2.1.CoordAtt.pool_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.1.conv2.1.CoordAtt.conv1', torch.Size([1, 16, 56, 1])), (
#       'layer2.1.conv2.1.CoordAtt.bn1', torch.Size([1, 16, 56, 1])), (
#       'layer2.1.conv2.1.CoordAtt.relu', torch.Size([1, 16, 56, 1])), (
#       'layer2.1.conv2.1.CoordAtt.conv_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.1.conv2.1.CoordAtt.conv_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.1.conv2.1.CoordAtt', torch.Size([1, 128, 28, 28])), ('layer2.1.conv2.1', torch.Size([1, 128, 28, 28])), (
#       'layer2.1.conv2', torch.Size([1, 128, 28, 28])), ('layer2.1.bn2', torch.Size([1, 128, 28, 28])), (
#       'layer2.1.conv3', torch.Size([1, 512, 28, 28])), ('layer2.1.bn3', torch.Size([1, 512, 28, 28])), (
#       'layer2.1.shortcut', torch.Size([1, 512, 28, 28])), ('layer2.1', torch.Size([1, 512, 28, 28])), (
#       'layer2.2.conv1', torch.Size([1, 128, 28, 28])), ('layer2.2.bn1', torch.Size([1, 128, 28, 28])), (
#       'layer2.2.conv2.0', torch.Size([1, 128, 28, 28])), (
#       'layer2.2.conv2.1.ChannelGate.avg_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate.max_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.2.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.2.conv2.1.ChannelGate', torch.Size([1, 128, 28, 28])), (
#       'layer2.2.conv2.1.CoordAtt.pool_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.2.conv2.1.CoordAtt.pool_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.2.conv2.1.CoordAtt.conv1', torch.Size([1, 16, 56, 1])), (
#       'layer2.2.conv2.1.CoordAtt.bn1', torch.Size([1, 16, 56, 1])), (
#       'layer2.2.conv2.1.CoordAtt.relu', torch.Size([1, 16, 56, 1])), (
#       'layer2.2.conv2.1.CoordAtt.conv_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.2.conv2.1.CoordAtt.conv_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.2.conv2.1.CoordAtt', torch.Size([1, 128, 28, 28])), ('layer2.2.conv2.1', torch.Size([1, 128, 28, 28])), (
#       'layer2.2.conv2', torch.Size([1, 128, 28, 28])), ('layer2.2.bn2', torch.Size([1, 128, 28, 28])), (
#       'layer2.2.conv3', torch.Size([1, 512, 28, 28])), ('layer2.2.bn3', torch.Size([1, 512, 28, 28])), (
#       'layer2.2.shortcut', torch.Size([1, 512, 28, 28])), ('layer2.2', torch.Size([1, 512, 28, 28])), (
#       'layer2.3.conv1', torch.Size([1, 128, 28, 28])), ('layer2.3.bn1', torch.Size([1, 128, 28, 28])), (
#       'layer2.3.conv2.0', torch.Size([1, 128, 28, 28])), (
#       'layer2.3.conv2.1.ChannelGate.avg_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate.max_pool', torch.Size([1, 128, 1, 1])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.0', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.1', torch.Size([1, 16])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.2', torch.Size([1, 16])), (
#       'layer2.3.conv2.1.ChannelGate.mlp.3', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate.mlp', torch.Size([1, 128])), (
#       'layer2.3.conv2.1.ChannelGate', torch.Size([1, 128, 28, 28])), (
#       'layer2.3.conv2.1.CoordAtt.pool_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.3.conv2.1.CoordAtt.pool_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.3.conv2.1.CoordAtt.conv1', torch.Size([1, 16, 56, 1])), (
#       'layer2.3.conv2.1.CoordAtt.bn1', torch.Size([1, 16, 56, 1])), (
#       'layer2.3.conv2.1.CoordAtt.relu', torch.Size([1, 16, 56, 1])), (
#       'layer2.3.conv2.1.CoordAtt.conv_h', torch.Size([1, 128, 28, 1])), (
#       'layer2.3.conv2.1.CoordAtt.conv_w', torch.Size([1, 128, 1, 28])), (
#       'layer2.3.conv2.1.CoordAtt', torch.Size([1, 128, 28, 28])), ('layer2.3.conv2.1', torch.Size([1, 128, 28, 28])), (
#       'layer2.3.conv2', torch.Size([1, 128, 28, 28])), ('layer2.3.bn2', torch.Size([1, 128, 28, 28])), (
#       'layer2.3.conv3', torch.Size([1, 512, 28, 28])), ('layer2.3.bn3', torch.Size([1, 512, 28, 28])), (
#       'layer2.3.shortcut', torch.Size([1, 512, 28, 28])), ('layer2.3', torch.Size([1, 512, 28, 28])), (
#       'layer2', torch.Size([1, 512, 28, 28])), ('layer3.0.conv1', torch.Size([1, 256, 28, 28])), (
#       'layer3.0.bn1', torch.Size([1, 256, 28, 28])), ('layer3.0.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.0.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.0.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.0.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.0.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.0.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.0.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.0.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.0.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.0.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.0.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.0.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.0.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.0.conv2', torch.Size([1, 256, 14, 14])), ('layer3.0.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.0.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.0.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.0.shortcut.0', torch.Size([1, 512, 14, 14])), ('layer3.0.shortcut.1', torch.Size([1, 1024, 14, 14])), (
#       'layer3.0.shortcut.2', torch.Size([1, 1024, 14, 14])), ('layer3.0.shortcut', torch.Size([1, 1024, 14, 14])), (
#       'layer3.0', torch.Size([1, 1024, 14, 14])), ('layer3.1.conv1', torch.Size([1, 256, 14, 14])), (
#       'layer3.1.bn1', torch.Size([1, 256, 14, 14])), ('layer3.1.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.1.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.1.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.1.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.1.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.1.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.1.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.1.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.1.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.1.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.1.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.1.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.1.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.1.conv2', torch.Size([1, 256, 14, 14])), ('layer3.1.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.1.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.1.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.1.shortcut', torch.Size([1, 1024, 14, 14])), ('layer3.1', torch.Size([1, 1024, 14, 14])), (
#       'layer3.2.conv1', torch.Size([1, 256, 14, 14])), ('layer3.2.bn1', torch.Size([1, 256, 14, 14])), (
#       'layer3.2.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.2.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.2.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.2.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.2.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.2.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.2.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.2.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.2.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.2.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.2.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.2.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.2.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.2.conv2', torch.Size([1, 256, 14, 14])), ('layer3.2.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.2.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.2.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.2.shortcut', torch.Size([1, 1024, 14, 14])), ('layer3.2', torch.Size([1, 1024, 14, 14])), (
#       'layer3.3.conv1', torch.Size([1, 256, 14, 14])), ('layer3.3.bn1', torch.Size([1, 256, 14, 14])), (
#       'layer3.3.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.3.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.3.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.3.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.3.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.3.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.3.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.3.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.3.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.3.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.3.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.3.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.3.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.3.conv2', torch.Size([1, 256, 14, 14])), ('layer3.3.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.3.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.3.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.3.shortcut', torch.Size([1, 1024, 14, 14])), ('layer3.3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.4.conv1', torch.Size([1, 256, 14, 14])), ('layer3.4.bn1', torch.Size([1, 256, 14, 14])), (
#       'layer3.4.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.4.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.4.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.4.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.4.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.4.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.4.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.4.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.4.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.4.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.4.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.4.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.4.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.4.conv2', torch.Size([1, 256, 14, 14])), ('layer3.4.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.4.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.4.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.4.shortcut', torch.Size([1, 1024, 14, 14])), ('layer3.4', torch.Size([1, 1024, 14, 14])), (
#       'layer3.5.conv1', torch.Size([1, 256, 14, 14])), ('layer3.5.bn1', torch.Size([1, 256, 14, 14])), (
#       'layer3.5.conv2.0', torch.Size([1, 256, 14, 14])), (
#       'layer3.5.conv2.1.ChannelGate.avg_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate.max_pool', torch.Size([1, 256, 1, 1])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.0', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.1', torch.Size([1, 32])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.2', torch.Size([1, 32])), (
#       'layer3.5.conv2.1.ChannelGate.mlp.3', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate.mlp', torch.Size([1, 256])), (
#       'layer3.5.conv2.1.ChannelGate', torch.Size([1, 256, 14, 14])), (
#       'layer3.5.conv2.1.CoordAtt.pool_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.5.conv2.1.CoordAtt.pool_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.5.conv2.1.CoordAtt.conv1', torch.Size([1, 32, 28, 1])), (
#       'layer3.5.conv2.1.CoordAtt.bn1', torch.Size([1, 32, 28, 1])), (
#       'layer3.5.conv2.1.CoordAtt.relu', torch.Size([1, 32, 28, 1])), (
#       'layer3.5.conv2.1.CoordAtt.conv_h', torch.Size([1, 256, 14, 1])), (
#       'layer3.5.conv2.1.CoordAtt.conv_w', torch.Size([1, 256, 1, 14])), (
#       'layer3.5.conv2.1.CoordAtt', torch.Size([1, 256, 14, 14])), ('layer3.5.conv2.1', torch.Size([1, 256, 14, 14])), (
#       'layer3.5.conv2', torch.Size([1, 256, 14, 14])), ('layer3.5.bn2', torch.Size([1, 256, 14, 14])), (
#       'layer3.5.conv3', torch.Size([1, 1024, 14, 14])), ('layer3.5.bn3', torch.Size([1, 1024, 14, 14])), (
#       'layer3.5.shortcut', torch.Size([1, 1024, 14, 14])), ('layer3.5', torch.Size([1, 1024, 14, 14])), (
#       'layer3', torch.Size([1, 1024, 14, 14])), ('layer4.0.conv1', torch.Size([1, 512, 14, 14])), (
#       'layer4.0.bn1', torch.Size([1, 512, 14, 14])), ('layer4.0.conv2.0.query', torch.Size([1, 512, 14, 14])), (
#       'layer4.0.conv2.0.key', torch.Size([1, 512, 14, 14])), ('layer4.0.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.0.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.0.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.0.conv2', torch.Size([1, 512, 14, 14])), ('layer4.0.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.0.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.0.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.0.shortcut.0', torch.Size([1, 2048, 14, 14])), ('layer4.0.shortcut.1', torch.Size([1, 2048, 14, 14])), (
#       'layer4.0.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.0', torch.Size([1, 2048, 14, 14])), (
#       'layer4.1.conv1', torch.Size([1, 512, 14, 14])), ('layer4.1.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.1.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.1.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.1.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.1.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.1.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.1.conv2', torch.Size([1, 512, 14, 14])), ('layer4.1.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.1.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.1.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.1.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.1', torch.Size([1, 2048, 14, 14])), (
#       'layer4.2.conv1', torch.Size([1, 512, 14, 14])), ('layer4.2.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.2.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.2.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.2.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.2.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.2.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.2.conv2', torch.Size([1, 512, 14, 14])), ('layer4.2.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.2.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.2.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.2.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.2', torch.Size([1, 2048, 14, 14])), (
#       'layer4.3.conv1', torch.Size([1, 512, 14, 14])), ('layer4.3.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.3.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.3.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.3.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.3.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.3.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.3.conv2', torch.Size([1, 512, 14, 14])), ('layer4.3.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.3.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.3.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.3.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.4.conv1', torch.Size([1, 512, 14, 14])), ('layer4.4.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.4.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.4.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.4.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.4.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.4.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.4.conv2', torch.Size([1, 512, 14, 14])), ('layer4.4.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.4.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.4.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.4.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.4', torch.Size([1, 2048, 14, 14])), (
#       'layer4.5.conv1', torch.Size([1, 512, 14, 14])), ('layer4.5.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.5.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.5.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.5.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.5.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.5.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.5.conv2', torch.Size([1, 512, 14, 14])), ('layer4.5.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.5.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.5.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.5.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.5', torch.Size([1, 2048, 14, 14])), (
#       'layer4.6.conv1', torch.Size([1, 512, 14, 14])), ('layer4.6.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.6.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.6.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.6.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.6.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.6.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.6.conv2', torch.Size([1, 512, 14, 14])), ('layer4.6.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.6.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.6.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.6.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.6', torch.Size([1, 2048, 14, 14])), (
#       'layer4.7.conv1', torch.Size([1, 512, 14, 14])), ('layer4.7.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.7.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.7.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.7.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.7.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.7.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.7.conv2', torch.Size([1, 512, 14, 14])), ('layer4.7.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.7.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.7.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.7.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.7', torch.Size([1, 2048, 14, 14])), (
#       'layer4.8.conv1', torch.Size([1, 512, 14, 14])), ('layer4.8.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.8.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.8.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.8.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.8.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.8.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.8.conv2', torch.Size([1, 512, 14, 14])), ('layer4.8.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.8.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.8.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.8.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.8', torch.Size([1, 2048, 14, 14])), (
#       'layer4.9.conv1', torch.Size([1, 512, 14, 14])), ('layer4.9.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.9.conv2.0.query', torch.Size([1, 512, 14, 14])), ('layer4.9.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.9.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.9.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.9.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.9.conv2', torch.Size([1, 512, 14, 14])), ('layer4.9.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.9.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.9.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.9.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.9', torch.Size([1, 2048, 14, 14])), (
#       'layer4.10.conv1', torch.Size([1, 512, 14, 14])), ('layer4.10.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv2.0.query', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv2.0.value', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.10.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv2', torch.Size([1, 512, 14, 14])), ('layer4.10.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.10.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.10.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.10.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.10', torch.Size([1, 2048, 14, 14])), (
#       'layer4.11.conv1', torch.Size([1, 512, 14, 14])), ('layer4.11.bn1', torch.Size([1, 512, 14, 14])), (
#       'layer4.11.conv2.0.query', torch.Size([1, 512, 14, 14])), (
#       'layer4.11.conv2.0.key', torch.Size([1, 512, 14, 14])), (
#       'layer4.11.conv2.0.value', torch.Size([1, 512, 14, 14])), (  ####################
#       'layer4.11.conv2.0.softmax', torch.Size([1, 4, 196, 196])), ('layer4.11.conv2.0', torch.Size([1, 512, 14, 14])), (
#       'layer4.11.conv2', torch.Size([1, 512, 14, 14])), ('layer4.11.bn2', torch.Size([1, 512, 14, 14])), (
#       'layer4.11.conv3', torch.Size([1, 2048, 14, 14])), ('layer4.11.bn3', torch.Size([1, 2048, 14, 14])), (
#       'layer4.11.shortcut', torch.Size([1, 2048, 14, 14])), ('layer4.11', torch.Size([1, 2048, 14, 14])), (
#       'layer4', torch.Size([1, 2048, 14, 14])), ('avgpool', torch.Size([1, 2048, 1, 1])), (
#       'fc.0', torch.Size([1, 2048])), ('fc.1', torch.Size([1, 2])), ('fc', torch.Size([1, 2])), (
#       '', torch.Size([1, 2]))]

# resnet
# ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
#  'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
#  'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight',
#  'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
#  'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean',
#  'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.0.weight',
#  'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias', 'layer1.0.downsample.1.running_mean',
#  'layer1.0.downsample.1.running_var', 'layer1.0.downsample.1.num_batches_tracked', 'layer1.1.conv1.weight',
#  'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var',
#  'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias',
#  'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight',
#  'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var',
#  'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias',
#  'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight',
#  'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var',
#  'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias',
#  'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.conv1.weight',
#  'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var',
#  'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias',
#  'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight',
#  'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var',
#  'layer2.0.bn3.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight',
#  'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var',
#  'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias',
#  'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
#  'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
#  'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
#  'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight',
#  'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var',
#  'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias',
#  'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight',
#  'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var',
#  'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias',
#  'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight',
#  'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var',
#  'layer2.3.bn2.num_batches_tracked', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias',
#  'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer3.0.conv1.weight',
#  'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var',
#  'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
#  'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight',
#  'layer3.0.bn3.weight', 'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var',
#  'layer3.0.bn3.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight',
#  'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var',
#  'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias',
#  'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight',
#  'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
#  'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias',
#  'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight',
#  'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var',
#  'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias',
#  'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.2.conv3.weight',
#  'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var',
#  'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
#  'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight',
#  'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
#  'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias',
#  'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked', 'layer3.4.conv1.weight',
#  'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var',
#  'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias',
#  'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.conv3.weight',
#  'layer3.4.bn3.weight', 'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var',
#  'layer3.4.bn3.num_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias',
#  'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight',
#  'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var',
#  'layer3.5.bn2.num_batches_tracked', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias',
#  'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer3.5.bn3.num_batches_tracked', 'layer3.6.conv1.weight',
#  'layer3.6.bn1.weight', 'layer3.6.bn1.bias', 'layer3.6.bn1.running_mean', 'layer3.6.bn1.running_var',
#  'layer3.6.bn1.num_batches_tracked', 'layer3.6.conv2.weight', 'layer3.6.bn2.weight', 'layer3.6.bn2.bias',
#  'layer3.6.bn2.running_mean', 'layer3.6.bn2.running_var', 'layer3.6.bn2.num_batches_tracked', 'layer3.6.conv3.weight',
#  'layer3.6.bn3.weight', 'layer3.6.bn3.bias', 'layer3.6.bn3.running_mean', 'layer3.6.bn3.running_var',
#  'layer3.6.bn3.num_batches_tracked', 'layer3.7.conv1.weight', 'layer3.7.bn1.weight', 'layer3.7.bn1.bias',
#  'layer3.7.bn1.running_mean', 'layer3.7.bn1.running_var', 'layer3.7.bn1.num_batches_tracked', 'layer3.7.conv2.weight',
#  'layer3.7.bn2.weight', 'layer3.7.bn2.bias', 'layer3.7.bn2.running_mean', 'layer3.7.bn2.running_var',
#  'layer3.7.bn2.num_batches_tracked', 'layer3.7.conv3.weight', 'layer3.7.bn3.weight', 'layer3.7.bn3.bias',
#  'layer3.7.bn3.running_mean', 'layer3.7.bn3.running_var', 'layer3.7.bn3.num_batches_tracked', 'layer3.8.conv1.weight',
#  'layer3.8.bn1.weight', 'layer3.8.bn1.bias', 'layer3.8.bn1.running_mean', 'layer3.8.bn1.running_var',
#  'layer3.8.bn1.num_batches_tracked', 'layer3.8.conv2.weight', 'layer3.8.bn2.weight', 'layer3.8.bn2.bias',
#  'layer3.8.bn2.running_mean', 'layer3.8.bn2.running_var', 'layer3.8.bn2.num_batches_tracked', 'layer3.8.conv3.weight',
#  'layer3.8.bn3.weight', 'layer3.8.bn3.bias', 'layer3.8.bn3.running_mean', 'layer3.8.bn3.running_var',
#  'layer3.8.bn3.num_batches_tracked', 'layer3.9.conv1.weight', 'layer3.9.bn1.weight', 'layer3.9.bn1.bias',
#  'layer3.9.bn1.running_mean', 'layer3.9.bn1.running_var', 'layer3.9.bn1.num_batches_tracked', 'layer3.9.conv2.weight',
#  'layer3.9.bn2.weight', 'layer3.9.bn2.bias', 'layer3.9.bn2.running_mean', 'layer3.9.bn2.running_var',
#  'layer3.9.bn2.num_batches_tracked', 'layer3.9.conv3.weight', 'layer3.9.bn3.weight', 'layer3.9.bn3.bias',
#  'layer3.9.bn3.running_mean', 'layer3.9.bn3.running_var', 'layer3.9.bn3.num_batches_tracked', 'layer3.10.conv1.weight',
#  'layer3.10.bn1.weight', 'layer3.10.bn1.bias', 'layer3.10.bn1.running_mean', 'layer3.10.bn1.running_var',
#  'layer3.10.bn1.num_batches_tracked', 'layer3.10.conv2.weight', 'layer3.10.bn2.weight', 'layer3.10.bn2.bias',
#  'layer3.10.bn2.running_mean', 'layer3.10.bn2.running_var', 'layer3.10.bn2.num_batches_tracked',
#  'layer3.10.conv3.weight', 'layer3.10.bn3.weight', 'layer3.10.bn3.bias', 'layer3.10.bn3.running_mean',
#  'layer3.10.bn3.running_var', 'layer3.10.bn3.num_batches_tracked', 'layer3.11.conv1.weight', 'layer3.11.bn1.weight',
#  'layer3.11.bn1.bias', 'layer3.11.bn1.running_mean', 'layer3.11.bn1.running_var', 'layer3.11.bn1.num_batches_tracked',
#  'layer3.11.conv2.weight', 'layer3.11.bn2.weight', 'layer3.11.bn2.bias', 'layer3.11.bn2.running_mean',
#  'layer3.11.bn2.running_var', 'layer3.11.bn2.num_batches_tracked', 'layer3.11.conv3.weight', 'layer3.11.bn3.weight',
#  'layer3.11.bn3.bias', 'layer3.11.bn3.running_mean', 'layer3.11.bn3.running_var', 'layer3.11.bn3.num_batches_tracked',
#  'layer3.12.conv1.weight', 'layer3.12.bn1.weight', 'layer3.12.bn1.bias', 'layer3.12.bn1.running_mean',
#  'layer3.12.bn1.running_var', 'layer3.12.bn1.num_batches_tracked', 'layer3.12.conv2.weight', 'layer3.12.bn2.weight',
#  'layer3.12.bn2.bias', 'layer3.12.bn2.running_mean', 'layer3.12.bn2.running_var', 'layer3.12.bn2.num_batches_tracked',
#  'layer3.12.conv3.weight', 'layer3.12.bn3.weight', 'layer3.12.bn3.bias', 'layer3.12.bn3.running_mean',
#  'layer3.12.bn3.running_var', 'layer3.12.bn3.num_batches_tracked', 'layer3.13.conv1.weight', 'layer3.13.bn1.weight',
#  'layer3.13.bn1.bias', 'layer3.13.bn1.running_mean', 'layer3.13.bn1.running_var', 'layer3.13.bn1.num_batches_tracked',
#  'layer3.13.conv2.weight', 'layer3.13.bn2.weight', 'layer3.13.bn2.bias', 'layer3.13.bn2.running_mean',
#  'layer3.13.bn2.running_var', 'layer3.13.bn2.num_batches_tracked', 'layer3.13.conv3.weight', 'layer3.13.bn3.weight',
#  'layer3.13.bn3.bias', 'layer3.13.bn3.running_mean', 'layer3.13.bn3.running_var', 'layer3.13.bn3.num_batches_tracked',
#  'layer3.14.conv1.weight', 'layer3.14.bn1.weight', 'layer3.14.bn1.bias', 'layer3.14.bn1.running_mean',
#  'layer3.14.bn1.running_var', 'layer3.14.bn1.num_batches_tracked', 'layer3.14.conv2.weight', 'layer3.14.bn2.weight',
#  'layer3.14.bn2.bias', 'layer3.14.bn2.running_mean', 'layer3.14.bn2.running_var', 'layer3.14.bn2.num_batches_tracked',
#  'layer3.14.conv3.weight', 'layer3.14.bn3.weight', 'layer3.14.bn3.bias', 'layer3.14.bn3.running_mean',
#  'layer3.14.bn3.running_var', 'layer3.14.bn3.num_batches_tracked', 'layer3.15.conv1.weight', 'layer3.15.bn1.weight',
#  'layer3.15.bn1.bias', 'layer3.15.bn1.running_mean', 'layer3.15.bn1.running_var', 'layer3.15.bn1.num_batches_tracked',
#  'layer3.15.conv2.weight', 'layer3.15.bn2.weight', 'layer3.15.bn2.bias', 'layer3.15.bn2.running_mean',
#  'layer3.15.bn2.running_var', 'layer3.15.bn2.num_batches_tracked', 'layer3.15.conv3.weight', 'layer3.15.bn3.weight',
#  'layer3.15.bn3.bias', 'layer3.15.bn3.running_mean', 'layer3.15.bn3.running_var', 'layer3.15.bn3.num_batches_tracked',
#  'layer3.16.conv1.weight', 'layer3.16.bn1.weight', 'layer3.16.bn1.bias', 'layer3.16.bn1.running_mean',
#  'layer3.16.bn1.running_var', 'layer3.16.bn1.num_batches_tracked', 'layer3.16.conv2.weight', 'layer3.16.bn2.weight',
#  'layer3.16.bn2.bias', 'layer3.16.bn2.running_mean', 'layer3.16.bn2.running_var', 'layer3.16.bn2.num_batches_tracked',
#  'layer3.16.conv3.weight', 'layer3.16.bn3.weight', 'layer3.16.bn3.bias', 'layer3.16.bn3.running_mean',
#  'layer3.16.bn3.running_var', 'layer3.16.bn3.num_batches_tracked', 'layer3.17.conv1.weight', 'layer3.17.bn1.weight',
#  'layer3.17.bn1.bias', 'layer3.17.bn1.running_mean', 'layer3.17.bn1.running_var', 'layer3.17.bn1.num_batches_tracked',
#  'layer3.17.conv2.weight', 'layer3.17.bn2.weight', 'layer3.17.bn2.bias', 'layer3.17.bn2.running_mean',
#  'layer3.17.bn2.running_var', 'layer3.17.bn2.num_batches_tracked', 'layer3.17.conv3.weight', 'layer3.17.bn3.weight',
#  'layer3.17.bn3.bias', 'layer3.17.bn3.running_mean', 'layer3.17.bn3.running_var', 'layer3.17.bn3.num_batches_tracked',
#  'layer3.18.conv1.weight', 'layer3.18.bn1.weight', 'layer3.18.bn1.bias', 'layer3.18.bn1.running_mean',
#  'layer3.18.bn1.running_var', 'layer3.18.bn1.num_batches_tracked', 'layer3.18.conv2.weight', 'layer3.18.bn2.weight',
#  'layer3.18.bn2.bias', 'layer3.18.bn2.running_mean', 'layer3.18.bn2.running_var', 'layer3.18.bn2.num_batches_tracked',
#  'layer3.18.conv3.weight', 'layer3.18.bn3.weight', 'layer3.18.bn3.bias', 'layer3.18.bn3.running_mean',
#  'layer3.18.bn3.running_var', 'layer3.18.bn3.num_batches_tracked', 'layer3.19.conv1.weight', 'layer3.19.bn1.weight',
#  'layer3.19.bn1.bias', 'layer3.19.bn1.running_mean', 'layer3.19.bn1.running_var', 'layer3.19.bn1.num_batches_tracked',
#  'layer3.19.conv2.weight', 'layer3.19.bn2.weight', 'layer3.19.bn2.bias', 'layer3.19.bn2.running_mean',
#  'layer3.19.bn2.running_var', 'layer3.19.bn2.num_batches_tracked', 'layer3.19.conv3.weight', 'layer3.19.bn3.weight',
#  'layer3.19.bn3.bias', 'layer3.19.bn3.running_mean', 'layer3.19.bn3.running_var', 'layer3.19.bn3.num_batches_tracked',
#  'layer3.20.conv1.weight', 'layer3.20.bn1.weight', 'layer3.20.bn1.bias', 'layer3.20.bn1.running_mean',
#  'layer3.20.bn1.running_var', 'layer3.20.bn1.num_batches_tracked', 'layer3.20.conv2.weight', 'layer3.20.bn2.weight',
#  'layer3.20.bn2.bias', 'layer3.20.bn2.running_mean', 'layer3.20.bn2.running_var', 'layer3.20.bn2.num_batches_tracked',
#  'layer3.20.conv3.weight', 'layer3.20.bn3.weight', 'layer3.20.bn3.bias', 'layer3.20.bn3.running_mean',
#  'layer3.20.bn3.running_var', 'layer3.20.bn3.num_batches_tracked', 'layer3.21.conv1.weight', 'layer3.21.bn1.weight',
#  'layer3.21.bn1.bias', 'layer3.21.bn1.running_mean', 'layer3.21.bn1.running_var', 'layer3.21.bn1.num_batches_tracked',
#  'layer3.21.conv2.weight', 'layer3.21.bn2.weight', 'layer3.21.bn2.bias', 'layer3.21.bn2.running_mean',
#  'layer3.21.bn2.running_var', 'layer3.21.bn2.num_batches_tracked', 'layer3.21.conv3.weight', 'layer3.21.bn3.weight',
#  'layer3.21.bn3.bias', 'layer3.21.bn3.running_mean', 'layer3.21.bn3.running_var', 'layer3.21.bn3.num_batches_tracked',
#  'layer3.22.conv1.weight', 'layer3.22.bn1.weight', 'layer3.22.bn1.bias', 'layer3.22.bn1.running_mean',
#  'layer3.22.bn1.running_var', 'layer3.22.bn1.num_batches_tracked', 'layer3.22.conv2.weight', 'layer3.22.bn2.weight',
#  'layer3.22.bn2.bias', 'layer3.22.bn2.running_mean', 'layer3.22.bn2.running_var', 'layer3.22.bn2.num_batches_tracked',
#  'layer3.22.conv3.weight', 'layer3.22.bn3.weight', 'layer3.22.bn3.bias', 'layer3.22.bn3.running_mean',
#  'layer3.22.bn3.running_var', 'layer3.22.bn3.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight',
#  'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked',
#  'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean',
#  'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight',
#  'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.bn3.num_batches_tracked',
#  'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias',
#  'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked',
#  'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean',
#  'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight',
#  'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked',
#  'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean',
#  'layer4.1.bn3.running_var', 'layer4.1.bn3.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight',
#  'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked',
#  'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean',
#  'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight',
#  'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked',
#  'fc.weight', 'fc.bias']

# tranfomred botnet

# ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
#  'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
#  'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight',
#  'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
#  'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean',
#  'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.0.weight',
#  'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias', 'layer1.0.downsample.1.running_mean',
#  'layer1.0.downsample.1.running_var', 'layer1.0.downsample.1.num_batches_tracked', 'layer1.1.conv1.weight',
#  'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var',
#  'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias',
#  'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight',
#  'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var',
#  'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias',
#  'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight',
#  'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var',
#  'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias',
#  'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.conv1.weight',
#  'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var',
#  'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.conv2.1.ChannelGate.mlp.1.weight',
#  'layer2.0.conv2.1.ChannelGate.mlp.1.bias', 'layer2.0.conv2.1.ChannelGate.mlp.3.weight',
#  'layer2.0.conv2.1.ChannelGate.mlp.3.bias', 'layer2.0.conv2.1.CoordAtt.conv1.weight',
#  'layer2.0.conv2.1.CoordAtt.bn1.weight', 'layer2.0.conv2.1.CoordAtt.bn1.bias',
#  'layer2.0.conv2.1.CoordAtt.bn1.running_mean', 'layer2.0.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.0.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.0.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.0.conv2.1.CoordAtt.conv_h.bias', 'layer2.0.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.0.conv2.1.CoordAtt.conv_w.bias', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean',
#  'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight',
#  'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked',
#  'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias',
#  'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked',
#  'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
#  'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
#  'layer2.1.conv2.1.ChannelGate.mlp.1.weight', 'layer2.1.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.1.conv2.1.ChannelGate.mlp.3.weight', 'layer2.1.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.1.conv2.1.CoordAtt.conv1.weight', 'layer2.1.conv2.1.CoordAtt.bn1.weight', 'layer2.1.conv2.1.CoordAtt.bn1.bias',
#  'layer2.1.conv2.1.CoordAtt.bn1.running_mean', 'layer2.1.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.1.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.1.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.1.conv2.1.CoordAtt.conv_h.bias', 'layer2.1.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.1.conv2.1.CoordAtt.conv_w.bias', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean',
#  'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight',
#  'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked',
#  'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean',
#  'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight',
#  'layer2.2.conv2.1.ChannelGate.mlp.1.weight', 'layer2.2.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.2.conv2.1.ChannelGate.mlp.3.weight', 'layer2.2.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.2.conv2.1.CoordAtt.conv1.weight', 'layer2.2.conv2.1.CoordAtt.bn1.weight', 'layer2.2.conv2.1.CoordAtt.bn1.bias',
#  'layer2.2.conv2.1.CoordAtt.bn1.running_mean', 'layer2.2.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.2.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.2.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.2.conv2.1.CoordAtt.conv_h.bias', 'layer2.2.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.2.conv2.1.CoordAtt.conv_w.bias', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean',
#  'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight',
#  'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked',
#  'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
#  'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight',
#  'layer2.3.conv2.1.ChannelGate.mlp.1.weight', 'layer2.3.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.3.conv2.1.ChannelGate.mlp.3.weight', 'layer2.3.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.3.conv2.1.CoordAtt.conv1.weight', 'layer2.3.conv2.1.CoordAtt.bn1.weight', 'layer2.3.conv2.1.CoordAtt.bn1.bias',
#  'layer2.3.conv2.1.CoordAtt.bn1.running_mean', 'layer2.3.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.3.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.3.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.3.conv2.1.CoordAtt.conv_h.bias', 'layer2.3.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.3.conv2.1.CoordAtt.conv_w.bias', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean',
#  'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight',
#  'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked',
#  'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean',
#  'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight',
#  'layer3.0.conv2.1.ChannelGate.mlp.1.weight', 'layer3.0.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.0.conv2.1.ChannelGate.mlp.3.weight', 'layer3.0.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.0.conv2.1.CoordAtt.conv1.weight', 'layer3.0.conv2.1.CoordAtt.bn1.weight', 'layer3.0.conv2.1.CoordAtt.bn1.bias',
#  'layer3.0.conv2.1.CoordAtt.bn1.running_mean', 'layer3.0.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.0.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.0.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.0.conv2.1.CoordAtt.conv_h.bias', 'layer3.0.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.0.conv2.1.CoordAtt.conv_w.bias', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean',
#  'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight',
#  'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked',
#  'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias',
#  'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked',
#  'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
#  'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight',
#  'layer3.1.conv2.1.ChannelGate.mlp.1.weight', 'layer3.1.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.1.conv2.1.ChannelGate.mlp.3.weight', 'layer3.1.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.1.conv2.1.CoordAtt.conv1.weight', 'layer3.1.conv2.1.CoordAtt.bn1.weight', 'layer3.1.conv2.1.CoordAtt.bn1.bias',
#  'layer3.1.conv2.1.CoordAtt.bn1.running_mean', 'layer3.1.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.1.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.1.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.1.conv2.1.CoordAtt.conv_h.bias', 'layer3.1.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.1.conv2.1.CoordAtt.conv_w.bias', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean',
#  'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight',
#  'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked',
#  'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
#  'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight',
#  'layer3.2.conv2.1.ChannelGate.mlp.1.weight', 'layer3.2.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.2.conv2.1.ChannelGate.mlp.3.weight', 'layer3.2.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.2.conv2.1.CoordAtt.conv1.weight', 'layer3.2.conv2.1.CoordAtt.bn1.weight', 'layer3.2.conv2.1.CoordAtt.bn1.bias',
#  'layer3.2.conv2.1.CoordAtt.bn1.running_mean', 'layer3.2.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.2.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.2.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.2.conv2.1.CoordAtt.conv_h.bias', 'layer3.2.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.2.conv2.1.CoordAtt.conv_w.bias', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean',
#  'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.2.conv3.weight', 'layer3.2.bn3.weight',
#  'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked',
#  'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean',
#  'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight',
#  'layer3.3.conv2.1.ChannelGate.mlp.1.weight', 'layer3.3.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.3.conv2.1.ChannelGate.mlp.3.weight', 'layer3.3.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.3.conv2.1.CoordAtt.conv1.weight', 'layer3.3.conv2.1.CoordAtt.bn1.weight', 'layer3.3.conv2.1.CoordAtt.bn1.bias',
#  'layer3.3.conv2.1.CoordAtt.bn1.running_mean', 'layer3.3.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.3.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.3.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.3.conv2.1.CoordAtt.conv_h.bias', 'layer3.3.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.3.conv2.1.CoordAtt.conv_w.bias', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean',
#  'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight',
#  'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked',
#  'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean',
#  'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight',
#  'layer3.4.conv2.1.ChannelGate.mlp.1.weight', 'layer3.4.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.4.conv2.1.ChannelGate.mlp.3.weight', 'layer3.4.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.4.conv2.1.CoordAtt.conv1.weight', 'layer3.4.conv2.1.CoordAtt.bn1.weight', 'layer3.4.conv2.1.CoordAtt.bn1.bias',
#  'layer3.4.conv2.1.CoordAtt.bn1.running_mean', 'layer3.4.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.4.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.4.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.4.conv2.1.CoordAtt.conv_h.bias', 'layer3.4.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.4.conv2.1.CoordAtt.conv_w.bias', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean',
#  'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight',
#  'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.4.bn3.num_batches_tracked',
#  'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean',
#  'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight',
#  'layer3.5.conv2.1.ChannelGate.mlp.1.weight', 'layer3.5.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.5.conv2.1.ChannelGate.mlp.3.weight', 'layer3.5.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.5.conv2.1.CoordAtt.conv1.weight', 'layer3.5.conv2.1.CoordAtt.bn1.weight', 'layer3.5.conv2.1.CoordAtt.bn1.bias',
#  'layer3.5.conv2.1.CoordAtt.bn1.running_mean', 'layer3.5.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.5.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.5.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.5.conv2.1.CoordAtt.conv_h.bias', 'layer3.5.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.5.conv2.1.CoordAtt.conv_w.bias', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean',
#  'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight',
#  'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer3.5.bn3.num_batches_tracked',
#  'layer3.6.conv1.weight', 'layer3.6.bn1.weight', 'layer3.6.bn1.bias', 'layer3.6.bn1.running_mean',
#  'layer3.6.bn1.running_var', 'layer3.6.bn1.num_batches_tracked', 'layer3.6.conv2.weight',
#  'layer3.6.conv2.1.ChannelGate.mlp.1.weight', 'layer3.6.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.6.conv2.1.ChannelGate.mlp.3.weight', 'layer3.6.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.6.conv2.1.CoordAtt.conv1.weight', 'layer3.6.conv2.1.CoordAtt.bn1.weight', 'layer3.6.conv2.1.CoordAtt.bn1.bias',
#  'layer3.6.conv2.1.CoordAtt.bn1.running_mean', 'layer3.6.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.6.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.6.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.6.conv2.1.CoordAtt.conv_h.bias', 'layer3.6.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.6.conv2.1.CoordAtt.conv_w.bias', 'layer3.6.bn2.weight', 'layer3.6.bn2.bias', 'layer3.6.bn2.running_mean',
#  'layer3.6.bn2.running_var', 'layer3.6.bn2.num_batches_tracked', 'layer3.6.conv3.weight', 'layer3.6.bn3.weight',
#  'layer3.6.bn3.bias', 'layer3.6.bn3.running_mean', 'layer3.6.bn3.running_var', 'layer3.6.bn3.num_batches_tracked',
#  'layer3.7.conv1.weight', 'layer3.7.bn1.weight', 'layer3.7.bn1.bias', 'layer3.7.bn1.running_mean',
#  'layer3.7.bn1.running_var', 'layer3.7.bn1.num_batches_tracked', 'layer3.7.conv2.weight',
#  'layer3.7.conv2.1.ChannelGate.mlp.1.weight', 'layer3.7.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.7.conv2.1.ChannelGate.mlp.3.weight', 'layer3.7.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.7.conv2.1.CoordAtt.conv1.weight', 'layer3.7.conv2.1.CoordAtt.bn1.weight', 'layer3.7.conv2.1.CoordAtt.bn1.bias',
#  'layer3.7.conv2.1.CoordAtt.bn1.running_mean', 'layer3.7.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.7.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.7.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.7.conv2.1.CoordAtt.conv_h.bias', 'layer3.7.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.7.conv2.1.CoordAtt.conv_w.bias', 'layer3.7.bn2.weight', 'layer3.7.bn2.bias', 'layer3.7.bn2.running_mean',
#  'layer3.7.bn2.running_var', 'layer3.7.bn2.num_batches_tracked', 'layer3.7.conv3.weight', 'layer3.7.bn3.weight',
#  'layer3.7.bn3.bias', 'layer3.7.bn3.running_mean', 'layer3.7.bn3.running_var', 'layer3.7.bn3.num_batches_tracked',
#  'layer3.8.conv1.weight', 'layer3.8.bn1.weight', 'layer3.8.bn1.bias', 'layer3.8.bn1.running_mean',
#  'layer3.8.bn1.running_var', 'layer3.8.bn1.num_batches_tracked', 'layer3.8.conv2.weight',
#  'layer3.8.conv2.1.ChannelGate.mlp.1.weight', 'layer3.8.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.8.conv2.1.ChannelGate.mlp.3.weight', 'layer3.8.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.8.conv2.1.CoordAtt.conv1.weight', 'layer3.8.conv2.1.CoordAtt.bn1.weight', 'layer3.8.conv2.1.CoordAtt.bn1.bias',
#  'layer3.8.conv2.1.CoordAtt.bn1.running_mean', 'layer3.8.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.8.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.8.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.8.conv2.1.CoordAtt.conv_h.bias', 'layer3.8.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.8.conv2.1.CoordAtt.conv_w.bias', 'layer3.8.bn2.weight', 'layer3.8.bn2.bias', 'layer3.8.bn2.running_mean',
#  'layer3.8.bn2.running_var', 'layer3.8.bn2.num_batches_tracked', 'layer3.8.conv3.weight', 'layer3.8.bn3.weight',
#  'layer3.8.bn3.bias', 'layer3.8.bn3.running_mean', 'layer3.8.bn3.running_var', 'layer3.8.bn3.num_batches_tracked',
#  'layer3.9.conv1.weight', 'layer3.9.bn1.weight', 'layer3.9.bn1.bias', 'layer3.9.bn1.running_mean',
#  'layer3.9.bn1.running_var', 'layer3.9.bn1.num_batches_tracked', 'layer3.9.conv2.weight',
#  'layer3.9.conv2.1.ChannelGate.mlp.1.weight', 'layer3.9.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.9.conv2.1.ChannelGate.mlp.3.weight', 'layer3.9.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.9.conv2.1.CoordAtt.conv1.weight', 'layer3.9.conv2.1.CoordAtt.bn1.weight', 'layer3.9.conv2.1.CoordAtt.bn1.bias',
#  'layer3.9.conv2.1.CoordAtt.bn1.running_mean', 'layer3.9.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.9.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.9.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.9.conv2.1.CoordAtt.conv_h.bias', 'layer3.9.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.9.conv2.1.CoordAtt.conv_w.bias', 'layer3.9.bn2.weight', 'layer3.9.bn2.bias', 'layer3.9.bn2.running_mean',
#  'layer3.9.bn2.running_var', 'layer3.9.bn2.num_batches_tracked', 'layer3.9.conv3.weight', 'layer3.9.bn3.weight',
#  'layer3.9.bn3.bias', 'layer3.9.bn3.running_mean', 'layer3.9.bn3.running_var', 'layer3.9.bn3.num_batches_tracked',
#  'layer3.10.conv1.weight', 'layer3.10.bn1.weight', 'layer3.10.bn1.bias', 'layer3.10.bn1.running_mean',
#  'layer3.10.bn1.running_var', 'layer3.10.bn1.num_batches_tracked', 'layer3.10.conv2.weight',
#  'layer3.10.conv2.1.ChannelGate.mlp.1.weight', 'layer3.10.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.10.conv2.1.ChannelGate.mlp.3.weight', 'layer3.10.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.10.conv2.1.CoordAtt.conv1.weight', 'layer3.10.conv2.1.CoordAtt.bn1.weight',
#  'layer3.10.conv2.1.CoordAtt.bn1.bias', 'layer3.10.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.10.conv2.1.CoordAtt.bn1.running_var', 'layer3.10.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.10.conv2.1.CoordAtt.conv_h.weight', 'layer3.10.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.10.conv2.1.CoordAtt.conv_w.weight', 'layer3.10.conv2.1.CoordAtt.conv_w.bias', 'layer3.10.bn2.weight',
#  'layer3.10.bn2.bias', 'layer3.10.bn2.running_mean', 'layer3.10.bn2.running_var', 'layer3.10.bn2.num_batches_tracked',
#  'layer3.10.conv3.weight', 'layer3.10.bn3.weight', 'layer3.10.bn3.bias', 'layer3.10.bn3.running_mean',
#  'layer3.10.bn3.running_var', 'layer3.10.bn3.num_batches_tracked', 'layer3.11.conv1.weight', 'layer3.11.bn1.weight',
#  'layer3.11.bn1.bias', 'layer3.11.bn1.running_mean', 'layer3.11.bn1.running_var', 'layer3.11.bn1.num_batches_tracked',
#  'layer3.11.conv2.weight', 'layer3.11.conv2.1.ChannelGate.mlp.1.weight', 'layer3.11.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.11.conv2.1.ChannelGate.mlp.3.weight', 'layer3.11.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.11.conv2.1.CoordAtt.conv1.weight', 'layer3.11.conv2.1.CoordAtt.bn1.weight',
#  'layer3.11.conv2.1.CoordAtt.bn1.bias', 'layer3.11.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.11.conv2.1.CoordAtt.bn1.running_var', 'layer3.11.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.11.conv2.1.CoordAtt.conv_h.weight', 'layer3.11.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.11.conv2.1.CoordAtt.conv_w.weight', 'layer3.11.conv2.1.CoordAtt.conv_w.bias', 'layer3.11.bn2.weight',
#  'layer3.11.bn2.bias', 'layer3.11.bn2.running_mean', 'layer3.11.bn2.running_var', 'layer3.11.bn2.num_batches_tracked',
#  'layer3.11.conv3.weight', 'layer3.11.bn3.weight', 'layer3.11.bn3.bias', 'layer3.11.bn3.running_mean',
#  'layer3.11.bn3.running_var', 'layer3.11.bn3.num_batches_tracked', 'layer3.12.conv1.weight', 'layer3.12.bn1.weight',
#  'layer3.12.bn1.bias', 'layer3.12.bn1.running_mean', 'layer3.12.bn1.running_var', 'layer3.12.bn1.num_batches_tracked',
#  'layer3.12.conv2.weight', 'layer3.12.conv2.1.ChannelGate.mlp.1.weight', 'layer3.12.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.12.conv2.1.ChannelGate.mlp.3.weight', 'layer3.12.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.12.conv2.1.CoordAtt.conv1.weight', 'layer3.12.conv2.1.CoordAtt.bn1.weight',
#  'layer3.12.conv2.1.CoordAtt.bn1.bias', 'layer3.12.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.12.conv2.1.CoordAtt.bn1.running_var', 'layer3.12.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.12.conv2.1.CoordAtt.conv_h.weight', 'layer3.12.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.12.conv2.1.CoordAtt.conv_w.weight', 'layer3.12.conv2.1.CoordAtt.conv_w.bias', 'layer3.12.bn2.weight',
#  'layer3.12.bn2.bias', 'layer3.12.bn2.running_mean', 'layer3.12.bn2.running_var', 'layer3.12.bn2.num_batches_tracked',
#  'layer3.12.conv3.weight', 'layer3.12.bn3.weight', 'layer3.12.bn3.bias', 'layer3.12.bn3.running_mean',
#  'layer3.12.bn3.running_var', 'layer3.12.bn3.num_batches_tracked', 'layer3.13.conv1.weight', 'layer3.13.bn1.weight',
#  'layer3.13.bn1.bias', 'layer3.13.bn1.running_mean', 'layer3.13.bn1.running_var', 'layer3.13.bn1.num_batches_tracked',
#  'layer3.13.conv2.weight', 'layer3.13.conv2.1.ChannelGate.mlp.1.weight', 'layer3.13.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.13.conv2.1.ChannelGate.mlp.3.weight', 'layer3.13.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.13.conv2.1.CoordAtt.conv1.weight', 'layer3.13.conv2.1.CoordAtt.bn1.weight',
#  'layer3.13.conv2.1.CoordAtt.bn1.bias', 'layer3.13.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.13.conv2.1.CoordAtt.bn1.running_var', 'layer3.13.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.13.conv2.1.CoordAtt.conv_h.weight', 'layer3.13.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.13.conv2.1.CoordAtt.conv_w.weight', 'layer3.13.conv2.1.CoordAtt.conv_w.bias', 'layer3.13.bn2.weight',
#  'layer3.13.bn2.bias', 'layer3.13.bn2.running_mean', 'layer3.13.bn2.running_var', 'layer3.13.bn2.num_batches_tracked',
#  'layer3.13.conv3.weight', 'layer3.13.bn3.weight', 'layer3.13.bn3.bias', 'layer3.13.bn3.running_mean',
#  'layer3.13.bn3.running_var', 'layer3.13.bn3.num_batches_tracked', 'layer3.14.conv1.weight', 'layer3.14.bn1.weight',
#  'layer3.14.bn1.bias', 'layer3.14.bn1.running_mean', 'layer3.14.bn1.running_var', 'layer3.14.bn1.num_batches_tracked',
#  'layer3.14.conv2.weight', 'layer3.14.conv2.1.ChannelGate.mlp.1.weight', 'layer3.14.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.14.conv2.1.ChannelGate.mlp.3.weight', 'layer3.14.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.14.conv2.1.CoordAtt.conv1.weight', 'layer3.14.conv2.1.CoordAtt.bn1.weight',
#  'layer3.14.conv2.1.CoordAtt.bn1.bias', 'layer3.14.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.14.conv2.1.CoordAtt.bn1.running_var', 'layer3.14.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.14.conv2.1.CoordAtt.conv_h.weight', 'layer3.14.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.14.conv2.1.CoordAtt.conv_w.weight', 'layer3.14.conv2.1.CoordAtt.conv_w.bias', 'layer3.14.bn2.weight',
#  'layer3.14.bn2.bias', 'layer3.14.bn2.running_mean', 'layer3.14.bn2.running_var', 'layer3.14.bn2.num_batches_tracked',
#  'layer3.14.conv3.weight', 'layer3.14.bn3.weight', 'layer3.14.bn3.bias', 'layer3.14.bn3.running_mean',
#  'layer3.14.bn3.running_var', 'layer3.14.bn3.num_batches_tracked', 'layer3.15.conv1.weight', 'layer3.15.bn1.weight',
#  'layer3.15.bn1.bias', 'layer3.15.bn1.running_mean', 'layer3.15.bn1.running_var', 'layer3.15.bn1.num_batches_tracked',
#  'layer3.15.conv2.weight', 'layer3.15.conv2.1.ChannelGate.mlp.1.weight', 'layer3.15.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.15.conv2.1.ChannelGate.mlp.3.weight', 'layer3.15.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.15.conv2.1.CoordAtt.conv1.weight', 'layer3.15.conv2.1.CoordAtt.bn1.weight',
#  'layer3.15.conv2.1.CoordAtt.bn1.bias', 'layer3.15.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.15.conv2.1.CoordAtt.bn1.running_var', 'layer3.15.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.15.conv2.1.CoordAtt.conv_h.weight', 'layer3.15.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.15.conv2.1.CoordAtt.conv_w.weight', 'layer3.15.conv2.1.CoordAtt.conv_w.bias', 'layer3.15.bn2.weight',
#  'layer3.15.bn2.bias', 'layer3.15.bn2.running_mean', 'layer3.15.bn2.running_var', 'layer3.15.bn2.num_batches_tracked',
#  'layer3.15.conv3.weight', 'layer3.15.bn3.weight', 'layer3.15.bn3.bias', 'layer3.15.bn3.running_mean',
#  'layer3.15.bn3.running_var', 'layer3.15.bn3.num_batches_tracked', 'layer3.16.conv1.weight', 'layer3.16.bn1.weight',
#  'layer3.16.bn1.bias', 'layer3.16.bn1.running_mean', 'layer3.16.bn1.running_var', 'layer3.16.bn1.num_batches_tracked',
#  'layer3.16.conv2.weight', 'layer3.16.conv2.1.ChannelGate.mlp.1.weight', 'layer3.16.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.16.conv2.1.ChannelGate.mlp.3.weight', 'layer3.16.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.16.conv2.1.CoordAtt.conv1.weight', 'layer3.16.conv2.1.CoordAtt.bn1.weight',
#  'layer3.16.conv2.1.CoordAtt.bn1.bias', 'layer3.16.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.16.conv2.1.CoordAtt.bn1.running_var', 'layer3.16.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.16.conv2.1.CoordAtt.conv_h.weight', 'layer3.16.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.16.conv2.1.CoordAtt.conv_w.weight', 'layer3.16.conv2.1.CoordAtt.conv_w.bias', 'layer3.16.bn2.weight',
#  'layer3.16.bn2.bias', 'layer3.16.bn2.running_mean', 'layer3.16.bn2.running_var', 'layer3.16.bn2.num_batches_tracked',
#  'layer3.16.conv3.weight', 'layer3.16.bn3.weight', 'layer3.16.bn3.bias', 'layer3.16.bn3.running_mean',
#  'layer3.16.bn3.running_var', 'layer3.16.bn3.num_batches_tracked', 'layer3.17.conv1.weight', 'layer3.17.bn1.weight',
#  'layer3.17.bn1.bias', 'layer3.17.bn1.running_mean', 'layer3.17.bn1.running_var', 'layer3.17.bn1.num_batches_tracked',
#  'layer3.17.conv2.weight', 'layer3.17.conv2.1.ChannelGate.mlp.1.weight', 'layer3.17.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.17.conv2.1.ChannelGate.mlp.3.weight', 'layer3.17.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.17.conv2.1.CoordAtt.conv1.weight', 'layer3.17.conv2.1.CoordAtt.bn1.weight',
#  'layer3.17.conv2.1.CoordAtt.bn1.bias', 'layer3.17.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.17.conv2.1.CoordAtt.bn1.running_var', 'layer3.17.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.17.conv2.1.CoordAtt.conv_h.weight', 'layer3.17.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.17.conv2.1.CoordAtt.conv_w.weight', 'layer3.17.conv2.1.CoordAtt.conv_w.bias', 'layer3.17.bn2.weight',
#  'layer3.17.bn2.bias', 'layer3.17.bn2.running_mean', 'layer3.17.bn2.running_var', 'layer3.17.bn2.num_batches_tracked',
#  'layer3.17.conv3.weight', 'layer3.17.bn3.weight', 'layer3.17.bn3.bias', 'layer3.17.bn3.running_mean',
#  'layer3.17.bn3.running_var', 'layer3.17.bn3.num_batches_tracked', 'layer3.18.conv1.weight', 'layer3.18.bn1.weight',
#  'layer3.18.bn1.bias', 'layer3.18.bn1.running_mean', 'layer3.18.bn1.running_var', 'layer3.18.bn1.num_batches_tracked',
#  'layer3.18.conv2.weight', 'layer3.18.conv2.1.ChannelGate.mlp.1.weight', 'layer3.18.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.18.conv2.1.ChannelGate.mlp.3.weight', 'layer3.18.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.18.conv2.1.CoordAtt.conv1.weight', 'layer3.18.conv2.1.CoordAtt.bn1.weight',
#  'layer3.18.conv2.1.CoordAtt.bn1.bias', 'layer3.18.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.18.conv2.1.CoordAtt.bn1.running_var', 'layer3.18.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.18.conv2.1.CoordAtt.conv_h.weight', 'layer3.18.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.18.conv2.1.CoordAtt.conv_w.weight', 'layer3.18.conv2.1.CoordAtt.conv_w.bias', 'layer3.18.bn2.weight',
#  'layer3.18.bn2.bias', 'layer3.18.bn2.running_mean', 'layer3.18.bn2.running_var', 'layer3.18.bn2.num_batches_tracked',
#  'layer3.18.conv3.weight', 'layer3.18.bn3.weight', 'layer3.18.bn3.bias', 'layer3.18.bn3.running_mean',
#  'layer3.18.bn3.running_var', 'layer3.18.bn3.num_batches_tracked', 'layer3.19.conv1.weight', 'layer3.19.bn1.weight',
#  'layer3.19.bn1.bias', 'layer3.19.bn1.running_mean', 'layer3.19.bn1.running_var', 'layer3.19.bn1.num_batches_tracked',
#  'layer3.19.conv2.weight', 'layer3.19.conv2.1.ChannelGate.mlp.1.weight', 'layer3.19.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.19.conv2.1.ChannelGate.mlp.3.weight', 'layer3.19.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.19.conv2.1.CoordAtt.conv1.weight', 'layer3.19.conv2.1.CoordAtt.bn1.weight',
#  'layer3.19.conv2.1.CoordAtt.bn1.bias', 'layer3.19.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.19.conv2.1.CoordAtt.bn1.running_var', 'layer3.19.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.19.conv2.1.CoordAtt.conv_h.weight', 'layer3.19.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.19.conv2.1.CoordAtt.conv_w.weight', 'layer3.19.conv2.1.CoordAtt.conv_w.bias', 'layer3.19.bn2.weight',
#  'layer3.19.bn2.bias', 'layer3.19.bn2.running_mean', 'layer3.19.bn2.running_var', 'layer3.19.bn2.num_batches_tracked',
#  'layer3.19.conv3.weight', 'layer3.19.bn3.weight', 'layer3.19.bn3.bias', 'layer3.19.bn3.running_mean',
#  'layer3.19.bn3.running_var', 'layer3.19.bn3.num_batches_tracked', 'layer3.20.conv1.weight', 'layer3.20.bn1.weight',
#  'layer3.20.bn1.bias', 'layer3.20.bn1.running_mean', 'layer3.20.bn1.running_var', 'layer3.20.bn1.num_batches_tracked',
#  'layer3.20.conv2.weight', 'layer3.20.conv2.1.ChannelGate.mlp.1.weight', 'layer3.20.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.20.conv2.1.ChannelGate.mlp.3.weight', 'layer3.20.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.20.conv2.1.CoordAtt.conv1.weight', 'layer3.20.conv2.1.CoordAtt.bn1.weight',
#  'layer3.20.conv2.1.CoordAtt.bn1.bias', 'layer3.20.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.20.conv2.1.CoordAtt.bn1.running_var', 'layer3.20.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.20.conv2.1.CoordAtt.conv_h.weight', 'layer3.20.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.20.conv2.1.CoordAtt.conv_w.weight', 'layer3.20.conv2.1.CoordAtt.conv_w.bias', 'layer3.20.bn2.weight',
#  'layer3.20.bn2.bias', 'layer3.20.bn2.running_mean', 'layer3.20.bn2.running_var', 'layer3.20.bn2.num_batches_tracked',
#  'layer3.20.conv3.weight', 'layer3.20.bn3.weight', 'layer3.20.bn3.bias', 'layer3.20.bn3.running_mean',
#  'layer3.20.bn3.running_var', 'layer3.20.bn3.num_batches_tracked', 'layer3.21.conv1.weight', 'layer3.21.bn1.weight',
#  'layer3.21.bn1.bias', 'layer3.21.bn1.running_mean', 'layer3.21.bn1.running_var', 'layer3.21.bn1.num_batches_tracked',
#  'layer3.21.conv2.weight', 'layer3.21.conv2.1.ChannelGate.mlp.1.weight', 'layer3.21.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.21.conv2.1.ChannelGate.mlp.3.weight', 'layer3.21.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.21.conv2.1.CoordAtt.conv1.weight', 'layer3.21.conv2.1.CoordAtt.bn1.weight',
#  'layer3.21.conv2.1.CoordAtt.bn1.bias', 'layer3.21.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.21.conv2.1.CoordAtt.bn1.running_var', 'layer3.21.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.21.conv2.1.CoordAtt.conv_h.weight', 'layer3.21.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.21.conv2.1.CoordAtt.conv_w.weight', 'layer3.21.conv2.1.CoordAtt.conv_w.bias', 'layer3.21.bn2.weight',
#  'layer3.21.bn2.bias', 'layer3.21.bn2.running_mean', 'layer3.21.bn2.running_var', 'layer3.21.bn2.num_batches_tracked',
#  'layer3.21.conv3.weight', 'layer3.21.bn3.weight', 'layer3.21.bn3.bias', 'layer3.21.bn3.running_mean',
#  'layer3.21.bn3.running_var', 'layer3.21.bn3.num_batches_tracked', 'layer3.22.conv1.weight', 'layer3.22.bn1.weight',
#  'layer3.22.bn1.bias', 'layer3.22.bn1.running_mean', 'layer3.22.bn1.running_var', 'layer3.22.bn1.num_batches_tracked',
#  'layer3.22.conv2.weight', 'layer3.22.conv2.1.ChannelGate.mlp.1.weight', 'layer3.22.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.22.conv2.1.ChannelGate.mlp.3.weight', 'layer3.22.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.22.conv2.1.CoordAtt.conv1.weight', 'layer3.22.conv2.1.CoordAtt.bn1.weight',
#  'layer3.22.conv2.1.CoordAtt.bn1.bias', 'layer3.22.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.22.conv2.1.CoordAtt.bn1.running_var', 'layer3.22.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.22.conv2.1.CoordAtt.conv_h.weight', 'layer3.22.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.22.conv2.1.CoordAtt.conv_w.weight', 'layer3.22.conv2.1.CoordAtt.conv_w.bias', 'layer3.22.bn2.weight',
#  'layer3.22.bn2.bias', 'layer3.22.bn2.running_mean', 'layer3.22.bn2.running_var', 'layer3.22.bn2.num_batches_tracked',
#  'layer3.22.conv3.weight', 'layer3.22.bn3.weight', 'layer3.22.bn3.bias', 'layer3.22.bn3.running_mean',
#  'layer3.22.bn3.running_var', 'layer3.22.bn3.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight',
#  'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked',
#  'layer4.0.conv2.0.relative_h', 'layer4.0.conv2.0.relative_w', 'layer4.0.conv2.0.query.weight',
#  'layer4.0.conv2.0.query.bias', 'layer4.0.conv2.0.key.weight', 'layer4.0.conv2.0.key.bias',
#  'layer4.0.conv2.0.value.weight', 'layer4.0.conv2.0.value.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
#  'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight',
#  'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var',
#  'layer4.0.bn3.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.0.bias',
#  'layer4.0.downsample.0.running_mean', 'layer4.0.downsample.0.running_var', 'layer4.0.downsample.0.num_batches_tracked',
#  'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean',
#  'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.0.relative_h',
#  'layer4.1.conv2.0.relative_w', 'layer4.1.conv2.0.query.weight', 'layer4.1.conv2.0.query.bias',
#  'layer4.1.conv2.0.key.weight', 'layer4.1.conv2.0.key.bias', 'layer4.1.conv2.0.value.weight',
#  'layer4.1.conv2.0.value.bias', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean',
#  'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight',
#  'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.1.bn3.num_batches_tracked',
#  'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean',
#  'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.0.relative_h',
#  'layer4.2.conv2.0.relative_w', 'layer4.2.conv2.0.query.weight', 'layer4.2.conv2.0.query.bias',
#  'layer4.2.conv2.0.key.weight', 'layer4.2.conv2.0.key.bias', 'layer4.2.conv2.0.value.weight',
#  'layer4.2.conv2.0.value.bias', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean',
#  'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight',
#  'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked',
#  'layer4.3.conv1.weight', 'layer4.3.bn1.weight', 'layer4.3.bn1.bias', 'layer4.3.bn1.running_mean',
#  'layer4.3.bn1.running_var', 'layer4.3.bn1.num_batches_tracked', 'layer4.3.conv2.0.relative_h',
#  'layer4.3.conv2.0.relative_w', 'layer4.3.conv2.0.query.weight', 'layer4.3.conv2.0.query.bias',
#  'layer4.3.conv2.0.key.weight', 'layer4.3.conv2.0.key.bias', 'layer4.3.conv2.0.value.weight',
#  'layer4.3.conv2.0.value.bias', 'layer4.3.bn2.weight', 'layer4.3.bn2.bias', 'layer4.3.bn2.running_mean',
#  'layer4.3.bn2.running_var', 'layer4.3.bn2.num_batches_tracked', 'layer4.3.conv3.weight', 'layer4.3.bn3.weight',
#  'layer4.3.bn3.bias', 'layer4.3.bn3.running_mean', 'layer4.3.bn3.running_var', 'layer4.3.bn3.num_batches_tracked',
#  'layer4.4.conv1.weight', 'layer4.4.bn1.weight', 'layer4.4.bn1.bias', 'layer4.4.bn1.running_mean',
#  'layer4.4.bn1.running_var', 'layer4.4.bn1.num_batches_tracked', 'layer4.4.conv2.0.relative_h',
#  'layer4.4.conv2.0.relative_w', 'layer4.4.conv2.0.query.weight', 'layer4.4.conv2.0.query.bias',
#  'layer4.4.conv2.0.key.weight', 'layer4.4.conv2.0.key.bias', 'layer4.4.conv2.0.value.weight',
#  'layer4.4.conv2.0.value.bias', 'layer4.4.bn2.weight', 'layer4.4.bn2.bias', 'layer4.4.bn2.running_mean',
#  'layer4.4.bn2.running_var', 'layer4.4.bn2.num_batches_tracked', 'layer4.4.conv3.weight', 'layer4.4.bn3.weight',
#  'layer4.4.bn3.bias', 'layer4.4.bn3.running_mean', 'layer4.4.bn3.running_var', 'layer4.4.bn3.num_batches_tracked',
#  'layer4.5.conv1.weight', 'layer4.5.bn1.weight', 'layer4.5.bn1.bias', 'layer4.5.bn1.running_mean',
#  'layer4.5.bn1.running_var', 'layer4.5.bn1.num_batches_tracked', 'layer4.5.conv2.0.relative_h',
#  'layer4.5.conv2.0.relative_w', 'layer4.5.conv2.0.query.weight', 'layer4.5.conv2.0.query.bias',
#  'layer4.5.conv2.0.key.weight', 'layer4.5.conv2.0.key.bias', 'layer4.5.conv2.0.value.weight',
#  'layer4.5.conv2.0.value.bias', 'layer4.5.bn2.weight', 'layer4.5.bn2.bias', 'layer4.5.bn2.running_mean',
#  'layer4.5.bn2.running_var', 'layer4.5.bn2.num_batches_tracked', 'layer4.5.conv3.weight', 'layer4.5.bn3.weight',
#  'layer4.5.bn3.bias', 'layer4.5.bn3.running_mean', 'layer4.5.bn3.running_var', 'layer4.5.bn3.num_batches_tracked',
#  'layer4.6.conv1.weight', 'layer4.6.bn1.weight', 'layer4.6.bn1.bias', 'layer4.6.bn1.running_mean',
#  'layer4.6.bn1.running_var', 'layer4.6.bn1.num_batches_tracked', 'layer4.6.conv2.0.relative_h',
#  'layer4.6.conv2.0.relative_w', 'layer4.6.conv2.0.query.weight', 'layer4.6.conv2.0.query.bias',
#  'layer4.6.conv2.0.key.weight', 'layer4.6.conv2.0.key.bias', 'layer4.6.conv2.0.value.weight',
#  'layer4.6.conv2.0.value.bias', 'layer4.6.bn2.weight', 'layer4.6.bn2.bias', 'layer4.6.bn2.running_mean',
#  'layer4.6.bn2.running_var', 'layer4.6.bn2.num_batches_tracked', 'layer4.6.conv3.weight', 'layer4.6.bn3.weight',
#  'layer4.6.bn3.bias', 'layer4.6.bn3.running_mean', 'layer4.6.bn3.running_var', 'layer4.6.bn3.num_batches_tracked',
#  'layer4.7.conv1.weight', 'layer4.7.bn1.weight', 'layer4.7.bn1.bias', 'layer4.7.bn1.running_mean',
#  'layer4.7.bn1.running_var', 'layer4.7.bn1.num_batches_tracked', 'layer4.7.conv2.0.relative_h',
#  'layer4.7.conv2.0.relative_w', 'layer4.7.conv2.0.query.weight', 'layer4.7.conv2.0.query.bias',
#  'layer4.7.conv2.0.key.weight', 'layer4.7.conv2.0.key.bias', 'layer4.7.conv2.0.value.weight',
#  'layer4.7.conv2.0.value.bias', 'layer4.7.bn2.weight', 'layer4.7.bn2.bias', 'layer4.7.bn2.running_mean',
#  'layer4.7.bn2.running_var', 'layer4.7.bn2.num_batches_tracked', 'layer4.7.conv3.weight', 'layer4.7.bn3.weight',
#  'layer4.7.bn3.bias', 'layer4.7.bn3.running_mean', 'layer4.7.bn3.running_var', 'layer4.7.bn3.num_batches_tracked',
#  'layer4.8.conv1.weight', 'layer4.8.bn1.weight', 'layer4.8.bn1.bias', 'layer4.8.bn1.running_mean',
#  'layer4.8.bn1.running_var', 'layer4.8.bn1.num_batches_tracked', 'layer4.8.conv2.0.relative_h',
#  'layer4.8.conv2.0.relative_w', 'layer4.8.conv2.0.query.weight', 'layer4.8.conv2.0.query.bias',
#  'layer4.8.conv2.0.key.weight', 'layer4.8.conv2.0.key.bias', 'layer4.8.conv2.0.value.weight',
#  'layer4.8.conv2.0.value.bias', 'layer4.8.bn2.weight', 'layer4.8.bn2.bias', 'layer4.8.bn2.running_mean',
#  'layer4.8.bn2.running_var', 'layer4.8.bn2.num_batches_tracked', 'layer4.8.conv3.weight', 'layer4.8.bn3.weight',
#  'layer4.8.bn3.bias', 'layer4.8.bn3.running_mean', 'layer4.8.bn3.running_var', 'layer4.8.bn3.num_batches_tracked',
#  'layer4.9.conv1.weight', 'layer4.9.bn1.weight', 'layer4.9.bn1.bias', 'layer4.9.bn1.running_mean',
#  'layer4.9.bn1.running_var', 'layer4.9.bn1.num_batches_tracked', 'layer4.9.conv2.0.relative_h',
#  'layer4.9.conv2.0.relative_w', 'layer4.9.conv2.0.query.weight', 'layer4.9.conv2.0.query.bias',
#  'layer4.9.conv2.0.key.weight', 'layer4.9.conv2.0.key.bias', 'layer4.9.conv2.0.value.weight',
#  'layer4.9.conv2.0.value.bias', 'layer4.9.bn2.weight', 'layer4.9.bn2.bias', 'layer4.9.bn2.running_mean',
#  'layer4.9.bn2.running_var', 'layer4.9.bn2.num_batches_tracked', 'layer4.9.conv3.weight', 'layer4.9.bn3.weight',
#  'layer4.9.bn3.bias', 'layer4.9.bn3.running_mean', 'layer4.9.bn3.running_var', 'layer4.9.bn3.num_batches_tracked',
#  'layer4.10.conv1.weight', 'layer4.10.bn1.weight', 'layer4.10.bn1.bias', 'layer4.10.bn1.running_mean',
#  'layer4.10.bn1.running_var', 'layer4.10.bn1.num_batches_tracked', 'layer4.10.conv2.0.relative_h',
#  'layer4.10.conv2.0.relative_w', 'layer4.10.conv2.0.query.weight', 'layer4.10.conv2.0.query.bias',
#  'layer4.10.conv2.0.key.weight', 'layer4.10.conv2.0.key.bias', 'layer4.10.conv2.0.value.weight',
#  'layer4.10.conv2.0.value.bias', 'layer4.10.bn2.weight', 'layer4.10.bn2.bias', 'layer4.10.bn2.running_mean',
#  'layer4.10.bn2.running_var', 'layer4.10.bn2.num_batches_tracked', 'layer4.10.conv3.weight', 'layer4.10.bn3.weight',
#  'layer4.10.bn3.bias', 'layer4.10.bn3.running_mean', 'layer4.10.bn3.running_var', 'layer4.10.bn3.num_batches_tracked',
#  'layer4.11.conv1.weight', 'layer4.11.bn1.weight', 'layer4.11.bn1.bias', 'layer4.11.bn1.running_mean',
#  'layer4.11.bn1.running_var', 'layer4.11.bn1.num_batches_tracked', 'layer4.11.conv2.0.relative_h',
#  'layer4.11.conv2.0.relative_w', 'layer4.11.conv2.0.query.weight', 'layer4.11.conv2.0.query.bias',
#  'layer4.11.conv2.0.key.weight', 'layer4.11.conv2.0.key.bias', 'layer4.11.conv2.0.value.weight',
#  'layer4.11.conv2.0.value.bias', 'layer4.11.bn2.weight', 'layer4.11.bn2.bias', 'layer4.11.bn2.running_mean',
#  'layer4.11.bn2.running_var', 'layer4.11.bn2.num_batches_tracked', 'layer4.11.conv3.weight', 'layer4.11.bn3.weight',
#  'layer4.11.bn3.bias', 'layer4.11.bn3.running_mean', 'layer4.11.bn3.running_var', 'layer4.11.bn3.num_batches_tracked',
#  'fc.1.weight', 'fc.1.bias']
# botnet
# ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
#  'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
#  'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.0.weight', 'layer1.0.bn2.weight',
#  'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
#  'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean',
#  'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.1.weight',
#  'layer1.0.downsample.2.weight', 'layer1.0.downsample.2.bias', 'layer1.0.downsample.2.running_mean',
#  'layer1.0.downsample.2.running_var', 'layer1.0.downsample.2.num_batches_tracked', 'layer1.1.conv1.weight',
#  'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var',
#  'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.0.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias',
#  'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight',
#  'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var',
#  'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias',
#  'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.0.weight',
#  'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var',
#  'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias',
#  'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.conv1.weight',
#  'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var',
#  'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.0.weight', 'layer2.0.conv2.1.ChannelGate.mlp.1.weight',
#  'layer2.0.conv2.1.ChannelGate.mlp.1.bias', 'layer2.0.conv2.1.ChannelGate.mlp.3.weight',
#  'layer2.0.conv2.1.ChannelGate.mlp.3.bias', 'layer2.0.conv2.1.CoordAtt.conv1.weight',
#  'layer2.0.conv2.1.CoordAtt.bn1.weight', 'layer2.0.conv2.1.CoordAtt.bn1.bias',
#  'layer2.0.conv2.1.CoordAtt.bn1.running_mean', 'layer2.0.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.0.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.0.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.0.conv2.1.CoordAtt.conv_h.bias', 'layer2.0.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.0.conv2.1.CoordAtt.conv_w.bias', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean',
#  'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight',
#  'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked',
#  'layer2.0.downsample.1.weight', 'layer2.0.downsample.2.weight', 'layer2.0.downsample.2.bias',
#  'layer2.0.downsample.2.running_mean', 'layer2.0.downsample.2.running_var', 'layer2.0.downsample.2.num_batches_tracked',
#  'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
#  'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.0.weight',
#  'layer2.1.conv2.1.ChannelGate.mlp.1.weight', 'layer2.1.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.1.conv2.1.ChannelGate.mlp.3.weight', 'layer2.1.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.1.conv2.1.CoordAtt.conv1.weight', 'layer2.1.conv2.1.CoordAtt.bn1.weight', 'layer2.1.conv2.1.CoordAtt.bn1.bias',
#  'layer2.1.conv2.1.CoordAtt.bn1.running_mean', 'layer2.1.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.1.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.1.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.1.conv2.1.CoordAtt.conv_h.bias', 'layer2.1.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.1.conv2.1.CoordAtt.conv_w.bias', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean',
#  'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight',
#  'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked',
#  'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean',
#  'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.0.weight',
#  'layer2.2.conv2.1.ChannelGate.mlp.1.weight', 'layer2.2.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.2.conv2.1.ChannelGate.mlp.3.weight', 'layer2.2.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.2.conv2.1.CoordAtt.conv1.weight', 'layer2.2.conv2.1.CoordAtt.bn1.weight', 'layer2.2.conv2.1.CoordAtt.bn1.bias',
#  'layer2.2.conv2.1.CoordAtt.bn1.running_mean', 'layer2.2.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.2.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.2.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.2.conv2.1.CoordAtt.conv_h.bias', 'layer2.2.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.2.conv2.1.CoordAtt.conv_w.bias', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean',
#  'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight',
#  'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked',
#  'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean',
#  'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.0.weight',
#  'layer2.3.conv2.1.ChannelGate.mlp.1.weight', 'layer2.3.conv2.1.ChannelGate.mlp.1.bias',
#  'layer2.3.conv2.1.ChannelGate.mlp.3.weight', 'layer2.3.conv2.1.ChannelGate.mlp.3.bias',
#  'layer2.3.conv2.1.CoordAtt.conv1.weight', 'layer2.3.conv2.1.CoordAtt.bn1.weight', 'layer2.3.conv2.1.CoordAtt.bn1.bias',
#  'layer2.3.conv2.1.CoordAtt.bn1.running_mean', 'layer2.3.conv2.1.CoordAtt.bn1.running_var',
#  'layer2.3.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer2.3.conv2.1.CoordAtt.conv_h.weight',
#  'layer2.3.conv2.1.CoordAtt.conv_h.bias', 'layer2.3.conv2.1.CoordAtt.conv_w.weight',
#  'layer2.3.conv2.1.CoordAtt.conv_w.bias', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean',
#  'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight',
#  'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked',
#  'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean',
#  'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.0.weight',
#  'layer3.0.conv2.1.ChannelGate.mlp.1.weight', 'layer3.0.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.0.conv2.1.ChannelGate.mlp.3.weight', 'layer3.0.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.0.conv2.1.CoordAtt.conv1.weight', 'layer3.0.conv2.1.CoordAtt.bn1.weight', 'layer3.0.conv2.1.CoordAtt.bn1.bias',
#  'layer3.0.conv2.1.CoordAtt.bn1.running_mean', 'layer3.0.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.0.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.0.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.0.conv2.1.CoordAtt.conv_h.bias', 'layer3.0.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.0.conv2.1.CoordAtt.conv_w.bias', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean',
#  'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight',
#  'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked',
#  'layer3.0.downsample.1.weight', 'layer3.0.downsample.2.weight', 'layer3.0.downsample.2.bias',
#  'layer3.0.downsample.2.running_mean', 'layer3.0.downsample.2.running_var', 'layer3.0.downsample.2.num_batches_tracked',
#  'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
#  'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.0.weight',
#  'layer3.1.conv2.1.ChannelGate.mlp.1.weight', 'layer3.1.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.1.conv2.1.ChannelGate.mlp.3.weight', 'layer3.1.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.1.conv2.1.CoordAtt.conv1.weight', 'layer3.1.conv2.1.CoordAtt.bn1.weight', 'layer3.1.conv2.1.CoordAtt.bn1.bias',
#  'layer3.1.conv2.1.CoordAtt.bn1.running_mean', 'layer3.1.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.1.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.1.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.1.conv2.1.CoordAtt.conv_h.bias', 'layer3.1.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.1.conv2.1.CoordAtt.conv_w.bias', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean',
#  'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight',
#  'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked',
#  'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean',
#  'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.0.weight',
#  'layer3.2.conv2.1.ChannelGate.mlp.1.weight', 'layer3.2.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.2.conv2.1.ChannelGate.mlp.3.weight', 'layer3.2.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.2.conv2.1.CoordAtt.conv1.weight', 'layer3.2.conv2.1.CoordAtt.bn1.weight', 'layer3.2.conv2.1.CoordAtt.bn1.bias',
#  'layer3.2.conv2.1.CoordAtt.bn1.running_mean', 'layer3.2.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.2.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.2.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.2.conv2.1.CoordAtt.conv_h.bias', 'layer3.2.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.2.conv2.1.CoordAtt.conv_w.bias', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean',
#  'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.2.conv3.weight', 'layer3.2.bn3.weight',
#  'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked',
#  'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean',
#  'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.0.weight',
#  'layer3.3.conv2.1.ChannelGate.mlp.1.weight', 'layer3.3.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.3.conv2.1.ChannelGate.mlp.3.weight', 'layer3.3.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.3.conv2.1.CoordAtt.conv1.weight', 'layer3.3.conv2.1.CoordAtt.bn1.weight', 'layer3.3.conv2.1.CoordAtt.bn1.bias',
#  'layer3.3.conv2.1.CoordAtt.bn1.running_mean', 'layer3.3.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.3.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.3.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.3.conv2.1.CoordAtt.conv_h.bias', 'layer3.3.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.3.conv2.1.CoordAtt.conv_w.bias', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean',
#  'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight',
#  'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked',
#  'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean',
#  'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.0.weight',
#  'layer3.4.conv2.1.ChannelGate.mlp.1.weight', 'layer3.4.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.4.conv2.1.ChannelGate.mlp.3.weight', 'layer3.4.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.4.conv2.1.CoordAtt.conv1.weight', 'layer3.4.conv2.1.CoordAtt.bn1.weight', 'layer3.4.conv2.1.CoordAtt.bn1.bias',
#  'layer3.4.conv2.1.CoordAtt.bn1.running_mean', 'layer3.4.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.4.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.4.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.4.conv2.1.CoordAtt.conv_h.bias', 'layer3.4.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.4.conv2.1.CoordAtt.conv_w.bias', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean',
#  'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight',
#  'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.4.bn3.num_batches_tracked',
#  'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean',
#  'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.0.weight',
#  'layer3.5.conv2.1.ChannelGate.mlp.1.weight', 'layer3.5.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.5.conv2.1.ChannelGate.mlp.3.weight', 'layer3.5.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.5.conv2.1.CoordAtt.conv1.weight', 'layer3.5.conv2.1.CoordAtt.bn1.weight', 'layer3.5.conv2.1.CoordAtt.bn1.bias',
#  'layer3.5.conv2.1.CoordAtt.bn1.running_mean', 'layer3.5.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.5.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.5.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.5.conv2.1.CoordAtt.conv_h.bias', 'layer3.5.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.5.conv2.1.CoordAtt.conv_w.bias', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean',
#  'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight',
#  'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer3.5.bn3.num_batches_tracked',
#  'layer3.6.conv1.weight', 'layer3.6.bn1.weight', 'layer3.6.bn1.bias', 'layer3.6.bn1.running_mean',
#  'layer3.6.bn1.running_var', 'layer3.6.bn1.num_batches_tracked', 'layer3.6.conv2.0.weight',
#  'layer3.6.conv2.1.ChannelGate.mlp.1.weight', 'layer3.6.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.6.conv2.1.ChannelGate.mlp.3.weight', 'layer3.6.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.6.conv2.1.CoordAtt.conv1.weight', 'layer3.6.conv2.1.CoordAtt.bn1.weight', 'layer3.6.conv2.1.CoordAtt.bn1.bias',
#  'layer3.6.conv2.1.CoordAtt.bn1.running_mean', 'layer3.6.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.6.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.6.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.6.conv2.1.CoordAtt.conv_h.bias', 'layer3.6.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.6.conv2.1.CoordAtt.conv_w.bias', 'layer3.6.bn2.weight', 'layer3.6.bn2.bias', 'layer3.6.bn2.running_mean',
#  'layer3.6.bn2.running_var', 'layer3.6.bn2.num_batches_tracked', 'layer3.6.conv3.weight', 'layer3.6.bn3.weight',
#  'layer3.6.bn3.bias', 'layer3.6.bn3.running_mean', 'layer3.6.bn3.running_var', 'layer3.6.bn3.num_batches_tracked',
#  'layer3.7.conv1.weight', 'layer3.7.bn1.weight', 'layer3.7.bn1.bias', 'layer3.7.bn1.running_mean',
#  'layer3.7.bn1.running_var', 'layer3.7.bn1.num_batches_tracked', 'layer3.7.conv2.0.weight',
#  'layer3.7.conv2.1.ChannelGate.mlp.1.weight', 'layer3.7.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.7.conv2.1.ChannelGate.mlp.3.weight', 'layer3.7.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.7.conv2.1.CoordAtt.conv1.weight', 'layer3.7.conv2.1.CoordAtt.bn1.weight', 'layer3.7.conv2.1.CoordAtt.bn1.bias',
#  'layer3.7.conv2.1.CoordAtt.bn1.running_mean', 'layer3.7.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.7.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.7.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.7.conv2.1.CoordAtt.conv_h.bias', 'layer3.7.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.7.conv2.1.CoordAtt.conv_w.bias', 'layer3.7.bn2.weight', 'layer3.7.bn2.bias', 'layer3.7.bn2.running_mean',
#  'layer3.7.bn2.running_var', 'layer3.7.bn2.num_batches_tracked', 'layer3.7.conv3.weight', 'layer3.7.bn3.weight',
#  'layer3.7.bn3.bias', 'layer3.7.bn3.running_mean', 'layer3.7.bn3.running_var', 'layer3.7.bn3.num_batches_tracked',
#  'layer3.8.conv1.weight', 'layer3.8.bn1.weight', 'layer3.8.bn1.bias', 'layer3.8.bn1.running_mean',
#  'layer3.8.bn1.running_var', 'layer3.8.bn1.num_batches_tracked', 'layer3.8.conv2.0.weight',
#  'layer3.8.conv2.1.ChannelGate.mlp.1.weight', 'layer3.8.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.8.conv2.1.ChannelGate.mlp.3.weight', 'layer3.8.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.8.conv2.1.CoordAtt.conv1.weight', 'layer3.8.conv2.1.CoordAtt.bn1.weight', 'layer3.8.conv2.1.CoordAtt.bn1.bias',
#  'layer3.8.conv2.1.CoordAtt.bn1.running_mean', 'layer3.8.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.8.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.8.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.8.conv2.1.CoordAtt.conv_h.bias', 'layer3.8.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.8.conv2.1.CoordAtt.conv_w.bias', 'layer3.8.bn2.weight', 'layer3.8.bn2.bias', 'layer3.8.bn2.running_mean',
#  'layer3.8.bn2.running_var', 'layer3.8.bn2.num_batches_tracked', 'layer3.8.conv3.weight', 'layer3.8.bn3.weight',
#  'layer3.8.bn3.bias', 'layer3.8.bn3.running_mean', 'layer3.8.bn3.running_var', 'layer3.8.bn3.num_batches_tracked',
#  'layer3.9.conv1.weight', 'layer3.9.bn1.weight', 'layer3.9.bn1.bias', 'layer3.9.bn1.running_mean',
#  'layer3.9.bn1.running_var', 'layer3.9.bn1.num_batches_tracked', 'layer3.9.conv2.0.weight',
#  'layer3.9.conv2.1.ChannelGate.mlp.1.weight', 'layer3.9.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.9.conv2.1.ChannelGate.mlp.3.weight', 'layer3.9.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.9.conv2.1.CoordAtt.conv1.weight', 'layer3.9.conv2.1.CoordAtt.bn1.weight', 'layer3.9.conv2.1.CoordAtt.bn1.bias',
#  'layer3.9.conv2.1.CoordAtt.bn1.running_mean', 'layer3.9.conv2.1.CoordAtt.bn1.running_var',
#  'layer3.9.conv2.1.CoordAtt.bn1.num_batches_tracked', 'layer3.9.conv2.1.CoordAtt.conv_h.weight',
#  'layer3.9.conv2.1.CoordAtt.conv_h.bias', 'layer3.9.conv2.1.CoordAtt.conv_w.weight',
#  'layer3.9.conv2.1.CoordAtt.conv_w.bias', 'layer3.9.bn2.weight', 'layer3.9.bn2.bias', 'layer3.9.bn2.running_mean',
#  'layer3.9.bn2.running_var', 'layer3.9.bn2.num_batches_tracked', 'layer3.9.conv3.weight', 'layer3.9.bn3.weight',
#  'layer3.9.bn3.bias', 'layer3.9.bn3.running_mean', 'layer3.9.bn3.running_var', 'layer3.9.bn3.num_batches_tracked',
#  'layer3.10.conv1.weight', 'layer3.10.bn1.weight', 'layer3.10.bn1.bias', 'layer3.10.bn1.running_mean',
#  'layer3.10.bn1.running_var', 'layer3.10.bn1.num_batches_tracked', 'layer3.10.conv2.0.weight',
#  'layer3.10.conv2.1.ChannelGate.mlp.1.weight', 'layer3.10.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.10.conv2.1.ChannelGate.mlp.3.weight', 'layer3.10.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.10.conv2.1.CoordAtt.conv1.weight', 'layer3.10.conv2.1.CoordAtt.bn1.weight',
#  'layer3.10.conv2.1.CoordAtt.bn1.bias', 'layer3.10.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.10.conv2.1.CoordAtt.bn1.running_var', 'layer3.10.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.10.conv2.1.CoordAtt.conv_h.weight', 'layer3.10.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.10.conv2.1.CoordAtt.conv_w.weight', 'layer3.10.conv2.1.CoordAtt.conv_w.bias', 'layer3.10.bn2.weight',
#  'layer3.10.bn2.bias', 'layer3.10.bn2.running_mean', 'layer3.10.bn2.running_var', 'layer3.10.bn2.num_batches_tracked',
#  'layer3.10.conv3.weight', 'layer3.10.bn3.weight', 'layer3.10.bn3.bias', 'layer3.10.bn3.running_mean',
#  'layer3.10.bn3.running_var', 'layer3.10.bn3.num_batches_tracked', 'layer3.11.conv1.weight', 'layer3.11.bn1.weight',
#  'layer3.11.bn1.bias', 'layer3.11.bn1.running_mean', 'layer3.11.bn1.running_var', 'layer3.11.bn1.num_batches_tracked',
#  'layer3.11.conv2.0.weight', 'layer3.11.conv2.1.ChannelGate.mlp.1.weight', 'layer3.11.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.11.conv2.1.ChannelGate.mlp.3.weight', 'layer3.11.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.11.conv2.1.CoordAtt.conv1.weight', 'layer3.11.conv2.1.CoordAtt.bn1.weight',
#  'layer3.11.conv2.1.CoordAtt.bn1.bias', 'layer3.11.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.11.conv2.1.CoordAtt.bn1.running_var', 'layer3.11.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.11.conv2.1.CoordAtt.conv_h.weight', 'layer3.11.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.11.conv2.1.CoordAtt.conv_w.weight', 'layer3.11.conv2.1.CoordAtt.conv_w.bias', 'layer3.11.bn2.weight',
#  'layer3.11.bn2.bias', 'layer3.11.bn2.running_mean', 'layer3.11.bn2.running_var', 'layer3.11.bn2.num_batches_tracked',
#  'layer3.11.conv3.weight', 'layer3.11.bn3.weight', 'layer3.11.bn3.bias', 'layer3.11.bn3.running_mean',
#  'layer3.11.bn3.running_var', 'layer3.11.bn3.num_batches_tracked', 'layer3.12.conv1.weight', 'layer3.12.bn1.weight',
#  'layer3.12.bn1.bias', 'layer3.12.bn1.running_mean', 'layer3.12.bn1.running_var', 'layer3.12.bn1.num_batches_tracked',
#  'layer3.12.conv2.0.weight', 'layer3.12.conv2.1.ChannelGate.mlp.1.weight', 'layer3.12.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.12.conv2.1.ChannelGate.mlp.3.weight', 'layer3.12.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.12.conv2.1.CoordAtt.conv1.weight', 'layer3.12.conv2.1.CoordAtt.bn1.weight',
#  'layer3.12.conv2.1.CoordAtt.bn1.bias', 'layer3.12.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.12.conv2.1.CoordAtt.bn1.running_var', 'layer3.12.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.12.conv2.1.CoordAtt.conv_h.weight', 'layer3.12.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.12.conv2.1.CoordAtt.conv_w.weight', 'layer3.12.conv2.1.CoordAtt.conv_w.bias', 'layer3.12.bn2.weight',
#  'layer3.12.bn2.bias', 'layer3.12.bn2.running_mean', 'layer3.12.bn2.running_var', 'layer3.12.bn2.num_batches_tracked',
#  'layer3.12.conv3.weight', 'layer3.12.bn3.weight', 'layer3.12.bn3.bias', 'layer3.12.bn3.running_mean',
#  'layer3.12.bn3.running_var', 'layer3.12.bn3.num_batches_tracked', 'layer3.13.conv1.weight', 'layer3.13.bn1.weight',
#  'layer3.13.bn1.bias', 'layer3.13.bn1.running_mean', 'layer3.13.bn1.running_var', 'layer3.13.bn1.num_batches_tracked',
#  'layer3.13.conv2.0.weight', 'layer3.13.conv2.1.ChannelGate.mlp.1.weight', 'layer3.13.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.13.conv2.1.ChannelGate.mlp.3.weight', 'layer3.13.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.13.conv2.1.CoordAtt.conv1.weight', 'layer3.13.conv2.1.CoordAtt.bn1.weight',
#  'layer3.13.conv2.1.CoordAtt.bn1.bias', 'layer3.13.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.13.conv2.1.CoordAtt.bn1.running_var', 'layer3.13.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.13.conv2.1.CoordAtt.conv_h.weight', 'layer3.13.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.13.conv2.1.CoordAtt.conv_w.weight', 'layer3.13.conv2.1.CoordAtt.conv_w.bias', 'layer3.13.bn2.weight',
#  'layer3.13.bn2.bias', 'layer3.13.bn2.running_mean', 'layer3.13.bn2.running_var', 'layer3.13.bn2.num_batches_tracked',
#  'layer3.13.conv3.weight', 'layer3.13.bn3.weight', 'layer3.13.bn3.bias', 'layer3.13.bn3.running_mean',
#  'layer3.13.bn3.running_var', 'layer3.13.bn3.num_batches_tracked', 'layer3.14.conv1.weight', 'layer3.14.bn1.weight',
#  'layer3.14.bn1.bias', 'layer3.14.bn1.running_mean', 'layer3.14.bn1.running_var', 'layer3.14.bn1.num_batches_tracked',
#  'layer3.14.conv2.0.weight', 'layer3.14.conv2.1.ChannelGate.mlp.1.weight', 'layer3.14.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.14.conv2.1.ChannelGate.mlp.3.weight', 'layer3.14.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.14.conv2.1.CoordAtt.conv1.weight', 'layer3.14.conv2.1.CoordAtt.bn1.weight',
#  'layer3.14.conv2.1.CoordAtt.bn1.bias', 'layer3.14.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.14.conv2.1.CoordAtt.bn1.running_var', 'layer3.14.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.14.conv2.1.CoordAtt.conv_h.weight', 'layer3.14.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.14.conv2.1.CoordAtt.conv_w.weight', 'layer3.14.conv2.1.CoordAtt.conv_w.bias', 'layer3.14.bn2.weight',
#  'layer3.14.bn2.bias', 'layer3.14.bn2.running_mean', 'layer3.14.bn2.running_var', 'layer3.14.bn2.num_batches_tracked',
#  'layer3.14.conv3.weight', 'layer3.14.bn3.weight', 'layer3.14.bn3.bias', 'layer3.14.bn3.running_mean',
#  'layer3.14.bn3.running_var', 'layer3.14.bn3.num_batches_tracked', 'layer3.15.conv1.weight', 'layer3.15.bn1.weight',
#  'layer3.15.bn1.bias', 'layer3.15.bn1.running_mean', 'layer3.15.bn1.running_var', 'layer3.15.bn1.num_batches_tracked',
#  'layer3.15.conv2.0.weight', 'layer3.15.conv2.1.ChannelGate.mlp.1.weight', 'layer3.15.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.15.conv2.1.ChannelGate.mlp.3.weight', 'layer3.15.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.15.conv2.1.CoordAtt.conv1.weight', 'layer3.15.conv2.1.CoordAtt.bn1.weight',
#  'layer3.15.conv2.1.CoordAtt.bn1.bias', 'layer3.15.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.15.conv2.1.CoordAtt.bn1.running_var', 'layer3.15.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.15.conv2.1.CoordAtt.conv_h.weight', 'layer3.15.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.15.conv2.1.CoordAtt.conv_w.weight', 'layer3.15.conv2.1.CoordAtt.conv_w.bias', 'layer3.15.bn2.weight',
#  'layer3.15.bn2.bias', 'layer3.15.bn2.running_mean', 'layer3.15.bn2.running_var', 'layer3.15.bn2.num_batches_tracked',
#  'layer3.15.conv3.weight', 'layer3.15.bn3.weight', 'layer3.15.bn3.bias', 'layer3.15.bn3.running_mean',
#  'layer3.15.bn3.running_var', 'layer3.15.bn3.num_batches_tracked', 'layer3.16.conv1.weight', 'layer3.16.bn1.weight',
#  'layer3.16.bn1.bias', 'layer3.16.bn1.running_mean', 'layer3.16.bn1.running_var', 'layer3.16.bn1.num_batches_tracked',
#  'layer3.16.conv2.0.weight', 'layer3.16.conv2.1.ChannelGate.mlp.1.weight', 'layer3.16.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.16.conv2.1.ChannelGate.mlp.3.weight', 'layer3.16.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.16.conv2.1.CoordAtt.conv1.weight', 'layer3.16.conv2.1.CoordAtt.bn1.weight',
#  'layer3.16.conv2.1.CoordAtt.bn1.bias', 'layer3.16.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.16.conv2.1.CoordAtt.bn1.running_var', 'layer3.16.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.16.conv2.1.CoordAtt.conv_h.weight', 'layer3.16.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.16.conv2.1.CoordAtt.conv_w.weight', 'layer3.16.conv2.1.CoordAtt.conv_w.bias', 'layer3.16.bn2.weight',
#  'layer3.16.bn2.bias', 'layer3.16.bn2.running_mean', 'layer3.16.bn2.running_var', 'layer3.16.bn2.num_batches_tracked',
#  'layer3.16.conv3.weight', 'layer3.16.bn3.weight', 'layer3.16.bn3.bias', 'layer3.16.bn3.running_mean',
#  'layer3.16.bn3.running_var', 'layer3.16.bn3.num_batches_tracked', 'layer3.17.conv1.weight', 'layer3.17.bn1.weight',
#  'layer3.17.bn1.bias', 'layer3.17.bn1.running_mean', 'layer3.17.bn1.running_var', 'layer3.17.bn1.num_batches_tracked',
#  'layer3.17.conv2.0.weight', 'layer3.17.conv2.1.ChannelGate.mlp.1.weight', 'layer3.17.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.17.conv2.1.ChannelGate.mlp.3.weight', 'layer3.17.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.17.conv2.1.CoordAtt.conv1.weight', 'layer3.17.conv2.1.CoordAtt.bn1.weight',
#  'layer3.17.conv2.1.CoordAtt.bn1.bias', 'layer3.17.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.17.conv2.1.CoordAtt.bn1.running_var', 'layer3.17.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.17.conv2.1.CoordAtt.conv_h.weight', 'layer3.17.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.17.conv2.1.CoordAtt.conv_w.weight', 'layer3.17.conv2.1.CoordAtt.conv_w.bias', 'layer3.17.bn2.weight',
#  'layer3.17.bn2.bias', 'layer3.17.bn2.running_mean', 'layer3.17.bn2.running_var', 'layer3.17.bn2.num_batches_tracked',
#  'layer3.17.conv3.weight', 'layer3.17.bn3.weight', 'layer3.17.bn3.bias', 'layer3.17.bn3.running_mean',
#  'layer3.17.bn3.running_var', 'layer3.17.bn3.num_batches_tracked', 'layer3.18.conv1.weight', 'layer3.18.bn1.weight',
#  'layer3.18.bn1.bias', 'layer3.18.bn1.running_mean', 'layer3.18.bn1.running_var', 'layer3.18.bn1.num_batches_tracked',
#  'layer3.18.conv2.0.weight', 'layer3.18.conv2.1.ChannelGate.mlp.1.weight', 'layer3.18.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.18.conv2.1.ChannelGate.mlp.3.weight', 'layer3.18.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.18.conv2.1.CoordAtt.conv1.weight', 'layer3.18.conv2.1.CoordAtt.bn1.weight',
#  'layer3.18.conv2.1.CoordAtt.bn1.bias', 'layer3.18.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.18.conv2.1.CoordAtt.bn1.running_var', 'layer3.18.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.18.conv2.1.CoordAtt.conv_h.weight', 'layer3.18.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.18.conv2.1.CoordAtt.conv_w.weight', 'layer3.18.conv2.1.CoordAtt.conv_w.bias', 'layer3.18.bn2.weight',
#  'layer3.18.bn2.bias', 'layer3.18.bn2.running_mean', 'layer3.18.bn2.running_var', 'layer3.18.bn2.num_batches_tracked',
#  'layer3.18.conv3.weight', 'layer3.18.bn3.weight', 'layer3.18.bn3.bias', 'layer3.18.bn3.running_mean',
#  'layer3.18.bn3.running_var', 'layer3.18.bn3.num_batches_tracked', 'layer3.19.conv1.weight', 'layer3.19.bn1.weight',
#  'layer3.19.bn1.bias', 'layer3.19.bn1.running_mean', 'layer3.19.bn1.running_var', 'layer3.19.bn1.num_batches_tracked',
#  'layer3.19.conv2.0.weight', 'layer3.19.conv2.1.ChannelGate.mlp.1.weight', 'layer3.19.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.19.conv2.1.ChannelGate.mlp.3.weight', 'layer3.19.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.19.conv2.1.CoordAtt.conv1.weight', 'layer3.19.conv2.1.CoordAtt.bn1.weight',
#  'layer3.19.conv2.1.CoordAtt.bn1.bias', 'layer3.19.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.19.conv2.1.CoordAtt.bn1.running_var', 'layer3.19.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.19.conv2.1.CoordAtt.conv_h.weight', 'layer3.19.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.19.conv2.1.CoordAtt.conv_w.weight', 'layer3.19.conv2.1.CoordAtt.conv_w.bias', 'layer3.19.bn2.weight',
#  'layer3.19.bn2.bias', 'layer3.19.bn2.running_mean', 'layer3.19.bn2.running_var', 'layer3.19.bn2.num_batches_tracked',
#  'layer3.19.conv3.weight', 'layer3.19.bn3.weight', 'layer3.19.bn3.bias', 'layer3.19.bn3.running_mean',
#  'layer3.19.bn3.running_var', 'layer3.19.bn3.num_batches_tracked', 'layer3.20.conv1.weight', 'layer3.20.bn1.weight',
#  'layer3.20.bn1.bias', 'layer3.20.bn1.running_mean', 'layer3.20.bn1.running_var', 'layer3.20.bn1.num_batches_tracked',
#  'layer3.20.conv2.0.weight', 'layer3.20.conv2.1.ChannelGate.mlp.1.weight', 'layer3.20.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.20.conv2.1.ChannelGate.mlp.3.weight', 'layer3.20.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.20.conv2.1.CoordAtt.conv1.weight', 'layer3.20.conv2.1.CoordAtt.bn1.weight',
#  'layer3.20.conv2.1.CoordAtt.bn1.bias', 'layer3.20.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.20.conv2.1.CoordAtt.bn1.running_var', 'layer3.20.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.20.conv2.1.CoordAtt.conv_h.weight', 'layer3.20.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.20.conv2.1.CoordAtt.conv_w.weight', 'layer3.20.conv2.1.CoordAtt.conv_w.bias', 'layer3.20.bn2.weight',
#  'layer3.20.bn2.bias', 'layer3.20.bn2.running_mean', 'layer3.20.bn2.running_var', 'layer3.20.bn2.num_batches_tracked',
#  'layer3.20.conv3.weight', 'layer3.20.bn3.weight', 'layer3.20.bn3.bias', 'layer3.20.bn3.running_mean',
#  'layer3.20.bn3.running_var', 'layer3.20.bn3.num_batches_tracked', 'layer3.21.conv1.weight', 'layer3.21.bn1.weight',
#  'layer3.21.bn1.bias', 'layer3.21.bn1.running_mean', 'layer3.21.bn1.running_var', 'layer3.21.bn1.num_batches_tracked',
#  'layer3.21.conv2.0.weight', 'layer3.21.conv2.1.ChannelGate.mlp.1.weight', 'layer3.21.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.21.conv2.1.ChannelGate.mlp.3.weight', 'layer3.21.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.21.conv2.1.CoordAtt.conv1.weight', 'layer3.21.conv2.1.CoordAtt.bn1.weight',
#  'layer3.21.conv2.1.CoordAtt.bn1.bias', 'layer3.21.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.21.conv2.1.CoordAtt.bn1.running_var', 'layer3.21.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.21.conv2.1.CoordAtt.conv_h.weight', 'layer3.21.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.21.conv2.1.CoordAtt.conv_w.weight', 'layer3.21.conv2.1.CoordAtt.conv_w.bias', 'layer3.21.bn2.weight',
#  'layer3.21.bn2.bias', 'layer3.21.bn2.running_mean', 'layer3.21.bn2.running_var', 'layer3.21.bn2.num_batches_tracked',
#  'layer3.21.conv3.weight', 'layer3.21.bn3.weight', 'layer3.21.bn3.bias', 'layer3.21.bn3.running_mean',
#  'layer3.21.bn3.running_var', 'layer3.21.bn3.num_batches_tracked', 'layer3.22.conv1.weight', 'layer3.22.bn1.weight',
#  'layer3.22.bn1.bias', 'layer3.22.bn1.running_mean', 'layer3.22.bn1.running_var', 'layer3.22.bn1.num_batches_tracked',
#  'layer3.22.conv2.0.weight', 'layer3.22.conv2.1.ChannelGate.mlp.1.weight', 'layer3.22.conv2.1.ChannelGate.mlp.1.bias',
#  'layer3.22.conv2.1.ChannelGate.mlp.3.weight', 'layer3.22.conv2.1.ChannelGate.mlp.3.bias',
#  'layer3.22.conv2.1.CoordAtt.conv1.weight', 'layer3.22.conv2.1.CoordAtt.bn1.weight',
#  'layer3.22.conv2.1.CoordAtt.bn1.bias', 'layer3.22.conv2.1.CoordAtt.bn1.running_mean',
#  'layer3.22.conv2.1.CoordAtt.bn1.running_var', 'layer3.22.conv2.1.CoordAtt.bn1.num_batches_tracked',
#  'layer3.22.conv2.1.CoordAtt.conv_h.weight', 'layer3.22.conv2.1.CoordAtt.conv_h.bias',
#  'layer3.22.conv2.1.CoordAtt.conv_w.weight', 'layer3.22.conv2.1.CoordAtt.conv_w.bias', 'layer3.22.bn2.weight',
#  'layer3.22.bn2.bias', 'layer3.22.bn2.running_mean', 'layer3.22.bn2.running_var', 'layer3.22.bn2.num_batches_tracked',
#  'layer3.22.conv3.weight', 'layer3.22.bn3.weight', 'layer3.22.bn3.bias', 'layer3.22.bn3.running_mean',
#  'layer3.22.bn3.running_var', 'layer3.22.bn3.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight',
#  'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked',
#  'layer4.0.conv2.0.relative_h', 'layer4.0.conv2.0.relative_w', 'layer4.0.conv2.0.query.weight',
#  'layer4.0.conv2.0.query.bias', 'layer4.0.conv2.0.key.weight', 'layer4.0.conv2.0.key.bias',
#  'layer4.0.conv2.0.value.weight', 'layer4.0.conv2.0.value.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
#  'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight',
#  'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var',
#  'layer4.0.bn3.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight',
#  'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var',
#  'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias',
#  'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked',
#  'layer4.1.conv2.0.relative_h', 'layer4.1.conv2.0.relative_w', 'layer4.1.conv2.0.query.weight',
#  'layer4.1.conv2.0.query.bias', 'layer4.1.conv2.0.key.weight', 'layer4.1.conv2.0.key.bias',
#  'layer4.1.conv2.0.value.weight', 'layer4.1.conv2.0.value.bias', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
#  'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight',
#  'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var',
#  'layer4.1.bn3.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias',
#  'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked',
#  'layer4.2.conv2.0.relative_h', 'layer4.2.conv2.0.relative_w', 'layer4.2.conv2.0.query.weight',
#  'layer4.2.conv2.0.query.bias', 'layer4.2.conv2.0.key.weight', 'layer4.2.conv2.0.key.bias',
#  'layer4.2.conv2.0.value.weight', 'layer4.2.conv2.0.value.bias', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias',
#  'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight',
#  'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var',
#  'layer4.2.bn3.num_batches_tracked', 'layer4.3.conv1.weight', 'layer4.3.bn1.weight', 'layer4.3.bn1.bias',
#  'layer4.3.bn1.running_mean', 'layer4.3.bn1.running_var', 'layer4.3.bn1.num_batches_tracked',
#  'layer4.3.conv2.0.relative_h', 'layer4.3.conv2.0.relative_w', 'layer4.3.conv2.0.query.weight',
#  'layer4.3.conv2.0.query.bias', 'layer4.3.conv2.0.key.weight', 'layer4.3.conv2.0.key.bias',
#  'layer4.3.conv2.0.value.weight', 'layer4.3.conv2.0.value.bias', 'layer4.3.bn2.weight', 'layer4.3.bn2.bias',
#  'layer4.3.bn2.running_mean', 'layer4.3.bn2.running_var', 'layer4.3.bn2.num_batches_tracked', 'layer4.3.conv3.weight',
#  'layer4.3.bn3.weight', 'layer4.3.bn3.bias', 'layer4.3.bn3.running_mean', 'layer4.3.bn3.running_var',
#  'layer4.3.bn3.num_batches_tracked', 'layer4.4.conv1.weight', 'layer4.4.bn1.weight', 'layer4.4.bn1.bias',
#  'layer4.4.bn1.running_mean', 'layer4.4.bn1.running_var', 'layer4.4.bn1.num_batches_tracked',
#  'layer4.4.conv2.0.relative_h', 'layer4.4.conv2.0.relative_w', 'layer4.4.conv2.0.query.weight',
#  'layer4.4.conv2.0.query.bias', 'layer4.4.conv2.0.key.weight', 'layer4.4.conv2.0.key.bias',
#  'layer4.4.conv2.0.value.weight', 'layer4.4.conv2.0.value.bias', 'layer4.4.bn2.weight', 'layer4.4.bn2.bias',
#  'layer4.4.bn2.running_mean', 'layer4.4.bn2.running_var', 'layer4.4.bn2.num_batches_tracked', 'layer4.4.conv3.weight',
#  'layer4.4.bn3.weight', 'layer4.4.bn3.bias', 'layer4.4.bn3.running_mean', 'layer4.4.bn3.running_var',
#  'layer4.4.bn3.num_batches_tracked', 'layer4.5.conv1.weight', 'layer4.5.bn1.weight', 'layer4.5.bn1.bias',
#  'layer4.5.bn1.running_mean', 'layer4.5.bn1.running_var', 'layer4.5.bn1.num_batches_tracked',
#  'layer4.5.conv2.0.relative_h', 'layer4.5.conv2.0.relative_w', 'layer4.5.conv2.0.query.weight',
#  'layer4.5.conv2.0.query.bias', 'layer4.5.conv2.0.key.weight', 'layer4.5.conv2.0.key.bias',
#  'layer4.5.conv2.0.value.weight', 'layer4.5.conv2.0.value.bias', 'layer4.5.bn2.weight', 'layer4.5.bn2.bias',
#  'layer4.5.bn2.running_mean', 'layer4.5.bn2.running_var', 'layer4.5.bn2.num_batches_tracked', 'layer4.5.conv3.weight',
#  'layer4.5.bn3.weight', 'layer4.5.bn3.bias', 'layer4.5.bn3.running_mean', 'layer4.5.bn3.running_var',
#  'layer4.5.bn3.num_batches_tracked', 'layer4.6.conv1.weight', 'layer4.6.bn1.weight', 'layer4.6.bn1.bias',
#  'layer4.6.bn1.running_mean', 'layer4.6.bn1.running_var', 'layer4.6.bn1.num_batches_tracked',
#  'layer4.6.conv2.0.relative_h', 'layer4.6.conv2.0.relative_w', 'layer4.6.conv2.0.query.weight',
#  'layer4.6.conv2.0.query.bias', 'layer4.6.conv2.0.key.weight', 'layer4.6.conv2.0.key.bias',
#  'layer4.6.conv2.0.value.weight', 'layer4.6.conv2.0.value.bias', 'layer4.6.bn2.weight', 'layer4.6.bn2.bias',
#  'layer4.6.bn2.running_mean', 'layer4.6.bn2.running_var', 'layer4.6.bn2.num_batches_tracked', 'layer4.6.conv3.weight',
#  'layer4.6.bn3.weight', 'layer4.6.bn3.bias', 'layer4.6.bn3.running_mean', 'layer4.6.bn3.running_var',
#  'layer4.6.bn3.num_batches_tracked', 'layer4.7.conv1.weight', 'layer4.7.bn1.weight', 'layer4.7.bn1.bias',
#  'layer4.7.bn1.running_mean', 'layer4.7.bn1.running_var', 'layer4.7.bn1.num_batches_tracked',
#  'layer4.7.conv2.0.relative_h', 'layer4.7.conv2.0.relative_w', 'layer4.7.conv2.0.query.weight',
#  'layer4.7.conv2.0.query.bias', 'layer4.7.conv2.0.key.weight', 'layer4.7.conv2.0.key.bias',
#  'layer4.7.conv2.0.value.weight', 'layer4.7.conv2.0.value.bias', 'layer4.7.bn2.weight', 'layer4.7.bn2.bias',
#  'layer4.7.bn2.running_mean', 'layer4.7.bn2.running_var', 'layer4.7.bn2.num_batches_tracked', 'layer4.7.conv3.weight',
#  'layer4.7.bn3.weight', 'layer4.7.bn3.bias', 'layer4.7.bn3.running_mean', 'layer4.7.bn3.running_var',
#  'layer4.7.bn3.num_batches_tracked', 'layer4.8.conv1.weight', 'layer4.8.bn1.weight', 'layer4.8.bn1.bias',
#  'layer4.8.bn1.running_mean', 'layer4.8.bn1.running_var', 'layer4.8.bn1.num_batches_tracked',
#  'layer4.8.conv2.0.relative_h', 'layer4.8.conv2.0.relative_w', 'layer4.8.conv2.0.query.weight',
#  'layer4.8.conv2.0.query.bias', 'layer4.8.conv2.0.key.weight', 'layer4.8.conv2.0.key.bias',
#  'layer4.8.conv2.0.value.weight', 'layer4.8.conv2.0.value.bias', 'layer4.8.bn2.weight', 'layer4.8.bn2.bias',
#  'layer4.8.bn2.running_mean', 'layer4.8.bn2.running_var', 'layer4.8.bn2.num_batches_tracked', 'layer4.8.conv3.weight',
#  'layer4.8.bn3.weight', 'layer4.8.bn3.bias', 'layer4.8.bn3.running_mean', 'layer4.8.bn3.running_var',
#  'layer4.8.bn3.num_batches_tracked', 'layer4.9.conv1.weight', 'layer4.9.bn1.weight', 'layer4.9.bn1.bias',
#  'layer4.9.bn1.running_mean', 'layer4.9.bn1.running_var', 'layer4.9.bn1.num_batches_tracked',
#  'layer4.9.conv2.0.relative_h', 'layer4.9.conv2.0.relative_w', 'layer4.9.conv2.0.query.weight',
#  'layer4.9.conv2.0.query.bias', 'layer4.9.conv2.0.key.weight', 'layer4.9.conv2.0.key.bias',
#  'layer4.9.conv2.0.value.weight', 'layer4.9.conv2.0.value.bias', 'layer4.9.bn2.weight', 'layer4.9.bn2.bias',
#  'layer4.9.bn2.running_mean', 'layer4.9.bn2.running_var', 'layer4.9.bn2.num_batches_tracked', 'layer4.9.conv3.weight',
#  'layer4.9.bn3.weight', 'layer4.9.bn3.bias', 'layer4.9.bn3.running_mean', 'layer4.9.bn3.running_var',
#  'layer4.9.bn3.num_batches_tracked', 'layer4.10.conv1.weight', 'layer4.10.bn1.weight', 'layer4.10.bn1.bias',
#  'layer4.10.bn1.running_mean', 'layer4.10.bn1.running_var', 'layer4.10.bn1.num_batches_tracked',
#  'layer4.10.conv2.0.relative_h', 'layer4.10.conv2.0.relative_w', 'layer4.10.conv2.0.query.weight',
#  'layer4.10.conv2.0.query.bias', 'layer4.10.conv2.0.key.weight', 'layer4.10.conv2.0.key.bias',
#  'layer4.10.conv2.0.value.weight', 'layer4.10.conv2.0.value.bias', 'layer4.10.bn2.weight', 'layer4.10.bn2.bias',
#  'layer4.10.bn2.running_mean', 'layer4.10.bn2.running_var', 'layer4.10.bn2.num_batches_tracked',
#  'layer4.10.conv3.weight', 'layer4.10.bn3.weight', 'layer4.10.bn3.bias', 'layer4.10.bn3.running_mean',
#  'layer4.10.bn3.running_var', 'layer4.10.bn3.num_batches_tracked', 'layer4.11.conv1.weight', 'layer4.11.bn1.weight',
#  'layer4.11.bn1.bias', 'layer4.11.bn1.running_mean', 'layer4.11.bn1.running_var', 'layer4.11.bn1.num_batches_tracked',
#  'layer4.11.conv2.0.relative_h', 'layer4.11.conv2.0.relative_w', 'layer4.11.conv2.0.query.weight',
#  'layer4.11.conv2.0.query.bias', 'layer4.11.conv2.0.key.weight', 'layer4.11.conv2.0.key.bias',
#  'layer4.11.conv2.0.value.weight', 'layer4.11.conv2.0.value.bias', 'layer4.11.bn2.weight', 'layer4.11.bn2.bias',
#  'layer4.11.bn2.running_mean', 'layer4.11.bn2.running_var', 'layer4.11.bn2.num_batches_tracked',
#  'layer4.11.conv3.weight', 'layer4.11.bn3.weight', 'layer4.11.bn3.bias', 'layer4.11.bn3.running_mean',
#  'layer4.11.bn3.running_var', 'layer4.11.bn3.num_batches_tracked', 'fc.1.weight', 'fc.1.bias']


