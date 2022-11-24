import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import os
from skimage import segmentation
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.morphology import disk
from scipy import ndimage as ndi
import torchvision
from torchvision import transforms
# import pandas as pd
from scipy.fft import irfft
from preprocess_one_img import preprocess_radiograph

NUM_COMPLEX = 257
SMOOTHING = 180
ANGLE_IGNORE_FACTOR = 0.85
FIRST_DER_TOLERANCE = 0.001
NUM_POINTS_TO_AVG = 5
# used to filter some undesired inflection points
MIN_INFLECTION_X = 200
MAX_INFLECTION_X = 800
BOTTOM_SLOPE_THRES = 0.1

AHE_RADIOGRAPH_PATH = "app/sunhl-1th-09-Jan-2017-218 C AP.jpg"
UNET_MODEL_PATH = "app/unet.pth.tar"
SPECTRUM_MODEL_PATH = "app/latent_space_spectrum.pth.tar"
ANGLE_MODEL_PATH = "app/spectrum_angle.pth.tar"
DEVICE = "cpu"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class _Identity(nn.Module):
    def __init__(self):
        super(_Identity, self).__init__()

    def forward(self, x):
        return x


# tl model modified from vgg 16
class VGG16FE(nn.Module):
    def __init__(self):
        super().__init__()

        self.tl_model = torchvision.models.vgg16(pretrained=True)
        for param in self.tl_model.parameters():
            param.requires_grad = False
        self.tl_model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.tl_model.classifier = nn.Flatten()
        # print(self.tl_model)

    def forward(self, x):
        ret = self.tl_model(x)
        return ret


class LatentSpaceToSpectrumModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcn = nn.Sequential(nn.Linear(25088, 8000),
                                 nn.BatchNorm1d(8000),
                                 nn.ReLU(),
                                 nn.Linear(8000, 2000),

                                 nn.ReLU(),
                                 nn.Linear(2000, 600),

                                 nn.ReLU(),
                                 nn.Linear(600, 258),
                                 )
        # print(self.fcn)

    def forward(self, x):
        return self.fcn(x)


class Spectrum2AnglesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fcn = nn.Sequential(nn.Linear(258, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 40),
                                 nn.ReLU(),
                                 nn.Linear(40, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 3),
                                 )

        # print(self.fcn)

    def forward(self, x):
        return self.fcn(x)


def generate_spectrum(latent_np):
    model = LatentSpaceToSpectrumModel()
    model.load_state_dict(
        torch.load(SPECTRUM_MODEL_PATH, map_location=torch.device('cpu'))["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    data = latent_np.astype(np.float32)
    data = torch.from_numpy(data.reshape(1, len(data))).to(DEVICE)

    with torch.no_grad():
        pred_spectrum = model(data)
        pred_spectrum = pred_spectrum.cpu().numpy()
        #pred_spectrum = pred_spectrum.reshape(1, pred_spectrum.shape[1])

        # data_target_df = pd.DataFrame({"spectrum": pred_spectrum[0, :].tolist(), "angles": gt_angles_pad[0, :].tolist()})
        # data_target_df.to_csv(os.path.join(OUTPUT_PREDICTED_SPECTRA_ANGLES_DIR, latent_space_csvs[i]))
    return pred_spectrum


def generate_latent(ahe_radiograph_np, mask_np):
    model = VGG16FE().to(DEVICE)

    ahe_image_numpy = cv2. GaussianBlur(
        ahe_radiograph_np, (11, 11), cv2.BORDER_DEFAULT)

    ahe_image_three_channels = cv2.cvtColor(
        ahe_image_numpy, cv2.COLOR_GRAY2RGB).astype(np.float32)
    ahe_image_three_channels = np.transpose(
        ahe_image_three_channels, (2, 0, 1))
    input_nparray_three_channels = ahe_image_three_channels / 255

    normalized_gradient_sobel = cal_gradient_sobel(
        ahe_image_numpy).astype(np.float32)
    #normalized_watershed = watershed_wrt_gradient(ahe_image_numpy).astype(np.float32)

    segment_img_numpy = mask_np / 255  # read grayscale
    resized_shape = (
        segment_img_numpy.shape[1] * 4, segment_img_numpy.shape[0] * 4)
    segment_img_resized = cv2.resize(
        segment_img_numpy, resized_shape, interpolation=cv2.INTER_AREA)

    #print(segment_img_resized.shape, input_nparray_three_channels.shape)

    input_nparray_three_channels[1, :, :] = normalized_gradient_sobel
    input_nparray_three_channels[2, :, :] = segment_img_resized

    input_tensor = torch.from_numpy(
        input_nparray_three_channels).unsqueeze(0).to(DEVICE)
    # print(input_tensor.shape)
    with torch.no_grad():
        latent_space_features = model(input_tensor)
        latent_space_features = latent_space_features.cpu().numpy()
        latent_space_features = latent_space_features[0, :]
        # print(latent_space_features.shape)

        # data_target_df = pd.DataFrame(latent_space_features.tolist())

        # data_target_df.to_csv(os.path.join(OUTPUT_CSV_DIR, spectra_csv[i]))
        return latent_space_features


def watershed_wrt_gradient(input_img_nparray):
    markers = rank.gradient(input_img_nparray, disk(5)) < 10
    markers, _ = ndi.label(markers)

    gradient = rank.gradient(input_img_nparray, disk(2))
    seg_img = watershed(gradient, markers)
    seg_img = seg_img / np.max(seg_img)
    return seg_img


def gammaCorrection(src, gamma):
    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


def cal_gradient_sobel(input_img_nparray):

    image_grad = np.gradient(input_img_nparray)
    gradient_intensity = np.hypot(
        image_grad[0], image_grad[1]).astype(np.uint8)

    gX = cv2.Sobel(input_img_nparray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(input_img_nparray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    sobel_xy = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    combined = gradient_intensity + sobel_xy
    combined = combined / np.max(combined) * 255
    combined_gamma_corrected = gammaCorrection(combined.astype(np.uint8), 1.25)

    return combined_gamma_corrected / 255


def generate_predicted_curve(spectrum_np, radiograph_rgb_pil):
    #print(spectrum_np.shape)
    x_unfiltered, y_unfiltered = get_t_domain_curve(spectrum_np, 129)
    x_plot, y_plot = filter_predicted_xy(x_unfiltered, y_unfiltered)
    for (y, x) in zip(y_plot, x_plot):
        if y > 0 and y < 511 and x > 0 and x < 1023:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    radiograph_rgb_pil.putpixel((y+i, x+j), (255, 0, 0))
    return radiograph_rgb_pil


def get_t_domain_curve(pred_complex, freq_num_complex):
    pred_real_yf = pred_complex[0, :freq_num_complex]
    pred_im_yf = pred_complex[0, freq_num_complex:]
    pred_yf = []
    for i in range(len(pred_real_yf)):
        pred_yf.append(pred_real_yf[i] + pred_im_yf[i] * 1j)
    pred_yf = np.array(pred_yf)
    y_t = irfft(pred_yf).astype(np.int16)
    x_t = np.arange(0, 1024, 8)
    return x_t, y_t


def filter_predicted_xy(full_x, full_y, threshold_slope=1.2):
    disregard_start_idx = -1
    for i in range(1, len(full_x)):
        del_x = full_x[i] - full_x[i - 1]
        del_y = full_y[i] - full_y[i - 1]
        slope = del_y / del_x
        if abs(slope) > threshold_slope:
            disregard_start_idx = i
            break
    if disregard_start_idx != -1:
        return full_x[:disregard_start_idx], full_y[:disregard_start_idx]
    else:
        return full_x, full_y


def calculate_angles_from_spectrum(spectrum_np):
    model = Spectrum2AnglesModel()
    checkpoint = torch.load(ANGLE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    
   
    model = model.to(DEVICE)
    model.eval()

    freq_spectrum = spectrum_np.reshape((1, -1)).astype("float32")
    input_tensor = torch.from_numpy(freq_spectrum).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = model(input_tensor)
    predictions = predictions[0, 0, :].cpu().numpy()
    return predictions

def generate_segmentation(radiograph_pil):
    model = UNET(in_channels=1, out_channels=1,
                 features=[4, 8, 16, 32]).to(DEVICE)
    checkpoint = torch.load(UNET_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_pil = radiograph_pil
    width, height = image_pil.size
    image = np.array(image_pil.resize(
        (int(width / 4), int(height / 4))), dtype=np.float32) / 256
    image = np.transpose(image, (1, 0))
    image = image.reshape(
        (1, 1, image.shape[0], image.shape[1]))  # 1, 1, w h
    input_tensor = torch.from_numpy(image).to(DEVICE)
    with torch.no_grad():
        preds = model(input_tensor)
        preds = (preds > 0.5).float() * 255
        preds_np = preds.squeeze(0).squeeze(0).numpy()
        preds_np = np.transpose(preds_np, (1, 0))
        # seg_img_pil = Image.fromarray(preds_np).convert("RGB")

        # # save image
        # seg_img_pil.save(AHE_RADIOGRAPH_PATH.replace(
        #     "_processed.png", "_seg.png"))

        return preds_np

def generate_marked_img(ahe_radiograph_np, ahe_radiograph_rgb_pil, segmentation_result_np):
    mask_np = segmentation_result_np
    ahe_np = ahe_radiograph_np
    ahe_rgb_pil = ahe_radiograph_rgb_pil
    latent = generate_latent(ahe_np, mask_np)
    spectrum_np = generate_spectrum(latent)
    new_img = generate_predicted_curve(spectrum_np, ahe_rgb_pil)
    return spectrum_np, new_img


def pipline(AHE_RADIOGRAPH_PATH):
    radiograph = Image.open(AHE_RADIOGRAPH_PATH).convert("L")
    radiograph_pil = preprocess_radiograph(radiograph)
    radiograph_rgb_pil = radiograph_pil.convert("RGB")
    segmentation_np = generate_segmentation(radiograph_pil)
    radiograph_np = np.array(radiograph_pil)
    spectrum_np, new_img = generate_marked_img(radiograph_np, radiograph_rgb_pil, segmentation_np)
    new_img.save(AHE_RADIOGRAPH_PATH.replace(".jpg", ".png"))
    return calculate_angles_from_spectrum(spectrum_np)

if __name__ == "__main__":
    pipline(AHE_RADIOGRAPH_PATH)