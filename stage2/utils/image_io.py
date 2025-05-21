import glob
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
matplotlib.use('agg')

def prepare_hazy_image(file_name):
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)

def prepare_gt_img(file_name, SOTS=True):
    if SOTS:
        img_pil = crop_image(crop_a_image(get_image(file_name, -1)[0], d=10), d=32)
    else:
        img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)

def crop_a_image(img, d=10):
    bbox = [
        int((d)),
        int((d)),
        int((img.size[0] - d)),
        int((img.size[1] - d)),
    ]
    img_cropped = img.crop(bbox)
    return img_cropped

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def crop_image(img, d=32):
    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)
    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]
    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))

def crop_torch_image(img, d=32):
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)
    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
    return params


def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    assert len(images_np) == 2
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def save_image_np(name, image_np, output_path="output/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def save_image_tensor(image_tensor, output_path="output/"):
    image_np = torch_to_np(image_tensor)
    p = np_to_pil(image_np)
    #print(p)
    p.save(output_path)


def video_to_images(file_name, name):
    video = prepare_video(file_name)
    for i, f in enumerate(video):
        save_image(name + "_{0:03d}".format(i), f)


def images_to_video(images_dir, name, gray=True):
    num = len(glob.glob(images_dir + "/*.jpg"))
    c = []
    for i in range(num):
        if gray:
            img = prepare_gray_image(images_dir + "/" + name + "_{}.jpg".format(i))
        else:
            img = prepare_image(images_dir + "/" + name + "_{}.jpg".format(i))
        print(img.shape)
        c.append(img)
    save_video(name, np.array(c))


def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path="output/"):
    plt.clf()
    plt.plot(graph_list)
    plt.savefig(output_path + name + ".png")


def create_augmentations(np_image):
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(),
            np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def create_video_augmentations(np_video):
    aug = [np_video.copy(), np.rot90(np_video, 1, (2, 3)).copy(),
           np.rot90(np_video, 2, (2, 3)).copy(), np.rot90(np_video, 3, (2, 3)).copy()]
    flipped = np_video[:, :, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (2, 3)).copy(), np.rot90(flipped, 2, (2, 3)).copy(),
            np.rot90(flipped, 3, (2, 3)).copy()]
    return aug


def save_graphs(name, graph_dict, output_path="output/"):
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
    ax.set_xlabel('iterations')
    ax.set_ylabel('MSE-loss')
    plt.legend()
    plt.savefig(output_path + name + ".png")


def load(path):
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    img = load(path)
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_gt(file_name):
    img = get_image(file_name, -1)
    img_pil = img[0].crop([10, 10, img[0].size[0] - 10, img[0].size[1] - 10])
    img_pil = crop_image(img_pil, d=32)

    return pil_to_np(img_pil)


def prepare_image(file_name):
    img = get_image(file_name, -1)

    img_pil = crop_image(img[0], d=16)

    return pil_to_np(img_pil)


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])


def pil_to_np(img_PIL, with_transpose=True):
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l // 2]
    return result


def average(img_np_list):
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np):
    #print(img_np)
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    #print(img_np.shape[0])

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)
    #print(ar)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    #print(img_var.detach().cpu().numpy()[0].shape)
    return img_var.detach().cpu().numpy()[0]

def gray_to_rgb(gray, data_YCbCr2,batch_size = 1):

    if batch_size == 1:
        YCbCr0 = data_YCbCr2[:,:,:,0]
        YCbCr1 = data_YCbCr2[:,:,:,1]
        YCbCr2 = data_YCbCr2[:,:,:,2]
        #print(YCbCr1.cuda().squeeze().unsqueeze(dim = 2))
        rgbfuse = torch.cat([(gray * 255.0).int().squeeze().unsqueeze(dim = 2), YCbCr1.cuda().squeeze().unsqueeze(dim = 2), YCbCr2.cuda().squeeze().unsqueeze(dim = 2)], dim = 2)
        rgbp = np.squeeze((rgbfuse).cpu().numpy())
        #print(rgbfuse[:,:,0])
        #rgbp = rgbp.astype(np.uint8)
        rgbp = cv2.cvtColor(rgbp,cv2.COLOR_YCR_CB2RGB)
        rgbp = torch.Tensor((rgbp/255.0).transpose(2,0,1)).cuda()
        rgbp = rgbp.unsqueeze(dim=0)
    else:
        YCbCr0 = data_YCbCr2[:,0,:,:]
        YCbCr1 = data_YCbCr2[:,1,:,:]
        YCbCr2 = data_YCbCr2[:,2,:,:]
        rgbfuse = torch.cat([(gray), YCbCr1.cuda().unsqueeze(dim = 1), YCbCr2.cuda().unsqueeze(dim = 1)], dim = 1)   
        rgbp = np.squeeze((rgbfuse * 255.0).detach().cpu().numpy().astype(np.float32))
        rgbb = np.zeros((batch_size,np.size(rgbp,2),np.size(rgbp,3),np.size(rgbp,1))).astype(np.float32)
        for i in range(0,batch_size):
            rgbb[i] = rgbp[i].transpose(1, 2, 0)
            #print(rgbb[i].shape)
            rgbb[i] = cv2.cvtColor(rgbb[i],cv2.COLOR_YCR_CB2RGB)
            rgbp[i] = (rgbb[i]/255.0).transpose(2,0,1)
        rgbp = torch.Tensor(rgbp)
    return rgbp