import sys
import time
import os
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from torchvision import datasets, transforms
import dataset
from torch.utils.data import Dataset
import cv2

class MonopoleLoader(Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, root, transform=None, batch_size=1, shape=(0, 0)):
        self.root = root
        self.transform = transform

        tmp = []
        sorted_files = sorted(os.listdir(root))
        for f in sorted_files:
            if f.split('.')[-1] != "png":
                continue
            tmp += [f]
        sorted_files = tmp

        #self.files = [[sorted_files[i + j*batch_size] for i in range(batch_size)] for j in range(ceil(len(sorted_files)/batch_size) - 1)]
        #self.files += [[sorted_files[-i-1] for i in range(len(sorted_files)%batch_size)]]
        self.files = [sorted_files[i] for i in range(len(sorted_files))]
        self.batch_size = batch_size

        if not (shape[0] or shape[1]):
            raise ValueError("Shape must have positive width or height") 
        self.shape = shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = []
        #for f in self.files[index]:
        img = cv2.imread(os.path.join(self.root, self.files[index]))
        sized = cv2.resize(img, (self.shape[0], self.shape[1]))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        return torch.tensor(np.transpose(sized, (2,0,1))).float().div(255.0), self.files[index]


def dir_setup(imgfile):
    if os.getcwd() != "/home/ashwin/pytorch-yolo2":
        print("predict.py must be run from /home/ashwin/pytorch-yolo2\nExiting...")
        sys.exit(0)
        
    # make a folder to put our predictions in if it does not yet exist
    if not os.path.isdir("./predictions"):
        os.mkdir("./predictions")
    
    if os.path.isdir(imgfile):
        imgfolder = imgfile.strip("/").split('/')[-1]
        if not os.path.isdir(os.path.join("predictions", imgfolder)):
            os.mkdir(os.path.join("predictions", imgfolder))
    elif imgfile.split("/")[-1].split('.')[1] == 'txt':
        filename = imgfile.split("/")[-1].split('.')[0]
        if not os.path.isdir(os.path.join("predictions", filename)):
            os.mkdir(os.path.join("predictions", filename))

def detect_cv2(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/monopoles.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)

    dir_setup(imgfile)

    if os.path.isdir(imgfile):

        batch_size = 32
        num_workers = 1
        init_width        = m.width
        init_height       = m.height

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

        data_loader = torch.utils.data.DataLoader(
            MonopoleLoader(imgfile, 
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]),
            batch_size=batch_size, shape=(init_width, init_height)),
            batch_size=batch_size, **kwargs)


        imgfolder = imgfile.strip("/").split('/')[-1]
        with open(os.path.join('predictions', imgfolder, '{}_list.txt'.format(imgfolder)), 'w') as out:

            for data, files in data_loader:
                if len(data.shape) == 3:
                    data = data.view(1, *data.shape)
                
                for i in range(2):
                    start = time.time()
                    boxes = do_detect(m, data, 0.5, 0.4, use_cuda)
                    finish = time.time()
                    if i == 0:
                        print('%i images predicted in %f seconds.' % (len(files), (finish-start)))
                for bx, f in zip(boxes, files):
                    if bx:
                        out.write(os.path.join(imgfile, f) + '\n')
                        imgname = f.split("/")[-1].split('.')[0]
                        plot_boxes_cv2(cv2.imread(os.path.join(imgfile, f)), bx, savename=os.path.join('predictions', imgfolder, '{}.jpg'.format(imgname)), class_names=class_names)



def detect(cfgfile, weightfile, imgfile):

    dir_setup(imgfile)

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/monopoles.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)

    if os.path.isfile(imgfile):
        filehandle = imgfile.split("/")[-1].split('.')[1]
        if filehandle == 'txt':
            filename = imgfile.split("/")[-1].split('.')[0]
            with open(imgfile, 'r') as infile:
                with open(os.path.join('predictions', filename, '{}_list.txt'.format(filename)), 'w') as f:
                    for files in infile:
                        files = files.strip('\n')
                        imgname = files.split("/")[-1].split('.')[0]
                        img = Image.open(files).convert('RGB')

                        sized = img.resize((m.width, m.height))
                        
                        for i in range(2):
                            start = time.time()
                            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                            finish = time.time()
                            if i == 1:
                                print('%s: Predicted in %f seconds.' % (files, (finish-start)))
                        
                        if boxes:
                            f.write(files + '\n')
                        plot_boxes(img, boxes, os.path.join('predictions', filename, '{}.jpg'.format(imgname)), class_names)
        else:
            imgname = imgfile.split("/")[-1].split('.')[0]
            img = Image.open(imgfile).convert('RGB')
            sized = img.resize((m.width, m.height))
            
            for i in range(2):
                start = time.time()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

            plot_boxes(img, boxes, os.path.join('predictions', '{}.jpg'.format(imgname)), class_names)
    elif os.path.isdir(imgfile):
        imgfolder = imgfile.strip("/").split('/')[-1]
        with open(os.path.join('predictions', imgfolder, '{}_list.txt'.format(imgfolder)), 'w') as f:
            for files in os.listdir(imgfile):
                if os.path.isdir(os.path.join(imgfile, files)):
                    continue
                imgname = files.split("/")[-1].split('.')[0]
                img = Image.open(os.path.join(imgfile, files)).convert('RGB')

                print(type(img))
                sys.exit(0)

                sized = img.resize((m.width, m.height))
                
                for i in range(2):
                    start = time.time()
                    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                    finish = time.time()
                    if i == 1:
                        print('%s: Predicted in %f seconds.' % (files, (finish-start)))
                if boxes:
                    f.write(os.path.join(imgfile, files) + '\n')
                plot_boxes(img, boxes, os.path.join('predictions', imgfolder, '{}.jpg'.format(imgname)), class_names)



def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/monopoles.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        # detect(cfgfile, weightfile, imgfile)
        detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python3 detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
