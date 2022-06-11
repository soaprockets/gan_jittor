
import model_demo as models
import utils.utils as utils
import config
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
data_root='/home/hexufeng/Data/gan_jittor/competition/landscape_comp'
transforms =transforms.Compose([
    transform.Resize(size=(384, 512)),
    transform.ToTensor()
])
val_set=ImageDataset(root=data_root,mode='val',transforms=transforms)
dataloader_val=DataLoader(val_set,batch_size=4,shuffle=True,drop_last=True)
#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    _, label = models.preprocess_input(opt, data_i)
    generated = model(None, label, "generate", None)
    image_saver(label, generated, data_i["name"])
