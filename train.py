import os
from model import alexnet, AlexNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use


# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
os.makedirs (CHECKPOINT_DIR, exist_ok = True)

if __name__ == "__main__":
    #printing the initial seed value

    seed = torch.inital_seed()
    print("Used seed: {}".format(seed))

    tbwriter = SummaryWriter(log_dir = LOG_DIR)
    print("TensorboardX summary writer created")


    #model
    alexnet = AlexNet(num_classes = NUM_CLASSES).to(device)
    #printing the model
    print(alexnet)
    print("Model created")

    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    print("Dataset Created")

    dataloader = data.DataLoader(
        dataset,
        shuffle = True,
        pin_memory = True,
        num_workers = 8,
        batch_size = BATCH_SIZE
    )

    print('Dataloader Created')

    #optimizer
    optimizer = optim.SGD(
        params=alexnet.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY)

    print("optimizer created")

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print("Starting Training")
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            

            output = alexnet(imgs)

            loss = F.cross_entropy(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #logging the information and adding to tensorboard 
            if total_steps % 10 == 0:
                 with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
            
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

                

            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1;

        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)



            

    


    
    