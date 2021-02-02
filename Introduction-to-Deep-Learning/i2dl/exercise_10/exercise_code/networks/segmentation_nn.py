"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):
    hparams = {
    "learning_rate": 0.0001,
    "batch_size":512
    # TODO: if you have any model arguments/hparams, define them here
}
    def __init__(self, num_classes=23, hparams=None):#, logger=None):
        super().__init__()
        self.hparams = hparams
        #self.logger = logger
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        pretrained = models.mobilenet_v2(pretrained=True, progress=True).eval()
        for param in pretrained.parameters():
            param.requires_grad = False
            
        self.model = nn.Sequential(
          *(list(pretrained.children())[:-1]),
          nn.ConvTranspose2d(1280, 150,3,stride=2),
          nn.ConvTranspose2d(150, 64,3,stride=2),
          nn.ConvTranspose2d(64, 23,1),

          torch.nn.Upsample(scale_factor=float(240/35)),
        )
        pass
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.model(x)
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x
    
    def training_step(self, batch, batch_idx):
        
        inputs, targets = batch[0], batch[1]
        outputs = self.forward(inputs)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(outputs, targets)
        logs = { 'train_loss': loss }
        return {'loss': loss} #, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[0] , batch[1]
        outputs = self.forward(inputs)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        val_loss = loss_func(outputs, targets)
        logs = {'val_loss' : val_loss}
        return {'val_loss':val_loss} #, 'log':logs}
    
    def validation_epoch_end(self, outputs):

        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        return {'val_loss':avg_loss}#, 'log':logs}
 
    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        return optim

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=512)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
