


import torch,timm


class Model(torch.nn.Module):
    def __init__(self,num_classes=10) -> None:
        super(Model,self).__init__()
        self.feature_extrector=timm.create_model('resnest50d', num_classes=0 ,pretrained=True)
        self.linear=torch.nn.Linear(2048,num_classes)
        if torch.cuda.is_available():
            self.feature_extrector=self.feature_extrector.cuda()
            self.linear=self.linear.cuda()
        
    
    def forward(self,x):
        batch_size=x.size(0)
        x=self.feature_extrector(x)
        x=x.view(batch_size,-1)
        x = self.linear(x)
        return x
    
if __name__=='__main__':
    net=Model(10)
    print(net)
    x=torch.randn(2,3,256,256)
    if torch.cuda.is_available():
        x=x.cuda()
    print(net(x).size())
