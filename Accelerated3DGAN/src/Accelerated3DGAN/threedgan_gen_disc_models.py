import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Generator(nn.Module):
    def __init__(self, latent_dim): #img_shape
        super().__init__()
        #self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.l1 = nn.Linear(self.latent_dim, 5184)
        self.up1 = nn.Upsample(scale_factor=(6, 6, 6), mode='trilinear', align_corners=False)
        self.conv1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(6, 6, 8), padding=0)
        nn.init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm3d(num_features=8, eps=1e-6) #num_features is the number of channels (see doc)
        self.pad1 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm3d(num_features=6, eps=1e-6)

        self.pad2 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)
        self.conv3 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad3 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)
        self.conv4 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad4 = nn.ConstantPad3d((0, 0, 1, 1, 1, 1), 0)
        self.conv5 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(3, 3, 5), padding=0)
        nn.init.kaiming_uniform_(self.conv5.weight)
        self.bn5 = nn.BatchNorm3d(num_features=6, eps=1e-6)

        self.pad5 = nn.ConstantPad3d((0, 0, 1, 1, 1, 1), 0)
        self.conv6 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(3, 3, 3), padding=0)
        nn.init.kaiming_uniform_(self.conv6.weight)
        self.conv7 = nn.Conv3d(in_channels=6, out_channels=1, kernel_size=(2, 2, 2), padding=0)
        nn.init.xavier_normal_(self.conv7.weight)


    def forward(self, z):
        img = self.l1(z)
        img = img.view(-1, 8, 9, 9, 8)
        img = self.up1(img)
        img = self.conv1(img)
        img = F.relu(img)
        img = self.bn1(img)
        img = self.pad1(img)
        img = self.conv2(img)
        img = F.relu(img)
        img = self.bn2(img)

        img = self.pad2(img)
        img = self.conv3(img)
        img = F.relu(img)
        img = self.bn3(img)
        img = self.pad3(img)
        img = self.conv4(img)
        img = F.relu(img)
        img = self.bn4(img)
        img = self.pad4(img)
        img = self.conv5(img)
        img = F.relu(img)
        img = self.bn5(img)

        img = self.pad5(img)
        img = self.conv6(img)
        img = F.relu(img)
        img = self.conv7(img)
        img = F.relu(img)

        return img
    

class Discriminator(nn.Module):
    def __init__(self, power):
        super().__init__()

        self.power = power

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 6, 6), padding=(2, 3, 3))
        self.drop1 = nn.Dropout(0.2)

        self.pad1 = nn.ConstantPad3d((1, 1, 0, 0, 0, 0), 0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(5, 6, 6), padding=0)
        self.bn1 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop2 = nn.Dropout(0.2)

        self.pad2 = nn.ConstantPad3d((1, 1, 0, 0, 0, 0), 0)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(5, 6, 6), padding=0)
        self.bn2 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(5, 6, 6), padding=0)
        self.bn3 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop4 = nn.Dropout(0.2)

        self.avgpool = nn.AvgPool3d((2, 2, 2))
        self.flatten = nn.Flatten()

        self.fakeout = nn.Linear(19152, 1) # The input features for the Linear layer need to be calculated based on the output shape from the previous layers.
        self.auxout = nn.Linear(19152, 1) # The same as above for this layer.

    # calculate sum of intensities
    def ecal_sum(self, image, daxis):
        sum = torch.sum(image, dim=daxis)
        return sum

    # angle calculation
    def ecal_angle(self, image, daxis1):
        image = torch.squeeze(image, dim=daxis1)  # squeeze along channel axis

        # get shapes
        x_shape = image.shape[1]
        y_shape = image.shape[2]
        z_shape = image.shape[3]
        sumtot = torch.sum(image, dim=(1, 2, 3))  # sum of events

        # get 1. where event sum is 0 and 0 elsewhere
        amask = torch.where(sumtot == 0.0, torch.ones_like(sumtot), torch.zeros_like(sumtot))
        masked_events = torch.sum(amask)  # counting zero sum events

        # ref denotes barycenter as that is our reference point
        x_ref = torch.sum(torch.sum(image, dim=(2, 3)) 
            * (torch.arange(x_shape, device=image.device, dtype=torch.float32).unsqueeze(0) + 0.5), 
            dim=1,) # sum for x position * x index
        y_ref = torch.sum(
            torch.sum(image, dim=(1, 3))
            * (torch.arange(y_shape, device=image.device, dtype=torch.float32).unsqueeze(0) + 0.5),
            dim=1,)
        z_ref = torch.sum(
            torch.sum(image, dim=(1, 2))
            * (torch.arange(z_shape, device=image.device, dtype=torch.float32).unsqueeze(0) + 0.5),
            dim=1,)

        x_ref = torch.where(sumtot == 0.0, torch.ones_like(x_ref), x_ref / sumtot) # return max position if sumtot=0 and divide by sumtot otherwise
        y_ref = torch.where(sumtot == 0.0, torch.ones_like(y_ref), y_ref / sumtot)
        z_ref = torch.where(sumtot == 0.0, torch.ones_like(z_ref), z_ref / sumtot)

        # reshape
        x_ref = x_ref.unsqueeze(1)
        y_ref = y_ref.unsqueeze(1)
        z_ref = z_ref.unsqueeze(1)

        sumz = torch.sum(image, dim=(1, 2))  # sum for x,y planes going along z

        # Get 0 where sum along z is 0 and 1 elsewhere
        zmask = torch.where(sumz == 0.0, torch.zeros_like(sumz), torch.ones_like(sumz))

        x = torch.arange(x_shape, device=image.device).unsqueeze(0)  # x indexes
        x = (x.unsqueeze(2).float()) + 0.5
        y = torch.arange(y_shape, device=image.device).unsqueeze(0)  # y indexes
        y = (y.unsqueeze(2).float()) + 0.5

        # barycenter for each z position
        x_mid = torch.sum(torch.sum(image, dim=2) * x, dim=1)
        y_mid = torch.sum(torch.sum(image, dim=1) * y, dim=1)

        x_mid = torch.where(sumz == 0.0, torch.zeros_like(sumz), x_mid / sumz) # if sum != 0 then divide by sum
        y_mid = torch.where(sumz == 0.0, torch.zeros_like(sumz), y_mid / sumz) # if sum != 0 then divide by sum

        # Angle Calculations
        z = (torch.arange(z_shape, device=image.device, dtype=torch.float32) + 0.5) * torch.ones_like(z_ref) # Make an array of z indexes for all events

        zproj = torch.sqrt(
            torch.max((x_mid - x_ref) ** 2.0 + (z - z_ref) ** 2.0, torch.tensor([torch.finfo(torch.float32).eps]).to(x_mid.device))) # projection from z axis with stability check
        #torch.finfo(torch.float32).eps))
        m = torch.where(zproj == 0.0, torch.zeros_like(zproj), (y_mid - y_ref) / zproj) # to avoid divide by zero for zproj =0
        m = torch.where(z < z_ref, -1 * m, m)  # sign inversion
        ang = (math.pi / 2.0) - torch.atan(m)  # angle correction
        zmask = torch.where(zproj == 0.0, torch.zeros_like(zproj), zmask)
        ang = ang * zmask  # place zero where zsum is zero
        ang = ang * z  # weighted by position
        sumz_tot = z * zmask  # removing indexes with 0 energies or angles

        # zunmasked = K.sum(zmask, axis=1) # used for simple mean
        # ang = K.sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

        ang = torch.sum(ang, dim=1) / torch.sum(sumz_tot, dim=1) # sum ( measured * weights)/sum(weights)
        ang = torch.where(amask == 0.0, ang, 100.0 * torch.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
        ang = ang.unsqueeze(1)
        return ang

    def forward(self, x):
        z = self.conv1(x)
        z = F.leaky_relu(z)
        z = self.drop1(z)
        z = self.pad1(z)
        z = self.conv2(z)
        z = F.leaky_relu(z)
        z = self.bn1(z)
        z = self.drop2(z)
        z = self.pad2(z)
        z = self.conv3(z)
        z = F.leaky_relu(z)
        z = self.bn2(z)
        z = self.drop3(z)
        z = self.conv4(z)
        z = F.leaky_relu(z)
        z = self.bn3(z)
        z = self.drop4(z)
        z = self.avgpool(z)
        z = self.flatten(z)

        fake = torch.sigmoid(self.fakeout(z)) #generation output that says fake/real
        aux = self.auxout(z) #auxiliary output
        inv_image = x.pow(1.0 / self.power)
        ang = self.ecal_angle(inv_image, 1) # angle calculation
        ecal = self.ecal_sum(inv_image, (2, 3, 4)) # sum of energies

        return fake, aux, ang, ecal
    
