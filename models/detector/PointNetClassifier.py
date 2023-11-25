import torch

class CartesianToSphericalLayer(torch.nn.Module):
    def __init__(self):
        super(CartesianToSphericalLayer, self).__init__()

    def forward(self, x):
        # Ensure input tensor has shape (batch_size, num_points, 3)
        if x.size(-1) != 3:
            raise ValueError("Input tensor must have shape (batch_size, num_points, 3)")
        # Extract Cartesian coordinates
        x_cartesian = x
        # Calculate spherical coordinates
        r = torch.norm(x_cartesian, p=2, dim=2)
        theta = torch.acos(x_cartesian[:,:,2] / r)
        phi = torch.atan2(x_cartesian[:,:,1], x_cartesian[:,:,0])
        # Stack spherical coordinates (r, theta, phi)

        return torch.stack((r, theta, phi), dim=-1)

class PointNetClassifier(torch.nn.Module):
    def __init__(self, num_points, num_features, num_classes, batchnorm):
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        self.num_features = num_features
        self.batchnorm = batchnorm

        self.conv1 = torch.nn.Conv1d(self.num_features, 128, 3)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128, 256, 3)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.conv3 = torch.nn.Conv1d(256, 512, 3)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)
        self.bn4 = torch.nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.bn1(self.conv1(x)) if not self.batchnorm else self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(self.conv2(x)) if not self.batchnorm else self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(self.conv3(x)) if not self.batchnorm else self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.bn4(self.conv4(x)) if not self.batchnorm else self.conv4(x)
        x = torch.nn.functional.relu(x)

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
class PointNetAngularClassifier(torch.nn.Module):
    def __init__(self, num_points, point_dim, num_disc_angles, batchnorm, spher_coord):
        super(PointNetAngularClassifier, self).__init__()
        self.batchnorm = batchnorm
        self.spherical_coordinates = spher_coord
        self.coord_transform = CartesianToSphericalLayer()

        self.conv1 = torch.nn.Conv1d(point_dim, 128, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(128, 256, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(256, 512, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.bn4 = torch.nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)

        self.fc_x = torch.nn.Linear(256, num_disc_angles)
        self.fc_y = torch.nn.Linear(256, num_disc_angles)
        self.fc_z = torch.nn.Linear(256, num_disc_angles)
                                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord_transform(x) if self.spherical_coordinates else x
        x = x.permute(0, 2, 1)
        x = self.bn1(self.conv1(x)) if self.batchnorm else self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(self.conv2(x)) if self.batchnorm else self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(self.conv3(x)) if self.batchnorm else self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.bn4(self.conv4(x)) if self.batchnorm else self.conv4(x)
        x = torch.nn.functional.relu(x)

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        
        # Predictions for each rotation axis
        pred_x = self.fc_x(x)  # Rotation around X axis
        pred_y = self.fc_y(x)  # Rotation around Y axis
        pred_z = self.fc_z(x)  # Rotation around Z axis

        result = torch.cat([pred_x.unsqueeze(2), pred_y.unsqueeze(2), pred_z.unsqueeze(2)], dim=2)
        result = result.permute(0, 2, 1)
        return result
