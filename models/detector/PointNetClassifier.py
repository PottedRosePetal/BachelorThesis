import torch
import torch.nn as nn

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
    def __init__(self, num_points, num_features, num_classes):
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        self.num_features = num_features

        # PointNet-like architecture
        self.conv1 = torch.nn.Conv1d(self.num_features, 128, 3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128, 256, 3)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = torch.nn.Conv1d(256, 512, 3)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x
        
class __PointNetAngularClassifier(torch.nn.Module):
    def __init__(self, num_points, point_dim, num_disc_angles):
        super(PointNetAngularClassifier, self).__init__()

        # Kartesian to Spherical coordinates
        # self.coord_transform = CartesianToSphericalLayer()

        # PointNet-like architecture
        self.conv1 = torch.nn.Conv1d(point_dim, 128, 3)
        self.conv2 = torch.nn.Conv1d(128, 256, 3)
        self.conv3 = torch.nn.Conv1d(256, 512, 3)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        # self.dir_layer = ConcatSquashLinearPCA(64)
        # self.combine = CombinedLinearLayer(1024,64,1024,hidden_layer=128)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)

        # Separate fully connected layers for each rotation axis
        self.fc_x = torch.nn.Linear(256, num_disc_angles)
        self.fc_y = torch.nn.Linear(256, num_disc_angles)
        self.fc_z = torch.nn.Linear(256, num_disc_angles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        # x_dir = torch.nn.functional.relu(self.dir_layer(x))
        # x_dir = x_dir.view(x_dir.size(0), -1)

        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        # x = torch.nn.functional.relu(self.combine(x,x_dir))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        
        # Predictions for each rotation axis
        pred_x = self.fc_x(x)  # Rotation around X axis
        pred_y = self.fc_y(x)  # Rotation around Y axis
        pred_z = self.fc_z(x)  # Rotation around Z axis

        result = torch.cat([pred_x.unsqueeze(2), pred_y.unsqueeze(2), pred_z.unsqueeze(2)], dim=2)
        result = result.permute(0, 2, 1)
        return result

class PointNetAngularClassifier(torch.nn.Module):
    def __init__(self, num_points, point_dim, num_disc_angles):
        super(PointNetAngularClassifier, self).__init__()

        # PointNet-like architecture
        self.conv1 = torch.nn.Conv1d(point_dim, 128, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(128, 256, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(256, 512, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)
        

        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.bn4 = torch.nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)

        # Separate fully connected layers for each rotation axis
        self.fc_x = torch.nn.Linear(256, num_disc_angles)
        self.fc_y = torch.nn.Linear(256, num_disc_angles)
        self.fc_z = torch.nn.Linear(256, num_disc_angles)
        
        # Residual connections
        self.residual = torch.nn.Conv1d(point_dim, 1024, 1)
                        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        
        # Predictions for each rotation axis
        pred_x = self.fc_x(x)  # Rotation around X axis
        pred_y = self.fc_y(x)  # Rotation around Y axis
        pred_z = self.fc_z(x)  # Rotation around Z axis

        result = torch.cat([pred_x.unsqueeze(2), pred_y.unsqueeze(2), pred_z.unsqueeze(2)], dim=2)
        result = result.permute(0, 2, 1)
        print(result.shape)
        return result

class ConcatSquashLinearPCA(torch.nn.Module):
    def __init__(self, dim_out, pca_rank = 128):
        super(ConcatSquashLinearPCA, self).__init__()
        self._pca_linear = torch.nn.Linear(6156, pca_rank)  # Linear layer for principal directions
        self._layer = torch.nn.Linear(pca_rank, dim_out)  # Combined input size

    def forward(self, x: torch.Tensor):
        U, S, V = torch.pca_lowrank(x) # Documentation: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        # Reshape and normalize matrices
        U_flat = U.reshape(x.size(0), -1)
        S_flat = S
        V_flat = V.reshape(x.size(0), -1)

        # Normalize data
        U_normalized = (U_flat - U_flat.mean()) / U_flat.std()
        S_normalized = (S_flat - S_flat.mean()) / S_flat.std()
        V_normalized = (V_flat - V_flat.mean()) / V_flat.std()

        combined_input = torch.cat((U_normalized, S_normalized, V_normalized), dim=1)
        
        pca_projection = torch.nn.functional.relu(self._pca_linear(combined_input))
        ret = torch.nn.functional.relu(self._layer(pca_projection))
        return ret

class CombinedLinearLayer(torch.nn.Module):
    def __init__(self, in_features1, in_features2, out_features, hidden_layer):
        super(CombinedLinearLayer, self).__init__()
        assert hidden_layer%2 == 0
        self.linear1 = torch.nn.Linear(in_features1, hidden_layer//2)
        self.linear2 = torch.nn.Linear(in_features2, hidden_layer//2)
        
        # Additional linear layer to combine the outputs
        self.combining_linear = torch.nn.Linear(hidden_layer, out_features)
        
    def forward(self, x1, x2):
        out1 = torch.nn.functional.relu(self.linear1(x1))
        out2 = torch.nn.functional.relu(self.linear2(x2))
        
        combined_output = torch.cat((out1, out2), dim=-1)
        final_output = torch.nn.functional.relu(self.combining_linear(combined_output))
        return final_output