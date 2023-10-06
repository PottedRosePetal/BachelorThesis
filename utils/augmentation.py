import torch
from pytorch3d.transforms import euler_angles_to_matrix



PI_3_2 = torch.pi*3/2
PI_1_2 = torch.pi/2
PI_2 = torch.pi*2
PI = torch.pi

class AugmentData():
    def __init__(self, args) -> None:
        self.deactivate_x = args.deactivate_x
        self.deactivate_y = args.deactivate_y
        self.deactivate_z = args.deactivate_z

        self.rotation_matrix_difference_threshold = 0.1
        self.do_rotation_check = False

        self.device = args.device 
        self.ensure_zero = args.ensure_zero
        self.num_disc_angles = args.num_disc_angles
        assert self.num_disc_angles > 0
        self.jitter_percentage = args.jitter_percentage
        self.angle_deadzone = args.angle_deadzone
        assert self.angle_deadzone < PI
        self.angle_tensor, self.rotation_matrices = self.create_rotation_matrices()
        self.augmentation_count = args.aug_multiplier

    def check_unique_rotations(self, rotation_matrices, angle_tensor):
        differences = []
        def radians_to_degrees(angles_radians):
            return angles_radians * 180.0 / torch.pi

        for i,rotaion_matrix_1 in enumerate(rotation_matrices):
            for ii,rotaion_matrix_2 in enumerate(rotation_matrices):
                matrix_difference = rotaion_matrix_1 - rotaion_matrix_2
                frobenius_norm = torch.norm(matrix_difference, 'fro')
                differences.append((i,ii,frobenius_norm))
        
        for difference in differences:
            if 0 < difference[2] < self.rotation_matrix_difference_threshold:
                print(radians_to_degrees(angle_tensor[difference[0]]), radians_to_degrees(angle_tensor[difference[1]]))
                print(rotation_matrices[difference[0]])
                print(rotation_matrices[difference[1]])
                raise ValueError(f"Two rotation Matrices seem to be almost identical. Difference is: {difference[2]}")


    def create_rotation_matrices(self) -> tuple[torch.Tensor,torch.Tensor]:
        angle_tensor = self.gen_roll_pitch_yaw()
        rotation_matrices = euler_angles_to_matrix(angle_tensor, "XYZ")
        if self.do_rotation_check:
            #may take very long for higher values of num_disc_angles
            self.check_unique_rotations(rotation_matrices,angle_tensor)
        return angle_tensor, rotation_matrices

    def get_rand_rot_matrix(self, batch_size) -> torch.Tensor:
        random_indices = torch.randint(0, self.rotation_matrices.shape[0], (batch_size,))
        self.angles = self.angle_tensor[random_indices]
        rand_rotation_matrices = self.rotation_matrices[random_indices]
        return rand_rotation_matrices

    def apply_augmentation(self, pointclouds:torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a batch of point clouds.

        Args:
            pointclouds (torch.Tensor): Input tensor representing a batch of point clouds.
                Shape: (batch_size, num_points, point_dim).
            rotations (torch.Tensor): Input tensor representing rotation angles for each point cloud.
                Shape: (batch_size, 3), where each row specifies rotation angles around X, Y, and Z axes.
            it (int): Iteration the augmentation is applied on.

        Returns:
            torch.Tensor: Augmented point clouds after rotation and jitter.
                Shape: Same as input 'pointclouds'.
        """
        pointclouds = pointclouds.to(self.device)

        center = torch.mean(pointclouds, dim=1)
        centered_pointcloud = pointclouds - center[:, None, :]

        rotation_matrices = self.get_rand_rot_matrix(pointclouds.shape[0]).to(self.device)
        rotated_pointcloud = torch.bmm(centered_pointcloud, rotation_matrices)

        noise = torch.randn(rotated_pointcloud.size(), device=self.device) * self.jitter_percentage
        rotated_pointcloud += noise
        
        return rotated_pointcloud

    def augment_data(self, data:dict, it:int) -> dict:
        new_id = data["id"] + torch.tensor([float(f".{i+1}") for i in range(data["id"].size(0))])
        augmented_pointcloud_tensor = self.apply_augmentation(data['pointcloud'])

        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(jittered_pointcloud_tensor[0,:,:].numpy())
        # o3d.visualization.draw_geometries([pointcloud])
        # exit()
        return {
            'pointcloud': augmented_pointcloud_tensor, 
            'cate': data["cate"], 
            'id':  new_id,
            'shift': None, 
            'scale': None,
            'angle': self.angles
        }

    def gen_pitch(self):
        if self.num_disc_angles % 2 == 0:
            pitch = torch.linspace(0, torch.pi, self.num_disc_angles)
        else:
            pitch = torch.linspace(0, torch.pi, self.num_disc_angles+1)[:-1]
        if PI_1_2 in pitch:
            raise ValueError()
        
        # selects angles in quadrant 2 and 3 of unit circle
        pitch_2_3 = pitch[(pitch > PI_1_2) & (pitch < PI_3_2)]

        if pitch_2_3.max() != pitch_2_3.min():  #avoids infinite scaling factor, might also be a problem with 2 disc angles
            lower_bound_2_3 = PI_1_2 + self.angle_deadzone/2
            upper_bound_2_3 = PI#_3_2 - self.angle_deadzone/2
            scaling_factor_2_3 = (upper_bound_2_3 - lower_bound_2_3) / (pitch_2_3.max() - pitch_2_3.min())

            #scales points on unit circle to avoid deadzone
            if (pitch_2_3.min() <= lower_bound_2_3 or pitch_2_3.max() >= upper_bound_2_3):
                pitch_2_3 = (pitch_2_3 - pitch_2_3.min()) * scaling_factor_2_3 + lower_bound_2_3
        else:
            pitch_2_3 = torch.ones_like(pitch_2_3) * PI

        # selects angles in quadrant 2 and 3 of unit circle
        pitch_1_4 = pitch[(pitch < PI_1_2) | (pitch > PI_3_2)]

        #transforms range 0-2pi to -pi-pi
        pitch_1_4 = torch.where(pitch_1_4 > PI, pitch_1_4 - PI_2, pitch_1_4)

        upper_bound_1_4 = PI_1_2 - self.angle_deadzone/2
        lower_bound_1_4 = 0 #-PI_1_2 + self.angle_deadzone/2
        scaling_factor_1_4 = (upper_bound_1_4 - lower_bound_1_4) / (pitch_1_4.max() - pitch_1_4.min())

        #scales points on unit circle to avoid deadzone
        if pitch_1_4.min() <= lower_bound_1_4 or pitch_1_4.max() >= upper_bound_1_4:    
            pitch_1_4 = (pitch_1_4 - pitch_1_4.min()) * scaling_factor_1_4 + lower_bound_1_4

        #transforms range back to 0-2pi
        pitch_1_4 = torch.where(pitch_1_4 < 0, pitch_1_4 + PI_2, pitch_1_4)    
        pitch = torch.cat((pitch_2_3, pitch_1_4)).sort()[0]

        if self.ensure_zero:
            pitch[0] = 0

        return pitch

    def gen_roll_pitch_yaw(self, return_single_tensor = True) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor] | torch.Tensor:
        
        pitch = self.gen_pitch()
        roll = torch.linspace(0, PI, pitch.shape[-1]+1)[:-1]
        yaw = torch.linspace(0, PI, pitch.shape[-1]+1)[:-1]
        if self.deactivate_x:
            roll = torch.zeros_like(roll)
        if self.deactivate_y:
            pitch = torch.zeros_like(pitch)
        if self.deactivate_z:
            yaw = torch.zeros_like(yaw)

        try:
            assert pitch.shape == roll.shape == yaw.shape
            assert pitch.shape[-1] ==  roll.shape[-1] == yaw.shape[-1] == self.num_disc_angles
            assert not torch.isnan(pitch).any()
        except AssertionError as AssErr:
            print(pitch.shape, roll.shape, yaw.shape, self.num_disc_angles)
            raise AssErr

        if return_single_tensor:
            angle_grid = torch.meshgrid(roll, pitch, yaw, indexing='xy')
            angle_tensor = torch.stack(angle_grid, dim=-1).reshape(-1, 3)
            return angle_tensor

        return roll,pitch,yaw

    def rand_angles(self, batch_size:int) -> torch.Tensor:
        random_rotations = torch.randint(0, self.angle_tensor.size(0), (batch_size,))
        return self.angle_tensor[random_rotations]

    def ind_from_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Get indices corresponding to discrete angles.

        Args:
            angle (torch.Tensor): Tensor containing angles in radians. Shape: (batch_size, 3).
            num_disc_angles (int): Number of discrete angles.

        Returns:
            torch.Tensor: Indices corresponding to the angles. Shape: (batch_size, 3).
        """
        roll, pitch, yaw = self.gen_roll_pitch_yaw(False)

        # Calculate absolute differences between angle sets and the angles in roll, pitch, yaw
        diff_roll = torch.abs(roll.view(1, -1) - angle[:, 0].view(-1, 1))
        diff_pitch = torch.abs(pitch.view(1, -1) - angle[:, 1].view(-1, 1))
        diff_yaw = torch.abs(yaw.view(1, -1) - angle[:, 2].view(-1, 1))

        # Find the indices of the minimum absolute differences along each axis (roll, pitch, yaw) for each angle set
        min_indices_roll = torch.argmin(diff_roll, dim=1)
        min_indices_pitch = torch.argmin(diff_pitch, dim=1)
        min_indices_yaw = torch.argmin(diff_yaw, dim=1)

        # Stack the indices for each axis to form the final result
        indices = torch.stack((min_indices_roll, min_indices_pitch, min_indices_yaw), dim=-1)
        return indices
    
    def ind_from_angle_ext(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Get indices corresponding to discrete angles. Each angle gets num_disc_angles dimensions with probabilities, 
        as this creates the ground truth tensor, one gets an 1 and the rest a 0. 

        Args:
            angle (torch.Tensor): Tensor containing angles in radians. Shape: (batch_size, 3).
            num_disc_angles (int): Number of discrete angles.

        Returns:
            torch.Tensor: Indices corresponding to the angles. Shape: (batch_size, 3, 100).
        """
        indices = self.ind_from_angle(angle)

        # Create a one-hot tensor for each angle
        batch_size = indices.size(0)
        one_hot = torch.zeros(batch_size, 3, indices.max().item()+1)
        index = indices.unsqueeze(2)
        one_hot.scatter_(2, index, 1)

        return one_hot
    
    @property
    def disc_angles(self):
        return self.gen_roll_pitch_yaw(False)[0].shape[-1]