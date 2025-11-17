from rain_autonomous_forklift_det.utils.decorator import measure_time
import small_gicp
import numpy as np

class ScanToScanMatchingOdometry():
    def __init__(self, num_threads, voxel_size, map_clear_cycle):
        self._num_threads = num_threads
        self._T_last_current = np.identity(4)
        self._T_world_lidar = np.identity(4)
        self._target = None
        self._target_map = small_gicp.GaussianVoxelMap(voxel_size)
        self._target_map.set_lru(horizon=100, clear_cycle=map_clear_cycle)
  
    @measure_time('Small-GICP estimation')
    def estimate(self, raw_points):
        downsampled, tree = small_gicp.preprocess_points(raw_points, 0.05, num_threads=self._num_threads)
        
        if self._target is None:
          self._target = (downsampled, tree)
          self._target_map.insert(downsampled)
          return self._T_world_lidar
    
        result = small_gicp.align(self._target[0], downsampled, self._target[1], self._T_last_current, num_threads=self._num_threads, max_iterations = 50)
        
        self._T_last_current = result.T_target_source
        self._T_world_lidar = self._T_world_lidar @ result.T_target_source
        self._target = (downsampled, tree)
    
        self._target_map.insert(downsampled, self._T_world_lidar)
        
        return self._T_world_lidar
    
    def get_map_points(self):
        return self._target_map.voxel_points()[:,:3]
  
