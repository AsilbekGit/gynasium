import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class TrainSpeedProfileEnv(gym.Env):
    """
    Custom Gymnasium environment for train speed profile optimization.
    
    Goal: Minimize energy consumption while respecting constraints:
    - Speed limits (already in m/s)
    - Station stops (speed_limit = 1)
    - Arrival time requirements
    - Comfort (acceleration/jerk limits)
    - Safety margins
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        coordinates_file: str,
        data_file: str,
        dt: float = 1.0,  # Time step in seconds
        target_time: Optional[float] = None,  # Target travel time in seconds
        target_time_tolerance: float = 0.05,  # 5% tolerance
        station_dwell_time: float = 30.0,  # Dwell time at stations in seconds
    ):
        super().__init__()
        
        coordinates_file = 'data/coordinates.dat'

        data_file = 'data/data.csv'

        # Load track data
        self.coordinates = pd.read_csv(coordinates_file, header=None, 
                                      names=['node', 'x', 'y'], sep=r'\s+')
        self.track_data = pd.read_csv(data_file)
        
        # Calculate distances between nodes
        self._calculate_distances()
        
        # Identify station stops (speed_limit = 1)
        self._identify_stations()
        
        # Train specifications (ER9E 6-car composition)
        self.train_specs = {
            'mass': 360000,  # kg (360 tons)
            'max_speed': 130 / 3.6,  # Convert km/h to m/s (36.11 m/s)
            'max_power': 3640000,  # W (3640 kW)
            'max_acceleration': 2.16 / 3.6,  # Convert km/h/s to m/s² (0.6 m/s²)
            'max_deceleration': 2.88 / 3.6,  # Convert km/h/s to m/s² (0.8 m/s²)
            'energy_efficiency': 0.85,  # Motor efficiency
            'regenerative_efficiency': 0.65,  # Regenerative braking efficiency
            'min_radius': 100,  # Minimum turning radius (m)
        }
        
        self.dt = dt  # Time step
        self.target_time = target_time
        self.target_time_tolerance = target_time_tolerance
        self.station_dwell_time = station_dwell_time
        
        # State space: [position_idx, speed, distance_to_end, current_grade, 
        #                current_curvature, speed_limit, time_elapsed, 
        #                distance_to_next_station, at_station_flag]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -10, 0, 0, 0, 0, 0]),
            high=np.array([len(self.track_data), 
                          self.train_specs['max_speed'], 
                          self.total_distance,
                          10,  # Max grade %
                          1,   # Max curvature %
                          self.train_specs['max_speed'],
                          np.inf,  # time
                          self.total_distance,  # distance to station
                          1]),  # at_station flag
            dtype=np.float32
        )
        
        # Action space: target acceleration (m/s²)
        # Continuous action between max deceleration and max acceleration
        self.action_space = spaces.Box(
            low=-self.train_specs['max_deceleration'],
            high=self.train_specs['max_acceleration'],
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _calculate_distances(self):
        """Calculate cumulative distances along the track."""
        coords = self.coordinates[['x', 'y']].values
        distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        self.segment_distances = distances
        self.cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        self.total_distance = self.cumulative_distances[-1]
    
    def _identify_stations(self):
        """Identify station locations where speed_limit = 1."""
        # Speed limit = 1 indicates a station stop
        station_mask = self.track_data['Speed_limit'] == 1
        self.station_indices = self.track_data[station_mask].index.tolist()
        self.station_positions = [self.cumulative_distances[idx] for idx in self.station_indices]
        
        print(f"Found {len(self.station_positions)} stations at positions: {self.station_positions}")
    
    def _get_segment_info(self, position: float) -> Dict[str, float]:
        """Get track information at current position."""
        idx = np.searchsorted(self.cumulative_distances, position) - 1
        idx = np.clip(idx, 0, len(self.track_data) - 1)
        
        segment = self.track_data.iloc[idx]
        
        # Check if this is a station (speed_limit = 1)
        is_station = segment['Speed_limit'] == 1
        
        # If station, actual speed limit is 0 (must stop)
        # Otherwise, use the given speed limit (already in m/s)
        if is_station:
            speed_limit = 0.0
        else:
            speed_limit = segment['Speed_limit']
        
        # Calculate radius from curvature percentage
        # Curvature (%) = (1 / Radius) × 100
        # So Radius = 100 / Curvature(%)
        curvature_pct = segment['Curvature']
        if curvature_pct > 0.001:  # Avoid division by very small numbers
            radius = 100.0 / curvature_pct
        else:
            radius = 10000.0  # Essentially straight track
        
        return {
            'grade': segment['Grade'],  # Already in percentage
            'curvature': curvature_pct,  # In percentage
            'radius': radius,  # Calculated radius in meters
            'speed_limit': speed_limit,  # m/s (0 if station)
            'segment_idx': idx,
            'is_station': is_station
        }
    
    def _get_distance_to_next_station(self, position: float) -> float:
        """Get distance to the next station."""
        future_stations = [pos for pos in self.station_positions if pos > position]
        if future_stations:
            return future_stations[0] - position
        return self.total_distance - position  # No more stations, return distance to end
    
    def _calculate_resistance_force(self, speed: float, grade: float, 
                                   curvature_pct: float, radius: float) -> float:
        """
        Calculate total resistance force on the train.
        
        Components:
        1. Rolling resistance
        2. Air resistance
        3. Grade resistance
        4. Curve resistance (based on radius)
        """
        mass = self.train_specs['mass']
        g = 9.81  # Gravity (m/s²)
        
        # Rolling resistance: F_roll = c_r * m * g
        c_roll = 0.0015  # Rolling resistance coefficient for steel on steel
        F_roll = c_roll * mass * g
        
        # Air resistance: F_air = 0.5 * rho * Cd * A * v²
        rho = 1.225  # Air density (kg/m³)
        Cd = 0.7  # Drag coefficient for trains
        A = 10  # Frontal area (m²) - estimated for 6-car train
        F_air = 0.5 * rho * Cd * A * speed**2
        
        # Grade resistance: F_grade = m * g * sin(theta) ≈ m * g * grade/100
        # Grade is already in percentage
        F_grade = mass * g * (grade / 100)
        
        # Curve resistance based on radius
        # Sharper curves (smaller radius) = more resistance
        # F_curve = k * m * g / radius (inversely proportional to radius)
        if radius < 10000:  # Only apply for curves
            k_curve = 650  # Empirical constant for railway curves
            F_curve = k_curve * mass * g / radius
        else:
            F_curve = 0
        
        total_resistance = F_roll + F_air + F_grade + F_curve
        return total_resistance
    
    def _calculate_energy_consumption(self, force: float, speed: float, 
                                     dt: float) -> float:
        """
        Calculate energy consumption in kWh.
        
        Positive force = traction (consuming energy)
        Negative force = braking (regenerating energy if possible)
        """
        power = force * speed  # Watts
        
        if power > 0:  # Traction
            # Account for motor efficiency
            energy = (power / self.train_specs['energy_efficiency']) * dt / 3600000  # kWh
        else:  # Braking
            # Regenerative braking recovers some energy
            energy = power * self.train_specs['regenerative_efficiency'] * dt / 3600000  # kWh
        
        return energy
    
    def reset(self, seed: Optional[int] = None, 
             options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initial state
        self.position = 0.0  # Start position (m)
        self.speed = 0.0  # Start at rest (m/s)
        self.time_elapsed = 0.0
        self.total_energy = 0.0
        self.max_speed_violation = 0.0
        self.comfort_violation = 0.0
        self.station_stops_completed = []
        self.at_station = False
        self.station_dwell_remaining = 0.0
        
        # Get initial segment info
        segment_info = self._get_segment_info(self.position)
        
        observation = self._get_observation(segment_info)
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self, segment_info: Dict) -> np.ndarray:
        """Construct observation vector."""
        distance_to_next_station = self._get_distance_to_next_station(self.position)
        
        return np.array([
            segment_info['segment_idx'],
            self.speed,
            self.total_distance - self.position,
            segment_info['grade'],
            segment_info['curvature'],
            segment_info['speed_limit'],
            self.time_elapsed,
            distance_to_next_station,
            1.0 if self.at_station else 0.0
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'position': self.position,
            'speed': self.speed,
            'time_elapsed': self.time_elapsed,
            'total_energy': self.total_energy,
            'distance_remaining': self.total_distance - self.position,
            'stations_completed': len(self.station_stops_completed),
            'at_station': self.at_station,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step."""
        
        # Get current segment information
        segment_info = self._get_segment_info(self.position)
        
        # Check if we're at a station and need to dwell
        if segment_info['is_station'] and self.speed < 0.5:
            if not self.at_station:
                # Just arrived at station
                self.at_station = True
                self.station_dwell_remaining = self.station_dwell_time
                current_station_idx = segment_info['segment_idx']
                if current_station_idx not in self.station_stops_completed:
                    self.station_stops_completed.append(current_station_idx)
            
            # Dwell at station
            if self.station_dwell_remaining > 0:
                self.station_dwell_remaining -= self.dt
                self.time_elapsed += self.dt
                self.speed = 0.0
                
                reward = -0.1  # Small penalty for time spent
                observation = self._get_observation(segment_info)
                info = self._get_info()
                
                return observation, reward, False, False, info
        else:
            self.at_station = False
            self.station_dwell_remaining = 0.0
        
        target_acceleration = float(action[0])
        
        # If approaching a station, enforce deceleration
        if segment_info['is_station'] and self.speed > 0.1:
            # Force braking to stop at station
            target_acceleration = min(target_acceleration, -0.3)
        
        # Calculate resistance force
        F_resistance = self._calculate_resistance_force(
            self.speed,
            segment_info['grade'],
            segment_info['curvature'],
            segment_info['radius']
        )
        
        # Calculate required traction force for target acceleration
        F_traction = (self.train_specs['mass'] * target_acceleration + 
                     F_resistance)
        
        # Limit traction force by available power: P = F * v
        if self.speed > 0.1:  # Avoid division by zero
            max_traction = self.train_specs['max_power'] / self.speed
            F_traction = np.clip(F_traction, -np.inf, max_traction)
        
        # Calculate actual acceleration
        actual_acceleration = (F_traction - F_resistance) / self.train_specs['mass']
        
        # Limit acceleration to physical constraints
        actual_acceleration = np.clip(
            actual_acceleration,
            -self.train_specs['max_deceleration'],
            self.train_specs['max_acceleration']
        )
        
        # Update speed and position
        new_speed = self.speed + actual_acceleration * self.dt
        
        # Enforce speed limits
        if segment_info['speed_limit'] > 0:
            new_speed = np.clip(new_speed, 0, min(segment_info['speed_limit'], 
                                                   self.train_specs['max_speed']))
        else:
            new_speed = np.clip(new_speed, 0, self.train_specs['max_speed'])
        
        # Average speed during time step for position update
        avg_speed = (self.speed + new_speed) / 2
        self.position += avg_speed * self.dt
        self.speed = new_speed
        
        # Calculate energy consumption
        energy = self._calculate_energy_consumption(F_traction, avg_speed, self.dt)
        self.total_energy += energy
        
        # Update time
        self.time_elapsed += self.dt
        
        # Check constraints and calculate reward
        speed_limit_violation = 0
        if segment_info['speed_limit'] > 0:
            speed_limit_violation = max(0, self.speed - segment_info['speed_limit'])
        self.max_speed_violation = max(self.max_speed_violation, speed_limit_violation)
        
        # Comfort constraint (jerk - rate of change of acceleration)
        if hasattr(self, 'prev_acceleration'):
            jerk = abs(actual_acceleration - self.prev_acceleration) / self.dt
            comfort_penalty = max(0, jerk - 1.0)  # Jerk limit: 1 m/s³
            self.comfort_violation += comfort_penalty
        self.prev_acceleration = actual_acceleration
        
        # Calculate reward
        reward = self._calculate_reward(
            energy,
            speed_limit_violation,
            segment_info,
            actual_acceleration
        )
        
        # Check if episode is done
        terminated = self.position >= self.total_distance
        truncated = self.time_elapsed > 7200  # Max 2 hours
        
        # Final reward adjustments
        if terminated:
            # Check if all stations were visited
            if len(self.station_stops_completed) < len(self.station_positions):
                reward -= 2000 * (len(self.station_positions) - len(self.station_stops_completed))
            
            # Time penalty/bonus
            if self.target_time is not None:
                time_error = abs(self.time_elapsed - self.target_time) / self.target_time
                if time_error > self.target_time_tolerance:
                    reward -= 1000 * time_error
            
            # Penalize if didn't come to complete stop at end
            if self.speed > 0.5:
                reward -= 500
            
            # Bonus for completing successfully
            reward += 1000
        
        # Get new observation
        observation = self._get_observation(segment_info)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, energy: float, speed_violation: float,
                         segment_info: Dict, acceleration: float) -> float:
        """
        Calculate reward function.
        
        Reward components:
        1. Energy efficiency (primary objective)
        2. Speed limit compliance
        3. Comfort (smooth acceleration)
        4. Progress towards goal
        5. Station stop compliance
        """
        # 1. Energy cost (negative reward for energy consumption)
        energy_cost = -energy * 100  # Scale factor
        
        # 2. Speed limit violation penalty
        speed_penalty = -speed_violation**2 * 1000
        
        # 3. Comfort penalty (penalize high acceleration changes)
        comfort_penalty = -abs(acceleration) * 0.1
        
        # 4. Progress reward (small positive reward for moving forward)
        progress_reward = self.speed * 0.01
        
        # 5. Station approach reward
        # Encourage proper deceleration when approaching stations
        distance_to_next = self._get_distance_to_next_station(self.position)
        if distance_to_next < 200 and self.speed > 5:
            # Should be slowing down
            station_penalty = -0.5
        else:
            station_penalty = 0
        
        # Total reward
        reward = (energy_cost + speed_penalty + comfort_penalty + 
                 progress_reward + station_penalty)
        
        return reward
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == 'human':
            print(f"Position: {self.position:.1f}m / {self.total_distance:.1f}m")
            print(f"Speed: {self.speed*3.6:.1f} km/h ({self.speed:.2f} m/s)")
            print(f"Time: {self.time_elapsed:.1f}s")
            print(f"Energy: {self.total_energy:.4f} kWh")
            print(f"Stations completed: {len(self.station_stops_completed)}/{len(self.station_positions)}")
            if self.at_station:
                print(f"AT STATION - Dwell remaining: {self.station_dwell_remaining:.1f}s")
            print("-" * 50)


# Example usage and training script
if __name__ == "__main__":
    from stable_baselines3 import PPO, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    # Create environment
    env = TrainSpeedProfileEnv(
        coordinates_file='data/coordinates.dat',
        data_file='data/data.csv',
        dt=1.0,  # 1 second time steps
        target_time=None,  # Let agent optimize freely
        station_dwell_time=30.0,  # 30 seconds at each station
    )
    
    # Wrap environment
    env = DummyVecEnv([lambda: env])
    
    # Choose algorithm: PPO (good for continuous control)
    print("Training with PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./train_rl_logs/"
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./train_models/',
        name_prefix='train_speed_model'
    )
    
    # Train the model
    total_timesteps = 500000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("train_speed_optimizer_final")
    
    print("\nTraining complete!")
    print("\nTesting the trained model...")
    
    # Test the trained model
    obs = env.reset()
    episode_reward = 0
    episode_energy = 0
    
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        episode_energy = info[0]['total_energy']
        
        if done:
            break
    
    print(f"\nTest Episode Results:")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Total Energy: {episode_energy:.4f} kWh")
    print(f"Total Time: {info[0]['time_elapsed']:.1f} seconds")
    print(f"Stations Completed: {info[0]['stations_completed']}")
    print(f"Average Speed: {(info[0]['position'] / info[0]['time_elapsed']) * 3.6:.1f} km/h")