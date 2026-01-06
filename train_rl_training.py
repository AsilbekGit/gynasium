"""
Training and Analysis Script for Train Speed Profile Optimization
This script provides utilities for training, evaluating, and comparing RL agents
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from typing import List, Dict
import json


class TrainingCallback(BaseCallback):
    """
    Custom callback for logging training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_energies = []
        self.episode_times = []
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_rewards.append(self.locals.get('rewards')[0])
            self.episode_energies.append(info.get('total_energy', 0))
            self.episode_times.append(info.get('time_elapsed', 0))
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_energy = np.mean(self.episode_energies[-10:])
                print(f"Episodes: {len(self.episode_rewards)}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Energy: {avg_energy:.4f} kWh")
        
        return True


def train_agent(
    env,
    algorithm='PPO',
    total_timesteps=500000,
    save_path='./models/',
    log_path='./logs/',
    hyperparameters=None
):
    """
    Train an RL agent with specified algorithm and hyperparameters.
    
    Args:
        env: Gymnasium environment
        algorithm: 'PPO', 'DDPG', or 'SAC'
        total_timesteps: Total training timesteps
        save_path: Path to save models
        log_path: Path for tensorboard logs
        hyperparameters: Dict of algorithm-specific hyperparameters
    
    Returns:
        Trained model
    """
    import os
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Optionally normalize observations
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {}
    
    # Create model based on algorithm
    if algorithm == 'PPO':
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
        }
        default_params.update(hyperparameters)
        model = PPO("MlpPolicy", env, verbose=1, 
                   tensorboard_log=log_path, **default_params)
    
    elif algorithm == 'DDPG':
        default_params = {
            'learning_rate': 1e-3,
            'buffer_size': 200000,
            'learning_starts': 100,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
        }
        default_params.update(hyperparameters)
        model = DDPG("MlpPolicy", env, verbose=1,
                    tensorboard_log=log_path, **default_params)
    
    elif algorithm == 'SAC':
        default_params = {
            'learning_rate': 3e-4,
            'buffer_size': 300000,
            'learning_starts': 100,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
        }
        default_params.update(hyperparameters)
        model = SAC("MlpPolicy", env, verbose=1,
                   tensorboard_log=log_path, **default_params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    training_callback = TrainingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix=f'{algorithm}_train_model'
    )
    
    # Train
    print(f"\nTraining {algorithm} agent for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[training_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_path}/{algorithm}_final")
    
    # Save training metrics (convert numpy types to Python types)
    metrics = {
        'rewards': [float(x) for x in training_callback.episode_rewards],
        'energies': [float(x) for x in training_callback.episode_energies],
        'times': [float(x) for x in training_callback.episode_times]
    }
    
    with open(f"{save_path}/{algorithm}_metrics.json", 'w') as f:
        json.dump(metrics, f)
    
    return model, metrics


def evaluate_agent(model, env, n_episodes=10):
    """
    Evaluate a trained agent over multiple episodes.
    
    Args:
        model: Trained SB3 model
        env: Unwrapped Gymnasium environment (NOT VecEnv)
        n_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with evaluation metrics (all Python native types)
    """
    rewards = []
    energies = []
    times = []
    speeds = []
    positions = []
    
    for episode in range(n_episodes):
        # Gym API: reset returns (obs, info)
        obs, info = env.reset()
        episode_reward = 0
        episode_speeds = []
        episode_positions = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Model expects shape (1, obs_dim) for prediction
            obs_array = obs.reshape(1, -1)
            action, _ = model.predict(obs_array, deterministic=True)
            
            # Gym API: step returns (obs, reward, terminated, truncated, info)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_speeds.append(float(info['speed']))
            episode_positions.append(float(info['position']))
        
        rewards.append(float(episode_reward))
        energies.append(float(info['total_energy']))
        times.append(float(info['time_elapsed']))
        speeds.append(episode_speeds)
        positions.append(episode_positions)
    
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'speeds': speeds,
        'positions': positions,
        'rewards': rewards,
        'energies': energies,
        'times': times
    }


def compare_baseline(env, n_episodes=10):
    """
    Create a simple baseline by maintaining constant speed.
    This helps evaluate if RL agent actually learned something useful.
    Returns all values as Python native types.
    
    Args:
        env: Unwrapped Gymnasium environment (NOT VecEnv)
    """
    baseline_energies = []
    baseline_times = []
    
    for _ in range(n_episodes):
        # Gym API: reset returns (obs, info)
        obs, info = env.reset()
        total_energy = 0
        
        # Simple strategy: maintain speed near average speed limit
        target_speed = 25  # m/s (~90 km/h)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            current_speed = obs[1]
            # Simple proportional control
            if current_speed < target_speed:
                action = np.array([0.3])  # Accelerate
            else:
                action = np.array([-0.1])  # Light braking
            
            obs, reward, done, truncated, info = env.step(action)
            total_energy = info['total_energy']
        
        baseline_energies.append(float(total_energy))
        baseline_times.append(float(info['time_elapsed']))
    
    return {
        'mean_energy': float(np.mean(baseline_energies)),
        'std_energy': float(np.std(baseline_energies)),
        'mean_time': float(np.mean(baseline_times)),
        'std_time': float(np.std(baseline_times))
    }


def plot_speed_profile(positions, speeds, track_data, cumulative_distances, save_path=None):
    """
    Plot the speed profile along with track characteristics.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Identify station positions
    station_mask = track_data['Speed_limit'] == 1
    station_positions = cumulative_distances[station_mask]
    
    # Speed profile
    speeds_kmh = np.array(speeds) * 3.6
    speeds_ms = np.array(speeds)
    axes[0].plot(positions, speeds_kmh, 'b-', linewidth=2, label='Train Speed (km/h)')
    axes[0].plot(positions, speeds_ms, 'g--', linewidth=1.5, alpha=0.7, label='Train Speed (m/s)')
    
    # Mark stations
    for station_pos in station_positions:
        axes[0].axvline(x=station_pos, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Plot speed limits
    speed_limits_ms = track_data['Speed_limit'].values
    speed_limits_ms[speed_limits_ms == 1] = 0  # Replace station markers with 0
    axes[0].plot(cumulative_distances[:-1], speed_limits_ms * 3.6, 'orange', 
                linewidth=1.5, alpha=0.6, label='Speed Limit (km/h)', linestyle='-.')
    
    axes[0].set_ylabel('Speed (km/h) / (m/s)')
    axes[0].set_title('Train Speed Profile with Stations (red dashed lines)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    
    # Grade profile
    axes[1].plot(cumulative_distances[:-1], track_data['Grade'], 'g-', linewidth=2)
    axes[1].fill_between(cumulative_distances[:-1], 0, track_data['Grade'], 
                         where=(track_data['Grade'] > 0), alpha=0.3, color='red', label='Uphill')
    axes[1].fill_between(cumulative_distances[:-1], 0, track_data['Grade'], 
                         where=(track_data['Grade'] < 0), alpha=0.3, color='blue', label='Downhill')
    for station_pos in station_positions:
        axes[1].axvline(x=station_pos, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[1].set_ylabel('Grade (%)')
    axes[1].set_title('Track Grade (Positive = Uphill, Negative = Downhill)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].legend()
    
    # Curvature profile
    axes[2].plot(cumulative_distances[:-1], track_data['Curvature'], 'r-', linewidth=2)
    for station_pos in station_positions:
        axes[2].axvline(x=station_pos, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[2].set_ylabel('Curvature (%)')
    axes[2].set_title('Track Curvature (Higher = Sharper Curve)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Radius profile (calculated from curvature)
    # Radius = 100 / Curvature(%)
    radius = np.where(track_data['Curvature'] > 0.001, 
                     100.0 / track_data['Curvature'], 
                     10000)
    axes[3].plot(cumulative_distances[:-1], radius, 'purple', linewidth=2)
    for station_pos in station_positions:
        axes[3].axvline(x=station_pos, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[3].set_ylabel('Radius (m)')
    axes[3].set_xlabel('Position (m)')
    axes[3].set_title('Curve Radius (Smaller = Sharper Turn)')
    axes[3].set_ylim([0, 5000])  # Limit y-axis for better visibility
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_progress(metrics_file):
    """
    Plot training metrics over time.
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Rewards
    episodes = range(len(metrics['rewards']))
    axes[0].plot(episodes, metrics['rewards'], alpha=0.6)
    
    # Smooth with moving average
    window = 50
    if len(metrics['rewards']) > window:
        smoothed = pd.Series(metrics['rewards']).rolling(window).mean()
        axes[0].plot(episodes, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Progress: Rewards')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Energy consumption
    axes[1].plot(episodes, metrics['energies'], alpha=0.6)
    
    if len(metrics['energies']) > window:
        smoothed = pd.Series(metrics['energies']).rolling(window).mean()
        axes[1].plot(episodes, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Energy Consumption (kWh)')
    axes[1].set_title('Training Progress: Energy Efficiency')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def hyperparameter_search(env, algorithm='PPO', n_trials=10):
    """
    Simple random search for hyperparameters.
    """
    best_energy = float('inf')
    best_params = None
    results = []
    
    for trial in range(n_trials):
        print(f"\n=== Trial {trial + 1}/{n_trials} ===")
        
        # Random hyperparameters
        if algorithm == 'PPO':
            params = {
                'learning_rate': 10 ** np.random.uniform(-4, -2),
                'n_steps': np.random.choice([1024, 2048, 4096]),
                'batch_size': np.random.choice([32, 64, 128]),
                'n_epochs': np.random.choice([5, 10, 20]),
                'gamma': np.random.uniform(0.95, 0.999),
                'clip_range': np.random.uniform(0.1, 0.3),
            }
        
        # Train with these parameters
        model, metrics = train_agent(
            env,
            algorithm=algorithm,
            total_timesteps=100000,  # Shorter for search
            hyperparameters=params
        )
        
        # Evaluate
        eval_results = evaluate_agent(model, env, n_episodes=5)
        
        results.append({
            'params': params,
            'energy': eval_results['mean_energy'],
            'time': eval_results['mean_time'],
            'reward': eval_results['mean_reward']
        })
        
        if eval_results['mean_energy'] < best_energy:
            best_energy = eval_results['mean_energy']
            best_params = params
            print(f"New best! Energy: {best_energy:.4f} kWh")
    
    return best_params, results


# Example usage
if __name__ == "__main__":
    from train_speed_env import TrainSpeedProfileEnv
    
    # Create environment (unwrapped)
    env_unwrapped = TrainSpeedProfileEnv(
        coordinates_file='data/coordinates.dat',
        data_file='data/data.csv',
        dt=1.0
    )
    
    # Option 1: Train a single agent
    print("Training PPO agent...")
    model, metrics = train_agent(
        env_unwrapped,  # Pass unwrapped env, it will be wrapped inside
        algorithm='PPO',
        total_timesteps=500000,
        save_path='./models/',
        log_path='./logs/'
    )
    
    # Evaluate (use unwrapped env)
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(model, env_unwrapped, n_episodes=10)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Energy: {eval_results['mean_energy']:.4f} ± {eval_results['std_energy']:.4f} kWh")
    print(f"Mean Time: {eval_results['mean_time']:.1f} ± {eval_results['std_time']:.1f} s")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Compare with baseline (use unwrapped env)
    print("\nBaseline comparison...")
    baseline = compare_baseline(env_unwrapped, n_episodes=10)
    print(f"Baseline Energy: {baseline['mean_energy']:.4f} ± {baseline['std_energy']:.4f} kWh")
    print(f"Energy Savings: {((baseline['mean_energy'] - eval_results['mean_energy']) / baseline['mean_energy'] * 100):.1f}%")
    
    # Plot speed profile
    plot_speed_profile(
        eval_results['positions'][0],
        eval_results['speeds'][0],
        env_unwrapped.track_data,
        env_unwrapped.cumulative_distances,
        save_path='speed_profile.png'
    )
    
    # Plot training progress
    plot_training_progress('./models/PPO_metrics.json')